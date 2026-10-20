"""Consistency critic: internal contradictions + SelfCheck-style agreement with independent samples."""
import asyncio

from app.critics.base import BaseCritic, CriticContext, make_issue
from app.prompts import (CONSISTENCY_HUMAN, CONSISTENCY_SYSTEM, SAMPLER_HUMAN, SAMPLER_SYSTEM, fmt_claims,
                         fmt_evidence, fmt_samples, pick_evidence)
from app.schemas import ConsistencyJudgement, CriticReport

CONTRADICTION_PENALTY = {"low": 5, "medium": 15, "high": 30, "critical": 50}
AGREEMENT_CREDIT = {"consistent": 1.0, "unaddressed": 0.7, "inconsistent": 0.0}


class ConsistencyCritic(BaseCritic):
    name = "consistency"

    async def _sample(self, ctx: CriticContext, k: int) -> list[str]:
        if k <= 0:
            return []
        evidence = fmt_evidence(pick_evidence(ctx.pool, ctx.question_evidence_ids))
        chain = self.hub.text(SAMPLER_SYSTEM, SAMPLER_HUMAN, temperature=0.8)
        results = await asyncio.gather(
            *[self.hub.ainvoke(chain, {"question": ctx.question, "evidence": evidence}) for _ in range(k)],
            return_exceptions=True)
        return [r.strip() for r in results if isinstance(r, str) and r.strip()]

    async def evaluate(self, ctx: CriticContext) -> CriticReport:
        k = ctx.options.selfcheck_samples if ctx.options.selfcheck_samples is not None else self.settings.selfcheck_samples
        samples = await self._sample(ctx, k)

        chain = self.hub.structured(ConsistencyJudgement, CONSISTENCY_SYSTEM, CONSISTENCY_HUMAN)
        j: ConsistencyJudgement = await self.hub.ainvoke(chain, {
            "question": ctx.question, "answer": ctx.answer,
            "claims": fmt_claims(ctx.claims), "samples": fmt_samples(samples),
        })

        issues = []
        penalty = 0
        for c in j.contradictions:
            penalty += CONTRADICTION_PENALTY.get(c.severity, 15)
            issues.append(make_issue(
                self.name, "internal_contradiction", c.severity,
                f'"{c.statement_a}" conflicts with "{c.statement_b}". {c.explanation}',
                suggestion="Decide which statement is correct and fix or remove the other."))
        internal_score = max(0.0, 100.0 - penalty)

        agreement_rows, sample_score = [], None
        if samples and ctx.claims:
            by_id = {a.claim_id: a for a in j.sample_agreement}
            total = 0.0
            for claim in ctx.claims:
                a = by_id.get(claim.id)
                kind = a.agreement if a else "unaddressed"
                total += AGREEMENT_CREDIT[kind]
                agreement_rows.append({"claim_id": claim.id, "claim": claim.text, "agreement": kind,
                                       "note": a.note if a else ""})
                if kind == "inconsistent":
                    issues.append(make_issue(
                        self.name, "unstable_claim", "medium",
                        "Independently sampled answers to the same question disagree with this claim, "
                        "which is a common sign of a hallucinated detail." + (f" {a.note}" if a and a.note else ""),
                        claim, suggestion="Verify this claim against a trusted source before keeping it."))
            sample_score = 100 * total / len(ctx.claims)

        score = internal_score if sample_score is None else 0.5 * internal_score + 0.5 * sample_score
        confidence = j.confidence * (1.0 if samples else 0.7)
        return CriticReport(
            critic=self.name, score=round(score, 1), confidence=round(confidence, 2),
            summary=j.summary or (f"{len(j.contradictions)} internal contradiction(s); "
                                  f"{len(samples)} independent sample(s) compared."),
            issues=issues,
            details={"contradictions": [c.model_dump() for c in j.contradictions],
                     "sample_agreement": agreement_rows, "samples": samples,
                     "internal_score": round(internal_score, 1),
                     "sample_score": None if sample_score is None else round(sample_score, 1)})
