"""Accuracy critic: is each claim TRUE? (evidence + the LLM's own knowledge)."""
from app.critics.base import BaseCritic, CriticContext, make_issue
from app.prompts import ACCURACY_HUMAN, ACCURACY_SYSTEM, fmt_claims, fmt_evidence, pick_evidence
from app.schemas import AccuracyJudgement, CriticReport

# credit given to an "incorrect" claim depending on how bad the error is
INCORRECT_CREDIT = {"low": 0.5, "medium": 0.25, "high": 0.0, "critical": 0.0}


class AccuracyCritic(BaseCritic):
    name = "accuracy"

    async def evaluate(self, ctx: CriticContext) -> CriticReport:
        if not ctx.claims:
            return CriticReport(critic=self.name, evaluable=False,
                                summary="No checkable factual claims were found in the answer.")
        ids = list(dict.fromkeys(ctx.question_evidence_ids + [i for v in ctx.claim_evidence.values() for i in v]))
        evidence = sorted(pick_evidence(ctx.pool, ids), key=lambda e: -e.score)[:8]
        chain = self.hub.structured(AccuracyJudgement, ACCURACY_SYSTEM, ACCURACY_HUMAN)
        j: AccuracyJudgement = await self.hub.ainvoke(chain, {
            "question": ctx.question, "answer": ctx.answer,
            "claims": fmt_claims(ctx.claims), "evidence": fmt_evidence(evidence),
        })

        by_id = {a.claim_id: a for a in j.assessments}
        points, issues, details, uncertain = 0.0, [], [], 0
        for claim in ctx.claims:
            a = by_id.get(claim.id)
            if a is None or a.assessment == "uncertain":
                points += 0.6
                uncertain += 1
                details.append({"claim_id": claim.id, "claim": claim.text, "assessment": "uncertain",
                                "explanation": a.explanation if a else "Not assessed."})
                continue
            if a.assessment == "correct":
                points += 1.0
            else:
                points += INCORRECT_CREDIT.get(a.severity, 0.25)
                fix = f"Correct this to: {a.correction}" if a.correction else "Correct or remove this claim."
                issues.append(make_issue(
                    self.name, "factual_error", a.severity,
                    f"Factually incorrect. {a.explanation}", claim,
                    ctx.claim_evidence.get(claim.id, []), fix))
            details.append({"claim_id": claim.id, "claim": claim.text, "assessment": a.assessment,
                            "severity": a.severity if a.assessment == "incorrect" else None,
                            "correction": a.correction, "explanation": a.explanation})

        n = len(ctx.claims)
        score = round(100 * points / n, 1)
        confidence = round(j.confidence * (1 - 0.5 * uncertain / n), 2)
        return CriticReport(critic=self.name, score=score, confidence=confidence,
                            summary=j.summary or f"{n - uncertain - len(issues)} of {n} claims verified correct.",
                            issues=issues, details={"claims": details})
