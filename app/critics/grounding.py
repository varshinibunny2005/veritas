"""Grounding critic (RAG): is each claim supported by the RETRIEVED reference documents?"""
from app.critics.base import BaseCritic, CriticContext, make_issue
from app.prompts import GROUNDING_HUMAN, GROUNDING_SYSTEM, fmt_claims_with_candidates, fmt_evidence, pick_evidence
from app.schemas import CriticReport, GroundingJudgement

VERDICT_CREDIT = {"supported": 1.0, "partially_supported": 0.6, "unsupported": 0.1, "contradicted": 0.0}
VERDICT_SEVERITY = {"contradicted": "high", "unsupported": "medium", "partially_supported": "low"}


class GroundingCritic(BaseCritic):
    name = "grounding"

    async def evaluate(self, ctx: CriticContext) -> CriticReport:
        if not ctx.claims:
            return CriticReport(critic=self.name, evaluable=False,
                                summary="No checkable factual claims were found in the answer.")
        used_ids = list(dict.fromkeys(i for v in ctx.claim_evidence.values() for i in v))
        if not used_ids:
            return CriticReport(
                critic=self.name, evaluable=False, confidence=0.0,
                summary="No relevant reference documents were retrieved, so grounding cannot be judged. "
                        "Add documents to the knowledge base or paste reference context.",
                details={"claims": [{"claim_id": c.id, "claim": c.text, "verdict": "no_evidence",
                                     "evidence_ids": [], "explanation": "Nothing retrieved."} for c in ctx.claims]})

        chain = self.hub.structured(GroundingJudgement, GROUNDING_SYSTEM, GROUNDING_HUMAN)
        j: GroundingJudgement = await self.hub.ainvoke(chain, {
            "claims": fmt_claims_with_candidates(ctx.claims, ctx.claim_evidence),
            "evidence": fmt_evidence(pick_evidence(ctx.pool, used_ids)),
        })

        by_id = {v.claim_id: v for v in j.verdicts}
        points, evaluable, issues, details = 0.0, 0, [], []
        for claim in ctx.claims:
            v = by_id.get(claim.id)
            verdict = v.verdict if v else "no_evidence"
            cited = [e for e in (v.evidence_ids if v else []) if e in ctx.pool]
            details.append({"claim_id": claim.id, "claim": claim.text, "verdict": verdict, "evidence_ids": cited,
                            "explanation": v.explanation if v else "Not judged."})
            if verdict == "no_evidence":
                continue
            evaluable += 1
            points += VERDICT_CREDIT[verdict]
            if verdict in VERDICT_SEVERITY:
                text = {"contradicted": "The reference documents contradict this claim.",
                        "unsupported": "No retrieved reference document supports this claim.",
                        "partially_supported": "The reference documents support only part of this claim."}[verdict]
                fix = {"contradicted": "Rewrite the claim to match the cited source.",
                       "unsupported": "Remove the claim or back it with a citation.",
                       "partially_supported": "Narrow the claim to what the source states."}[verdict]
                issues.append(make_issue(self.name, verdict, VERDICT_SEVERITY[verdict],
                                         f"{text} {v.explanation if v else ''}", claim, cited, fix))

        total = len(ctx.claims)
        if evaluable == 0:
            return CriticReport(critic=self.name, evaluable=False, confidence=0.0,
                                summary="Retrieved documents were not relevant to any claim.",
                                details={"claims": details})
        coverage = evaluable / total
        if coverage < 0.5:
            issues.append(make_issue(self.name, "low_evidence_coverage", "low",
                                     f"Only {evaluable} of {total} claims had relevant reference evidence, "
                                     "so the grounding score is based on a small part of the answer.",
                                     suggestion="Add more reference documents that cover this topic."))
        score = round(100 * points / evaluable, 1)
        confidence = round(j.confidence * (0.4 + 0.6 * coverage), 2)
        counts = {k: sum(1 for d in details if d["verdict"] == k) for k in
                  ("supported", "partially_supported", "unsupported", "contradicted", "no_evidence")}
        return CriticReport(critic=self.name, score=score, confidence=confidence,
                            summary=f"{counts['supported']} of {total} claims fully supported by retrieved sources; "
                                    f"{counts['contradicted']} contradicted, {counts['unsupported']} unsupported.",
                            issues=issues, details={"claims": details, "counts": counts, "coverage": round(coverage, 2)})
