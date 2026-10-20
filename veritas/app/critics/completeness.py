"""Completeness critic: does the answer cover every aspect the question requires?"""
from app.critics.base import BaseCritic, CriticContext, make_issue
from app.prompts import COMPLETENESS_HUMAN, COMPLETENESS_SYSTEM, fmt_evidence, pick_evidence
from app.schemas import CompletenessJudgement, CriticReport

IMPORTANCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}
STATUS_CREDIT = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
MISSING_SEVERITY = {"high": "high", "medium": "medium", "low": "low"}
PARTIAL_SEVERITY = {"high": "medium", "medium": "low"}


class CompletenessCritic(BaseCritic):
    name = "completeness"

    async def evaluate(self, ctx: CriticContext) -> CriticReport:
        evidence = pick_evidence(ctx.pool, ctx.question_evidence_ids)
        chain = self.hub.structured(CompletenessJudgement, COMPLETENESS_SYSTEM, COMPLETENESS_HUMAN)
        j: CompletenessJudgement = await self.hub.ainvoke(chain, {
            "question": ctx.question, "answer": ctx.answer, "evidence": fmt_evidence(evidence),
        })
        if not j.aspects:
            return CriticReport(critic=self.name, evaluable=False, summary="Could not decompose the question.")

        weight_sum = sum(IMPORTANCE_WEIGHT[a.importance] for a in j.aspects)
        earned = sum(IMPORTANCE_WEIGHT[a.importance] * STATUS_CREDIT[a.status] for a in j.aspects)
        issues = []
        for a in j.aspects:
            if a.status == "missing":
                sev, verb = MISSING_SEVERITY[a.importance], "does not address"
            elif a.status == "partial" and a.importance in PARTIAL_SEVERITY:
                sev, verb = PARTIAL_SEVERITY[a.importance], "only partly addresses"
            else:
                continue
            issues.append(make_issue(self.name, "missing_information", sev,
                                     f'The answer {verb}: "{a.aspect}". {a.explanation}',
                                     suggestion=f'Add a clear statement covering "{a.aspect}".'))
        covered = sum(1 for a in j.aspects if a.status == "covered")
        confidence = j.confidence * (0.7 if len(j.aspects) < 2 else 1.0)
        return CriticReport(
            critic=self.name, score=round(100 * earned / weight_sum, 1), confidence=round(confidence, 2),
            summary=j.summary or f"{covered} of {len(j.aspects)} required aspects fully covered.",
            issues=issues, details={"aspects": [a.model_dump() for a in j.aspects]})
