"""The arbiter combines independent critic reports into one ruling.

Deterministic part (transparent + testable):
  * weighted overall score over the critics that could actually evaluate
  * cross-critic corroboration: the same claim flagged by 2+ critics -> severity +1
  * verdict = score band, then capped by the worst issue severities
  * confidence = agreement between critics + critic self-confidence + panel coverage
LLM part (LangChain chain): plain-language explanation + prioritised recommendations, with a
deterministic fallback if the LLM call fails.
"""
import logging
import math
from typing import Optional

from app.config import Settings
from app.llm import LLMHub
from app.prompts import ARBITER_HUMAN, ARBITER_SYSTEM
from app.schemas import (
    CRITIC_NAMES, RANK_SEVERITY, SEVERITY_RANK, ArbiterNarrative, ArbitrationResult, Claim, CriticReport,
    CriticScore, Evidence, Issue, Recommendation, ResultMeta,
)

logger = logging.getLogger(__name__)

# verdict bands on the weighted overall score
ACCEPT_MIN, WARN_MIN, REVISE_MIN = 85.0, 70.0, 50.0
VERDICT_RANK = {"REJECT": 0, "REVISE": 1, "ACCEPT_WITH_WARNINGS": 2, "ACCEPT": 3}
RANK_VERDICT = {v: k for k, v in VERDICT_RANK.items()}
CLAIM_LEVEL_CRITICS = {"accuracy", "grounding", "consistency"}


class ArbitrationError(RuntimeError):
    pass


def _cap(verdict: str, cap: str) -> str:
    return RANK_VERDICT[min(VERDICT_RANK[verdict], VERDICT_RANK[cap])]


def corroborate(issues: list[Issue]) -> None:
    """If independent critics flag the same claim, the problem is more credible: bump its severity."""
    by_claim: dict[int, list[Issue]] = {}
    for it in issues:
        if it.claim_id is not None and it.critic in CLAIM_LEVEL_CRITICS and SEVERITY_RANK[it.severity] >= 2:
            by_claim.setdefault(it.claim_id, []).append(it)
    for group in by_claim.values():
        critics = {i.critic for i in group}
        if len(critics) < 2:
            continue
        top = max(SEVERITY_RANK[i.severity] for i in group)
        for it in group:
            it.corroborated_by = sorted(critics - {it.critic})
            if SEVERITY_RANK[it.severity] == top and top < 4:
                it.severity = RANK_SEVERITY[top + 1]  # type: ignore[assignment]


def decide_verdict(overall: float, issues: list[Issue]) -> tuple[str, list[str]]:
    rules: list[str] = []
    verdict = ("ACCEPT" if overall >= ACCEPT_MIN else "ACCEPT_WITH_WARNINGS" if overall >= WARN_MIN
               else "REVISE" if overall >= REVISE_MIN else "REJECT")
    rules.append(f"Score band: overall {overall:.1f} -> {verdict}")

    crit = [i for i in issues if i.severity == "critical"]
    crit_units = {i.claim_id if i.claim_id is not None else i.id for i in crit}
    has_high = any(i.severity == "high" for i in issues)
    has_medium = any(i.severity == "medium" for i in issues)

    if len(crit_units) >= 2 or (len(crit_units) == 1 and overall < 60):
        if verdict != "REJECT":
            rules.append("Rule: 2+ critical problems (or 1 critical with a weak score) -> REJECT")
        verdict = "REJECT"
    elif crit:
        capped = _cap(verdict, "REVISE")
        if capped != verdict:
            rules.append("Rule: a critical issue caps the verdict at REVISE")
        verdict = capped
    elif has_high:
        capped = _cap(verdict, "REVISE")
        if capped != verdict:
            rules.append("Rule: a high-severity issue caps the verdict at REVISE")
        verdict = capped
    elif has_medium:
        capped = _cap(verdict, "ACCEPT_WITH_WARNINGS")
        if capped != verdict:
            rules.append("Rule: a medium-severity issue caps the verdict at ACCEPT_WITH_WARNINGS")
        verdict = capped
    return verdict, rules


def aggregate(reports: list[CriticReport], weights: dict[str, float]) -> dict:
    active = [r for r in reports if not r.skipped]
    usable = [r for r in active if r.evaluable and r.score is not None]
    if not usable:
        errors = "; ".join(f"{r.critic}: {r.error or r.summary}" for r in active) or "no critics enabled"
        raise ArbitrationError(f"No critic could evaluate this answer ({errors}).")

    w = {r.critic: max(0.0, weights.get(r.critic, 0.0)) or 0.01 for r in usable}
    total_w = sum(w.values())
    overall = sum(w[r.critic] * r.score for r in usable) / total_w  # type: ignore[operator]
    disagreement = math.sqrt(sum(w[r.critic] * (r.score - overall) ** 2 for r in usable) / total_w)  # type: ignore[operator]
    critic_conf = sum(w[r.critic] * r.confidence for r in usable) / total_w
    all_w = sum(max(0.0, weights.get(r.critic, 0.0)) or 0.01 for r in active)
    coverage = total_w / all_w
    agreement = max(0.0, 1.0 - disagreement / 35.0)
    confidence = min(0.99, max(0.05, 0.4 * agreement + 0.4 * critic_conf + 0.2 * coverage))
    return {"usable": usable, "active": active, "overall": overall, "disagreement": disagreement,
            "confidence": confidence, "weights": {k: round(v / total_w, 3) for k, v in w.items()}}


def collect_issues(reports: list[CriticReport]) -> list[Issue]:
    issues = [i.model_copy(deep=True) for r in reports for i in r.issues]
    corroborate(issues)
    order = {n: k for k, n in enumerate(CRITIC_NAMES)}
    issues.sort(key=lambda i: (-SEVERITY_RANK[i.severity], order.get(i.critic, 9),
                               i.claim_id if i.claim_id is not None else 999))
    for n, it in enumerate(issues, 1):
        it.id = f"I{n}"
    return issues


def fallback_recommendations(issues: list[Issue]) -> list[Recommendation]:
    recs, seen = [], set()
    for it in issues:
        action = it.suggestion or f"Resolve: {it.description}"
        if action in seen:
            continue
        seen.add(action)
        recs.append(Recommendation(priority=len(recs) + 1, action=action,
                                   rationale=f"{it.severity.capitalize()} {it.critic} issue: {it.category.replace('_', ' ')}.",
                                   addresses=[it.id]))
        if len(recs) >= 6:
            break
    return recs


def _fmt_issues(issues: list[Issue]) -> str:
    lines = []
    for it in issues[:15]:
        claim = f' [claim: "{it.claim}"]' if it.claim else ""
        lines.append(f"[{it.id}] ({it.severity}) {it.critic}: {it.description}{claim} || fix: {it.suggestion}")
    return "\n".join(lines) or "(no issues found)"


class Arbiter:
    def __init__(self, hub: LLMHub, settings: Settings):
        self.hub = hub
        self.settings = settings

    async def narrate(self, question: str, verdict: str, overall: float, confidence: float,
                      scores: dict[str, CriticScore], issues: list[Issue]) -> ArbiterNarrative:
        score_lines = "\n".join(f"{k}: {v.score if v.score is not None else 'n/a'} - {v.summary}"
                                for k, v in scores.items())
        chain = self.hub.structured(ArbiterNarrative, ARBITER_SYSTEM, ARBITER_HUMAN)
        return await self.hub.ainvoke(chain, {
            "verdict": verdict, "overall": round(overall, 1), "confidence": round(confidence, 2),
            "scores": score_lines, "issues": _fmt_issues(issues), "question": question})

    async def rule(self, question: str, reports: list[CriticReport], claims: list[Claim],
                   evidence: list[Evidence], meta: ResultMeta) -> ArbitrationResult:
        agg = aggregate(reports, self.settings.weights)
        issues = collect_issues(reports)
        verdict, rules = decide_verdict(agg["overall"], issues)

        scores: dict[str, CriticScore] = {}
        for r in reports:
            status = ("skipped" if r.skipped else "failed" if r.error else "ok" if r.evaluable and r.score is not None
                      else "not_evaluable")
            scores[r.critic] = CriticScore(score=r.score, confidence=r.confidence,
                                           weight=agg["weights"].get(r.critic, 0.0), summary=r.summary, status=status)
            if r.error:
                meta.warnings.append(f"{r.critic} critic failed: {r.error}")
            elif status == "not_evaluable":
                meta.warnings.append(f"{r.critic} critic could not evaluate: {r.summary}")
        if agg["disagreement"] > 25:
            meta.warnings.append("Critics disagree strongly; treat the verdict with extra care.")

        narrative: Optional[ArbiterNarrative] = None
        try:
            narrative = await self.narrate(question, verdict, agg["overall"], agg["confidence"], scores, issues)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Arbiter narrative failed, using deterministic fallback: %s", exc)
            meta.warnings.append("Recommendation writer (LLM) failed; showing rule-based recommendations.")

        valid_ids = {i.id for i in issues}
        recs = []
        if narrative:
            for n, rec in enumerate(sorted(narrative.recommendations, key=lambda r: r.priority)[:6], 1):
                rec.priority = n
                rec.addresses = [a for a in rec.addresses if a in valid_ids]
                recs.append(rec)
        if not recs:
            recs = fallback_recommendations(issues)

        explanation = (narrative.verdict_explanation if narrative and narrative.verdict_explanation else
                       f"Overall score {agg['overall']:.0f}/100 with {len(issues)} issue(s) found.")
        meta.disagreement = round(agg["disagreement"], 1)
        meta.rules_triggered = rules
        meta.weights = agg["weights"]
        return ArbitrationResult(
            verdict=verdict, confidence=round(agg["confidence"], 2), overall_score=round(agg["overall"], 1),  # type: ignore[arg-type]
            verdict_explanation=explanation, scores=scores, issues=issues, recommendations=recs,
            claims=claims, evidence=evidence, critic_reports=reports, meta=meta)
