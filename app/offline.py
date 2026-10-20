"""Offline heuristic "LLM" so the whole pipeline runs with no API key.

It is a LangChain `BaseChatModel` that implements `with_structured_output` by reading the tagged
prompt sections and applying simple lexical rules (token overlap, number mismatch, negation).
It is meant for demos and automated tests - NOT a substitute for a real LLM judge.
"""
import re
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.schemas import (
    AccuracyAssessment, AccuracyJudgement, ArbiterNarrative, AspectCoverage, ClaimExtraction,
    CompletenessJudgement, ConsistencyJudgement, Contradiction, GroundingJudgement, GroundingVerdict,
    Recommendation, SampleAgreement,
)
from app.textutil import numbers, sentences, token_set


# ------------------------------------------------------------------ prompt parsing
def _section(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1) if m else ""


def _parse_claims(block: str) -> list[tuple[int, str, list[str]]]:
    out: list[tuple[int, str, list[str]]] = []
    for line in block.splitlines():
        m = re.match(r"^\[C(\d+)\]\s*(.*)$", line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip(), []))
            continue
        m = re.match(r"^\s+candidates:\s*(.*)$", line)
        if m and out and m.group(1).strip().lower() != "none":
            out[-1][2].extend(x.strip() for x in m.group(1).split(",") if x.strip())
    return out


def _parse_evidence(block: str) -> dict[str, str]:
    parts = re.split(r"^\[(E\d+)\][^\n]*\n", block, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _parse_samples(block: str) -> list[str]:
    parts = re.split(r"^\[S\d+\]\s*", block, flags=re.M)
    return [p.strip() for p in parts if p.strip() and p.strip() != "(none)"]


# ------------------------------------------------------------------ heuristics
def _grade(claim: str, evidence: list[tuple[str, str]]) -> dict:
    """Compare one claim to evidence sentences -> verdict + best evidence."""
    ct, cn = token_set(claim), numbers(claim)
    claim_negated = bool(re.search(r"\b(not|never|no)\b", claim.lower()))
    claim_years = {n for n in cn if re.fullmatch(r"(1[5-9]|20)\d\d", n)}
    scored = []  # (overlap, evidence id, sentence)
    for eid, text in evidence:
        for sent in sentences(text):
            scored.append((len(ct & token_set(sent)) / max(1, len(ct)), eid, sent))
    if not scored:
        return {"verdict": "no_evidence", "id": None, "ov": 0.0, "why": "No relevant evidence found.", "text": ""}
    scored.sort(key=lambda x: -x[0])
    ov, eid, sent = scored[0]
    verdict, why, cite, cite_text = "no_evidence", "No relevant evidence found.", eid, sent

    # contradiction 1: a related sentence negates a word used by the claim ("made of iron, not steel")
    negated = None
    if not claim_negated:
        for o, e, s_ in scored:
            if o < 0.25:
                break
            for m in re.finditer(r"\b(?:not|never|no)\s+(?:a\s+|an\s+|the\s+|to\s+)?([a-z0-9]+)", s_.lower()):
                if token_set(m.group(1)) & ct:
                    negated = (e, s_)
    # contradiction 2: same topic, but a different year / number
    numeric = None
    if claim_years and not any(y in numbers(s_) for _, _, s_ in scored for y in claim_years):
        cy = min(int(y) for y in claim_years)
        best_gap = None
        for o, e, s_ in scored:
            years = [int(n) for n in numbers(s_) if re.fullmatch(r"(1[5-9]|20)\d\d", n)]
            if o >= 0.25 and years:
                gap = min(abs(y - cy) for y in years)   # prefer the sentence with the closest year
                if best_gap is None or gap < best_gap:
                    best_gap, numeric = gap, (e, s_)
    elif cn and not cn <= numbers(sent) and numbers(sent) and ov >= 0.35:
        numeric = (eid, sent)

    hit = negated or numeric
    if hit:
        verdict, cite, cite_text = "contradicted", hit[0], hit[1]
        why = f"Evidence says: {hit[1]}"
    elif ov >= 0.65:
        verdict, why = "supported", "Evidence sentence closely matches the claim."
    elif ov >= 0.4:
        verdict, why = "partially_supported", "Evidence overlaps only partly with the claim."
    elif ov >= 0.18:
        verdict, why = "unsupported", "Evidence is on-topic but does not back this claim."
    return {"verdict": verdict, "id": cite, "ov": ov, "why": why, "text": cite_text}


def _claims_task(human: str) -> ClaimExtraction:
    answer = _section(human, "answer")
    claims = []
    for s in sentences(answer):
        s = re.sub(r"^[\-\*\d\.\)\s]+", "", s).strip()
        if len(s.split()) >= 4:
            claims.append(s)
    return ClaimExtraction(claims=claims[:8])


def _grounding_task(human: str) -> GroundingJudgement:
    ev = _parse_evidence(_section(human, "evidence"))
    verdicts = []
    for cid, text, cands in _parse_claims(_section(human, "claims")):
        g = _grade(text, [(e, ev[e]) for e in cands if e in ev])
        verdicts.append(GroundingVerdict(claim_id=cid, verdict=g["verdict"],
                                         evidence_ids=[g["id"]] if g["id"] else [], explanation=g["why"]))
    return GroundingJudgement(verdicts=verdicts, confidence=0.6)


def _accuracy_task(human: str) -> AccuracyJudgement:
    ev = list(_parse_evidence(_section(human, "evidence")).items())
    out, score_sum = [], 0.0
    claims = _parse_claims(_section(human, "claims"))
    for cid, text, _ in claims:
        g = _grade(text, ev)
        if g["verdict"] == "contradicted":
            a = AccuracyAssessment(claim_id=cid, assessment="incorrect", severity="high",
                                   correction=g["text"], explanation=g["why"])
            score_sum += 0
        elif g["verdict"] == "supported" or (g["verdict"] == "partially_supported" and g["ov"] >= 0.5):
            a = AccuracyAssessment(claim_id=cid, assessment="correct", explanation=g["why"])
            score_sum += 1
        else:
            a = AccuracyAssessment(claim_id=cid, assessment="uncertain", explanation=g["why"])
            score_sum += 0.5
        out.append(a)
    overall = 100 * score_sum / len(claims) if claims else 50
    return AccuracyJudgement(assessments=out, overall_score=overall, confidence=0.55,
                             summary="Heuristic check against retrieved evidence (offline mode).")


def _completeness_task(human: str) -> CompletenessJudgement:
    question, answer = _section(human, "question"), _section(human, "answer")
    ev = _parse_evidence(_section(human, "evidence"))
    at = token_set(answer)
    generic = {"explain", "describe", "tell", "include", "including", "list", "give", "summarize", "summarise",
               "discuss", "stage", "main", "key", "overview", "happen", "occur", "work", "use", "thing", "detail"}
    parts = [p.strip() for p in re.split(r"\?|,|;|\band\b|\bas well as\b", question) if p.strip()]
    aspects: list[tuple[str, str]] = [(p, "high") for p in parts if token_set(p) - generic][:5]
    qt = token_set(question)
    ranked = []
    for text in list(ev.values())[:2]:
        for s in sentences(text):
            ranked.append((len(qt & token_set(s)), s))
    ranked.sort(key=lambda x: -x[0])
    for ov, s in ranked[:2]:
        if ov > 0:
            aspects.append((("Key fact: " + s)[:110], "medium"))
    covered = []
    for name, imp in aspects:
        toks = token_set(name.replace("Key fact:", "")) - generic
        ov = len(toks & at) / max(1, len(toks))
        status = "covered" if ov >= 0.5 else "partial" if ov >= 0.2 else "missing"
        covered.append(AspectCoverage(aspect=name, importance=imp, status=status,
                                      explanation=f"Keyword coverage {ov:.0%}."))
    return CompletenessJudgement(aspects=covered, confidence=0.55,
                                 summary="Keyword-level coverage check (offline mode).")


def _consistency_task(human: str) -> ConsistencyJudgement:
    answer = _section(human, "answer")
    sents = [s for s in sentences(answer) if len(s.split()) >= 4]
    contradictions = []
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            a, b = token_set(sents[i]), token_set(sents[j])
            na, nb = numbers(sents[i]), numbers(sents[j])
            shared = {t for t in a & b if not t.isdigit()}
            if na and nb and not (na & nb) and len(shared) >= 2 and len(shared) / max(1, min(len(a), len(b))) >= 0.35:
                contradictions.append(Contradiction(
                    statement_a=sents[i], statement_b=sents[j], severity="high",
                    explanation="Same topic stated with different numbers/dates."))
    samples = _parse_samples(_section(human, "samples"))
    agreement = []
    if samples:
        for cid, text, _ in _parse_claims(_section(human, "claims")):
            ct, cn = token_set(text), numbers(text)
            best, kind = 0.0, "unaddressed"
            for smp in samples:
                for s in sentences(smp):
                    ov = len(ct & token_set(s)) / max(1, len(ct))
                    sn = numbers(s)
                    if cn and sn and not cn <= sn and ov >= 0.35:
                        kind, best = "inconsistent", 1.0
                    elif ov >= 0.5 and best < 1.0:
                        kind, best = "consistent", max(best, ov)
            agreement.append(SampleAgreement(claim_id=cid, agreement=kind))
    return ConsistencyJudgement(contradictions=contradictions, sample_agreement=agreement, confidence=0.55,
                                summary="Heuristic contradiction and sample-agreement check (offline mode).")


def _narrative_task(human: str) -> ArbiterNarrative:
    verdict = _section(human, "verdict")
    recs = []
    for line in _section(human, "issues").splitlines():
        m = re.match(r"^\[(I\d+)\]\s*\((\w+)\)\s*(\w+):\s*(.*?)(?:\s*\|\|\s*fix:\s*(.*))?$", line.strip())
        if not m or len(recs) >= 6:
            continue
        iid, sev, critic, desc, fix = m.groups()
        recs.append(Recommendation(priority=len(recs) + 1, addresses=[iid],
                                   action=(fix or f"Resolve: {desc}").strip(),
                                   rationale=f"{sev.capitalize()} {critic} issue."))
    return ArbiterNarrative(verdict_explanation=f"Panel result: {verdict}.", recommendations=recs)


_TASKS = {
    "ClaimExtraction": _claims_task,
    "GroundingJudgement": _grounding_task,
    "AccuracyJudgement": _accuracy_task,
    "CompletenessJudgement": _completeness_task,
    "ConsistencyJudgement": _consistency_task,
    "ArbiterNarrative": _narrative_task,
}


def _last_human(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


class OfflineChatModel(BaseChatModel):
    """Deterministic stand-in used when no LLM API key is configured."""

    @property
    def _llm_type(self) -> str:
        return "offline-heuristic"

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        """Used for sampled answers: return the evidence sentences most related to the question."""
        human = _last_human(messages)
        question = _section(human, "question")
        qt = token_set(question)
        ranked = []
        for text in _parse_evidence(_section(human, "evidence")).values():
            for s in sentences(text):
                ranked.append((len(qt & token_set(s)), s))
        ranked.sort(key=lambda x: -x[0])
        reply = " ".join(s for ov, s in ranked[:2] if ov > 0) or "I am not certain about this."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def with_structured_output(self, schema: Any, **kwargs: Any):  # type: ignore[override]
        task = _TASKS[schema.__name__]

        def run(prompt_value: Any):
            messages = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
            return task(_last_human(messages))

        return RunnableLambda(run)
