"""Pydantic models.

Two families:
  * API models        - request / response of the arbitration system
  * LLM output models - what each critic asks the LLM to return (kept simple on
                        purpose so every provider's structured-output mode accepts them)
"""
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Severity = Literal["low", "medium", "high", "critical"]
Verdict = Literal["ACCEPT", "ACCEPT_WITH_WARNINGS", "REVISE", "REJECT"]
CRITIC_NAMES = ["accuracy", "consistency", "completeness", "grounding"]

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RANK_SEVERITY = {v: k for k, v in SEVERITY_RANK.items()}

_SEVERITY_ALIASES = {
    "minor": "low", "info": "low", "trivial": "low",
    "moderate": "medium", "med": "medium",
    "major": "high", "serious": "high", "severe": "critical", "fatal": "critical",
}


def _norm_severity(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        return _SEVERITY_ALIASES.get(s, s)
    return v


def _norm_token(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower().replace(" ", "_").replace("-", "_")
    return v


def _norm_id(v: Any) -> Any:
    """LLMs sometimes answer claim_id as 'C2'; accept that."""
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        return int(m.group()) if m else v
    return v


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #
class EvalOptions(BaseModel):
    selfcheck_samples: Optional[int] = Field(None, ge=0, le=5, description="Override SELFCHECK_SAMPLES")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Override RETRIEVAL_TOP_K")
    critics: list[str] = Field(default_factory=lambda: list(CRITIC_NAMES))

    @field_validator("critics")
    @classmethod
    def _valid_critics(cls, v: list[str]) -> list[str]:
        v = [c.lower() for c in v]
        bad = [c for c in v if c not in CRITIC_NAMES]
        if bad:
            raise ValueError(f"Unknown critics: {bad}. Valid: {CRITIC_NAMES}")
        if not v:
            raise ValueError("Enable at least one critic")
        return v


class EvaluationRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    answer: str = Field(..., min_length=3, max_length=12000)
    extra_context: Optional[str] = Field(
        None, max_length=60000,
        description="Optional reference text used as additional RAG evidence for this request only",
    )
    options: EvalOptions = Field(default_factory=EvalOptions)


class Claim(BaseModel):
    id: int
    text: str


class Evidence(BaseModel):
    id: str
    source: str
    chunk_index: int = 0
    text: str
    score: float = 0.0
    origin: Literal["knowledge_base", "user_context"] = "knowledge_base"


class Issue(BaseModel):
    id: str = ""
    critic: str
    category: str
    severity: Severity
    description: str
    claim_id: Optional[int] = None
    claim: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)
    suggestion: str = ""
    corroborated_by: list[str] = Field(default_factory=list)


class CriticReport(BaseModel):
    critic: str
    skipped: bool = False
    evaluable: bool = True
    score: Optional[float] = None  # 0-100
    confidence: float = 0.0        # 0-1
    summary: str = ""
    issues: list[Issue] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class CriticScore(BaseModel):
    score: Optional[float] = None
    confidence: float = 0.0
    weight: float = 0.0
    summary: str = ""
    status: Literal["ok", "not_evaluable", "failed", "skipped"] = "ok"


class Recommendation(BaseModel):
    priority: int = 1
    action: str
    rationale: str = ""
    addresses: list[str] = Field(default_factory=list)

    @field_validator("priority", mode="before")
    @classmethod
    def _prio(cls, v: Any) -> int:
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            return 1


class ResultMeta(BaseModel):
    llm_mode: str = ""
    model: str = ""
    embeddings: str = ""
    kb_chunks: int = 0
    elapsed_ms: int = 0
    disagreement: float = 0.0
    rules_triggered: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)


class ArbitrationResult(BaseModel):
    verdict: Verdict
    confidence: float
    overall_score: float
    verdict_explanation: str = ""
    scores: dict[str, CriticScore]
    issues: list[Issue]
    recommendations: list[Recommendation]
    claims: list[Claim]
    evidence: list[Evidence]
    critic_reports: list[CriticReport]
    meta: ResultMeta


# --------------------------------------------------------------------------- #
# LLM structured-output models (one per critic step)
# --------------------------------------------------------------------------- #
class ClaimExtraction(BaseModel):
    claims: list[str] = Field(
        default_factory=list,
        description="Atomic, self-contained, checkable factual claims made by the answer",
    )


# --- grounding ---
class GroundingVerdict(BaseModel):
    claim_id: int

    @field_validator("claim_id", mode="before")
    @classmethod
    def _cid(cls, v: Any) -> Any:
        return _norm_id(v)

    verdict: Literal["supported", "partially_supported", "unsupported", "contradicted", "no_evidence"]
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _v(cls, v: Any) -> Any:
        v = _norm_token(v)
        return {"partial": "partially_supported", "partially": "partially_supported",
                "not_supported": "unsupported", "contradiction": "contradicted",
                "no_support": "unsupported", "none": "no_evidence"}.get(v, v)


class GroundingJudgement(BaseModel):
    verdicts: list[GroundingVerdict] = Field(default_factory=list)
    confidence: float = 0.7

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        return _clamp(v, 0.0, 1.0, 0.7)


# --- accuracy ---
class AccuracyAssessment(BaseModel):
    claim_id: int

    @field_validator("claim_id", mode="before")
    @classmethod
    def _cid(cls, v: Any) -> Any:
        return _norm_id(v)

    assessment: Literal["correct", "incorrect", "uncertain"]
    severity: Severity = "medium"
    correction: str = ""
    explanation: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _s(cls, v: Any) -> Any:
        return _norm_severity(v) or "medium"

    @field_validator("assessment", mode="before")
    @classmethod
    def _a(cls, v: Any) -> Any:
        v = _norm_token(v)
        return {"true": "correct", "accurate": "correct", "false": "incorrect",
                "wrong": "incorrect", "inaccurate": "incorrect", "unverifiable": "uncertain",
                "unknown": "uncertain"}.get(v, v)


class AccuracyJudgement(BaseModel):
    assessments: list[AccuracyAssessment] = Field(default_factory=list)
    overall_score: float = 50.0
    confidence: float = 0.7
    summary: str = ""

    @field_validator("overall_score", mode="before")
    @classmethod
    def _o(cls, v: Any) -> float:
        return _clamp(v, 0.0, 100.0, 50.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        return _clamp(v, 0.0, 1.0, 0.7)


# --- consistency ---
class Contradiction(BaseModel):
    statement_a: str
    statement_b: str
    explanation: str = ""
    severity: Severity = "medium"

    @field_validator("severity", mode="before")
    @classmethod
    def _s(cls, v: Any) -> Any:
        return _norm_severity(v) or "medium"


class SampleAgreement(BaseModel):
    claim_id: int

    @field_validator("claim_id", mode="before")
    @classmethod
    def _cid(cls, v: Any) -> Any:
        return _norm_id(v)

    agreement: Literal["consistent", "inconsistent", "unaddressed"]
    note: str = ""

    @field_validator("agreement", mode="before")
    @classmethod
    def _a(cls, v: Any) -> Any:
        v = _norm_token(v)
        return {"agree": "consistent", "agrees": "consistent", "contradicts": "inconsistent",
                "contradicted": "inconsistent", "not_addressed": "unaddressed",
                "neutral": "unaddressed", "unknown": "unaddressed"}.get(v, v)


class ConsistencyJudgement(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)
    sample_agreement: list[SampleAgreement] = Field(default_factory=list)
    confidence: float = 0.7
    summary: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        return _clamp(v, 0.0, 1.0, 0.7)


# --- completeness ---
class AspectCoverage(BaseModel):
    aspect: str
    importance: Literal["high", "medium", "low"] = "medium"
    status: Literal["covered", "partial", "missing"]
    explanation: str = ""

    @field_validator("importance", mode="before")
    @classmethod
    def _i(cls, v: Any) -> Any:
        v = _norm_token(v)
        return {"critical": "high", "essential": "high", "moderate": "medium", "minor": "low"}.get(v, v)

    @field_validator("status", mode="before")
    @classmethod
    def _s(cls, v: Any) -> Any:
        v = _norm_token(v)
        return {"partially_covered": "partial", "partially": "partial", "complete": "covered",
                "addressed": "covered", "absent": "missing", "not_covered": "missing",
                "omitted": "missing"}.get(v, v)


class CompletenessJudgement(BaseModel):
    aspects: list[AspectCoverage] = Field(default_factory=list)
    confidence: float = 0.7
    summary: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        return _clamp(v, 0.0, 1.0, 0.7)


# --- arbiter narrative ---
class ArbiterNarrative(BaseModel):
    verdict_explanation: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list)
