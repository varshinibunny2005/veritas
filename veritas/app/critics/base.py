"""Shared plumbing for the four critics."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings
from app.llm import LLMHub
from app.schemas import Claim, CriticReport, Evidence, EvalOptions, Issue

logger = logging.getLogger(__name__)


@dataclass
class CriticContext:
    question: str
    answer: str
    claims: list[Claim]
    pool: dict[str, Evidence]                 # evidence id -> chunk
    question_evidence_ids: list[str]          # retrieved for the question
    claim_evidence: dict[int, list[str]]      # retrieved per claim
    options: EvalOptions = field(default_factory=EvalOptions)

    def claim(self, claim_id: int) -> Optional[Claim]:
        return next((c for c in self.claims if c.id == claim_id), None)


def make_issue(critic: str, category: str, severity: str, description: str, claim: Optional[Claim] = None,
               evidence_ids: Optional[list[str]] = None, suggestion: str = "") -> Issue:
    return Issue(critic=critic, category=category, severity=severity, description=description.strip(),
                 claim_id=claim.id if claim else None, claim=claim.text if claim else None,
                 evidence_ids=evidence_ids or [], suggestion=suggestion.strip())


class BaseCritic(ABC):
    name: str = ""

    def __init__(self, hub: LLMHub, settings: Settings):
        self.hub = hub
        self.settings = settings

    @abstractmethod
    async def evaluate(self, ctx: CriticContext) -> CriticReport: ...

    async def run(self, ctx: CriticContext) -> CriticReport:
        """Never raises: a failing critic becomes a 'failed' report so the panel can still rule."""
        if self.name not in ctx.options.critics:
            return CriticReport(critic=self.name, skipped=True, evaluable=False, summary="Critic disabled for this run.")
        try:
            return await self.evaluate(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Critic %s failed", self.name)
            return CriticReport(critic=self.name, evaluable=False, confidence=0.0,
                                summary="This critic failed to run.", error=f"{type(exc).__name__}: {exc}")
