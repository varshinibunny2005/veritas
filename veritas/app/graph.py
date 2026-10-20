"""LangGraph pipeline:

  START -> extract_claims -> retrieve --+--> accuracy     --+
                                        +--> consistency  --+--> arbitrate -> END
                                        +--> completeness --+
                                        +--> grounding    --+

The four critics run in PARALLEL and never see each other's output (independence).
"""
import asyncio
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.arbiter import Arbiter, ArbitrationError
from app.config import Settings
from app.critics import ALL_CRITICS
from app.critics.base import CriticContext
from app.llm import LLMHub
from app.prompts import CLAIM_HUMAN, CLAIM_SYSTEM
from app.rag.knowledge_base import KnowledgeBase
from app.rag.retriever import EvidenceBuilder
from app.schemas import (ArbitrationResult, Claim, ClaimExtraction, CriticReport, EvaluationRequest, Evidence,
                         ResultMeta)


class PipelineState(TypedDict, total=False):
    request: EvaluationRequest
    claims: list
    pool: list
    question_ids: list
    claim_ids: dict
    warnings: list
    reports: Annotated[list, operator.add]   # every critic appends its report here
    result: ArbitrationResult


def build_graph(settings: Settings, hub: LLMHub, kb: KnowledgeBase):
    builder = EvidenceBuilder(settings, kb)
    arbiter = Arbiter(hub, settings)
    critics = [cls(hub, settings) for cls in ALL_CRITICS]

    async def extract_claims(state: PipelineState) -> dict:
        req = state["request"]
        chain = hub.structured(ClaimExtraction, CLAIM_SYSTEM, CLAIM_HUMAN)
        try:
            out: ClaimExtraction = await hub.ainvoke(chain, {
                "question": req.question, "answer": req.answer, "max_claims": settings.max_claims})
        except Exception as exc:  # noqa: BLE001 - LLM unreachable / wrong model / bad key: say so clearly
            raise ArbitrationError(f"The LLM call failed while extracting claims: {type(exc).__name__}: {exc}") from exc
        texts = [c.strip() for c in out.claims if c and c.strip()][: settings.max_claims]
        return {"claims": [Claim(id=i, text=t) for i, t in enumerate(texts, 1)], "warnings": []}

    async def retrieve(state: PipelineState) -> dict:
        req = state["request"]
        top_k = req.options.top_k or settings.retrieval_top_k
        bundle = await asyncio.to_thread(
            builder.build, req.question, state.get("claims", []), req.extra_context, top_k, settings.claim_top_k)
        return {"pool": bundle.pool, "question_ids": bundle.question_ids, "claim_ids": bundle.claim_ids}

    def make_critic_node(critic):
        async def node(state: PipelineState) -> dict:
            req = state["request"]
            ctx = CriticContext(
                question=req.question, answer=req.answer, claims=state.get("claims", []),
                pool={e.id: e for e in state.get("pool", [])},
                question_evidence_ids=state.get("question_ids", []),
                claim_evidence=state.get("claim_ids", {}), options=req.options)
            report: CriticReport = await critic.run(ctx)
            return {"reports": [report]}
        return node

    async def arbitrate(state: PipelineState) -> dict:
        req = state["request"]
        order = [c.name for c in critics]
        reports = sorted(state["reports"], key=lambda r: order.index(r.critic))
        meta = ResultMeta(
            llm_mode=hub.mode, model=hub.model, embeddings=f"{kb.bundle.provider}:{kb.bundle.model}",
            kb_chunks=kb.count(), warnings=list(state.get("warnings", [])))
        result = await arbiter.rule(req.question, reports, state.get("claims", []), state.get("pool", []), meta)
        return {"result": result}

    g = StateGraph(PipelineState)
    g.add_node("extract_claims", extract_claims)
    g.add_node("retrieve", retrieve)
    for c in critics:
        g.add_node(c.name, make_critic_node(c))
    g.add_node("arbitrate", arbitrate)

    g.add_edge(START, "extract_claims")
    g.add_edge("extract_claims", "retrieve")
    for c in critics:
        g.add_edge("retrieve", c.name)
    g.add_edge([c.name for c in critics], "arbitrate")   # join: wait for all four critics
    g.add_edge("arbitrate", END)
    return g.compile()
