"""Wires settings + LLM hub + knowledge base + graph together."""
import asyncio
import logging
import time
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.graph import build_graph
from app.llm import LLMHub
from app.rag.embeddings import EmbeddingBundle, build_embeddings
from app.rag.knowledge_base import KnowledgeBase
from app.schemas import ArbitrationResult, EvaluationRequest

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    hub: LLMHub
    embeddings: EmbeddingBundle
    kb: KnowledgeBase
    graph: object

    async def evaluate(self, req: EvaluationRequest) -> ArbitrationResult:
        start = time.perf_counter()
        final = await asyncio.wait_for(self.graph.ainvoke({"request": req, "reports": []}),
                                       timeout=self.settings.request_timeout_s)
        result: ArbitrationResult = final["result"]
        result.meta.elapsed_ms = int((time.perf_counter() - start) * 1000)
        return result


def build_services(settings: Settings = None) -> Services:
    settings = settings or get_settings()
    hub = LLMHub(settings)
    bundle = build_embeddings(settings)
    kb = KnowledgeBase(settings, bundle)
    graph = build_graph(settings, hub, kb)
    logger.info("LLM mode=%s model=%s | embeddings=%s | KB chunks=%d", hub.mode, hub.model, bundle.provider, kb.count())
    return Services(settings, hub, bundle, kb, graph)
