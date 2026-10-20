"""Builds the evidence pool used by the critics (the "R" in RAG).

  * one retrieval for the QUESTION      -> evidence used by accuracy / completeness / consistency
  * one retrieval per extracted CLAIM   -> evidence used by the grounding critic
  * optional per-request `extra_context` is chunked + embedded into an in-memory
    vector store and searched together with the persistent Chroma knowledge base.
"""
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import Settings
from app.rag.knowledge_base import KnowledgeBase, make_splitter
from app.schemas import Claim, Evidence


@dataclass
class EvidenceBundle:
    pool: list[Evidence] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)
    claim_ids: dict[int, list[str]] = field(default_factory=dict)  # claim id -> evidence ids


class EvidenceBuilder:
    def __init__(self, settings: Settings, kb: KnowledgeBase):
        self.settings = settings
        self.kb = kb
        self.min_relevance = kb.bundle.min_relevance

    def _ephemeral_store(self, text: Optional[str]) -> Optional[InMemoryVectorStore]:
        if not text or not text.strip():
            return None
        chunks = make_splitter(self.settings).split_text(text)
        store = InMemoryVectorStore(self.kb.bundle.embeddings)
        store.add_documents(
            [Document(page_content=c, metadata={"source": "user_provided_context", "chunk_index": i})
             for i, c in enumerate(chunks)]
        )
        return store

    def _search(self, query: str, k: int, eph: Optional[InMemoryVectorStore]) -> list[tuple[Document, float, str]]:
        hits = [(d, s, "knowledge_base") for d, s in self.kb.search(query, k)]
        if eph is not None:
            hits += [(d, float(s), "user_context") for d, s in eph.similarity_search_with_score(query, k=k)]
        hits.sort(key=lambda h: h[1], reverse=True)
        return [h for h in hits[:k] if h[1] >= self.min_relevance]

    def build(self, question: str, claims: list[Claim], extra_context: Optional[str],
              top_k: int, claim_top_k: int) -> EvidenceBundle:
        eph = self._ephemeral_store(extra_context)
        bundle = EvidenceBundle()
        index: dict[tuple, str] = {}  # (origin, source, chunk) -> evidence id

        def register(doc: Document, score: float, origin: str) -> str:
            key = (origin, doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if key in index:
                ev = next(e for e in bundle.pool if e.id == index[key])
                ev.score = max(ev.score, round(score, 3))
                return index[key]
            eid = f"E{len(bundle.pool) + 1}"
            index[key] = eid
            bundle.pool.append(Evidence(
                id=eid, source=str(doc.metadata.get("source", "unknown")),
                chunk_index=int(doc.metadata.get("chunk_index", 0)),
                text=doc.page_content, score=round(score, 3), origin=origin,
            ))
            return eid

        bundle.question_ids = [register(*h) for h in self._search(question, top_k, eph)]
        for claim in claims:
            bundle.claim_ids[claim.id] = [register(*h) for h in self._search(claim.text, claim_top_k, eph)]
        return bundle
