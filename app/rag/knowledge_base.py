"""Persistent knowledge base = LangChain + Chroma vector database."""
import hashlib
import io
import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.rag.embeddings import EmbeddingBundle

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def read_file_text(name: str, data: bytes) -> str:
    """Extract text from an uploaded/disk file (.md, .txt, .pdf)."""
    if Path(name).suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def make_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )


class KnowledgeBase:
    def __init__(self, settings: Settings, bundle: EmbeddingBundle):
        self.settings = settings
        self.bundle = bundle
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=f"kb_{bundle.key}",
            embedding_function=bundle.embeddings,
            persist_directory=str(settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
        )
        self.splitter = make_splitter(settings)

    # ---------------------------------------------------------------- write
    def add_document(self, source: str, text: str) -> int:
        """(Re)index one document. Idempotent: old chunks of the same source are replaced."""
        text = (text or "").strip()
        if not text:
            return 0
        self.delete_source(source)
        chunks = self.splitter.split_text(text)
        docs, ids = [], []
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": source, "chunk_index": i}))
            ids.append(hashlib.sha1(f"{source}::{i}".encode()).hexdigest())
        self.store.add_documents(docs, ids=ids)
        logger.info("Indexed %s -> %d chunks", source, len(chunks))
        return len(chunks)

    def ingest_directory(self, directory: Optional[Path] = None) -> dict:
        directory = Path(directory or self.settings.kb_dir)
        summary: dict = {}
        if not directory.exists():
            return summary
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                summary[path.name] = self.add_document(path.name, read_file_text(path.name, path.read_bytes()))
        return summary

    def delete_source(self, source: str) -> int:
        found = self.store.get(where={"source": source})
        ids = found.get("ids", [])
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    # ---------------------------------------------------------------- read
    def count(self) -> int:
        try:
            return self.store._collection.count()  # noqa: SLF001
        except Exception:
            return len(self.store.get().get("ids", []))

    def list_sources(self) -> list[dict]:
        data = self.store.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            src = (meta or {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Top-k chunks as (document, cosine similarity)."""
        n = self.count()
        if n == 0:
            return []
        hits = self.store.similarity_search_with_score(query, k=min(k, n))
        return [(doc, 1.0 - float(dist)) for doc, dist in hits]
