"""Embedding back-ends exposed as LangChain `Embeddings` objects.

  fastembed : local ONNX model (default, free, no API key)
  openai    : OpenAI embeddings (needs `pip install langchain-openai`)
  hash      : dependency-free feature-hashing embeddings (offline fallback / tests)
"""
import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.textutil import tokens

logger = logging.getLogger(__name__)


class HashingEmbeddings(Embeddings):
    """Signed feature hashing over unigrams + bigrams. Lexical, but deterministic and offline."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = tokens(text)
        feats = Counter(toks)
        feats.update(Counter(f"{a}_{b}" for a, b in zip(toks, toks[1:])))
        v = np.zeros(self.dim, dtype=np.float64)
        for feat, count in feats.items():
            h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
            sign = 1.0 if (h >> 40) & 1 else -1.0
            weight = (1.0 + math.log(count)) * (0.6 if "_" in feat else 1.0)
            v[h % self.dim] += sign * weight
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([text]))).tolist()


@dataclass
class EmbeddingBundle:
    embeddings: Embeddings
    provider: str          # provider actually in use
    model: str
    key: str               # used in the Chroma collection name
    min_relevance: float   # default cosine-similarity floor for retrieval
    note: str = ""


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-_")[:60]


def build_embeddings(settings: Settings) -> EmbeddingBundle:
    provider = settings.embedding_provider.lower()
    note = ""

    if provider == "fastembed":
        try:
            settings.fastembed_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Loading embedding model %s (first run downloads ~130MB)...", settings.embedding_model)
            emb = FastEmbedEmbeddings(settings.embedding_model, str(settings.fastembed_cache_dir))
            emb.embed_query("warm-up")
            return EmbeddingBundle(emb, "fastembed", settings.embedding_model,
                                   _safe("fastembed-" + settings.embedding_model),
                                   settings.min_relevance if settings.min_relevance is not None else 0.35)
        except Exception as exc:  # no internet / blocked download / unsupported CPU
            note = f"fastembed unavailable ({type(exc).__name__}: {exc}); using hash embeddings instead"
            logger.warning(note)
            provider = "hash"

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
            return EmbeddingBundle(emb, "openai", settings.openai_embedding_model,
                                   _safe("openai-" + settings.openai_embedding_model),
                                   settings.min_relevance if settings.min_relevance is not None else 0.25)
        except Exception as exc:
            note = f"openai embeddings unavailable ({type(exc).__name__}: {exc}); using hash embeddings instead"
            logger.warning(note)

    return EmbeddingBundle(HashingEmbeddings(), "hash", "feature-hashing-1024", "hash-1024",
                           settings.min_relevance if settings.min_relevance is not None else 0.12, note)
