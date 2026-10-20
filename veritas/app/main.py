"""FastAPI application: REST API + static web UI."""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.arbiter import ArbitrationError
from app.examples import EXAMPLES
from app.rag.knowledge_base import SUPPORTED_SUFFIXES, read_file_text
from app.schemas import ArbitrationResult, EvaluationRequest
from app.services import Services, build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
services: Optional[Services] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global services
    services = await asyncio.to_thread(build_services)
    await asyncio.to_thread(services.hub.check_model)
    if services.kb.count() == 0:
        logger.info("Knowledge base is empty - indexing sample documents from %s", services.settings.kb_dir)
        summary = await asyncio.to_thread(services.kb.ingest_directory)
        logger.info("Indexed: %s", summary)
    logger.info("Ready -> open http://127.0.0.1:8000")
    yield


app = FastAPI(title="LLM Output Arbitration System", version="1.0.0", lifespan=lifespan,
              description="Independent critics (accuracy, consistency, completeness, RAG grounding) + an arbiter.")


def svc() -> Services:
    if services is None:
        raise HTTPException(503, "Service is still starting")
    return services


# ------------------------------------------------------------------ evaluation
@app.post("/api/evaluate", response_model=ArbitrationResult, tags=["arbitration"])
async def evaluate(req: EvaluationRequest):
    try:
        return await svc().evaluate(req)
    except ArbitrationError as exc:
        raise HTTPException(502, str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(504, "Evaluation timed out. Try fewer self-check samples or a faster model.")


@app.get("/api/health", tags=["system"])
async def health():
    s = svc()
    return {"status": "ok", "llm_mode": s.hub.mode, "model": s.hub.model, "llm_warning": s.hub.warning,
            "embeddings": f"{s.embeddings.provider}:{s.embeddings.model}", "embedding_note": s.embeddings.note,
            "kb_chunks": s.kb.count()}


@app.get("/api/examples", tags=["system"])
async def examples():
    return EXAMPLES


# ------------------------------------------------------------------ knowledge base
class TextDoc(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    text: str = Field(..., min_length=10, max_length=200000)


@app.get("/api/documents", tags=["knowledge-base"])
async def list_documents():
    s = svc()
    return {"chunks": await asyncio.to_thread(s.kb.count), "sources": await asyncio.to_thread(s.kb.list_sources)}


@app.post("/api/documents", tags=["knowledge-base"])
async def upload_documents(files: list[UploadFile] = File(...)):
    s, added = svc(), {}
    for f in files:
        name = Path(f.filename or "upload.txt").name
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"{name}: only {', '.join(sorted(SUPPORTED_SUFFIXES))} files are supported")
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        try:
            text = read_file_text(name, data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not read {name}: {exc}")
        added[name] = await asyncio.to_thread(s.kb.add_document, name, text)
    return {"indexed": added, "chunks": s.kb.count()}


@app.post("/api/documents/text", tags=["knowledge-base"])
async def add_text_document(doc: TextDoc):
    s = svc()
    name = doc.name if Path(doc.name).suffix else doc.name + ".txt"
    n = await asyncio.to_thread(s.kb.add_document, Path(name).name, doc.text)
    return {"indexed": {Path(name).name: n}, "chunks": s.kb.count()}


@app.delete("/api/documents/{source}", tags=["knowledge-base"])
async def delete_document(source: str):
    s = svc()
    removed = await asyncio.to_thread(s.kb.delete_source, source)
    if not removed:
        raise HTTPException(404, f"No document named {source}")
    return {"removed_chunks": removed, "chunks": s.kb.count()}


@app.post("/api/reindex", tags=["knowledge-base"])
async def reindex_samples():
    s = svc()
    summary = await asyncio.to_thread(s.kb.ingest_directory)
    return {"indexed": summary, "chunks": s.kb.count()}


# ------------------------------------------------------------------ frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")
