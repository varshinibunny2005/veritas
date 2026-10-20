import os
import sys
import tempfile
from pathlib import Path

# Run every test fully offline: heuristic critics + hashing embeddings + throw-away vector DB.
os.environ["LLM_PROVIDER"] = "offline"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="arbiter_chroma_")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
