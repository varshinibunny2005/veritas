# LLM Output Arbitration System

Give it a **question** and an **LLM-generated answer**. Four independent critics check the answer, then an
**arbiter** combines their findings into one ruling: verdict, confidence, per-critic scores, issues with severity,
and actionable recommendations.

Main technologies: **RAG** (Chroma vector DB + embeddings) and **LangChain** (LCEL chains, structured output,
text splitters, vector-store integrations). Supporting: LangGraph, FastAPI, Pydantic, vanilla HTML/CSS/JS.

```
                         +-> Accuracy     (is each claim true?)              --+
question + answer        |
      |                  +-> Grounding    (RAG: is each claim backed by      --+
      v                  |                 retrieved documents?)               |
 extract claims -> retrieve evidence                                           +-> Arbiter -> verdict, confidence,
 (LangChain)       (Chroma + embeddings)  +-> Consistency  (self-contradiction --+    scores, issues, recommendations
                                          |                + SelfCheck samples) |
                                          +-> Completeness (covers every part  --+
                                                            of the question?)
        LangGraph runs the four critics in parallel; they never see each other's output.
```

---

## Run it in VS Code (5 minutes)

**Needs:** Python 3.10, 3.11 or 3.12 and VS Code with the *Python* extension. Internet is needed once
(package install + a ~130 MB embedding model download on first start).

1. **Unzip** the project, then in VS Code choose **File > Open Folder** and open the `llm-output-arbitration` folder.
2. Open a terminal: **Terminal > New Terminal**.
3. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process Bypass` once and activate again.

   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   If VS Code asks *"We noticed a new environment"*, click **Yes**. Or press `Ctrl+Shift+P`,
   run **Python: Select Interpreter** and pick the `.venv` one.
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Add your LLM key.** Copy the example file and edit it:
   ```bash
   copy .env.example .env        # Windows
   cp .env.example .env          # macOS / Linux
   ```
   Open `.env` and paste a free Groq key (https://console.groq.com/keys):
   ```
   GROQ_API_KEY=your_key_here
   ```
   No key yet? Skip this step. The app starts in **offline demo mode** (simple heuristic critics) so you can
   see the whole flow, but real LLM judging needs a key.
6. **Start the server**
   ```bash
   python run.py
   ```
   or press **F5** and choose **Run API (FastAPI)** (auto-reload while you edit code).
7. Open **http://127.0.0.1:8000** in your browser. Choose a sample, then **Run arbitration**.
   Interactive API docs are at http://127.0.0.1:8000/docs.

First start: the embedding model downloads, then the 6 sample documents are indexed into the vector DB
(`storage/chroma`). Wait for `Ready -> open http://127.0.0.1:8000` in the terminal.

Run the tests (they run offline, no key needed): `pytest -q`, or the **Testing** icon in the VS Code sidebar.

---

## Using it

* **Samples**: four ready-made cases (wrong facts, incomplete + overclaim, a good answer, a self-contradiction).
* **Your own case**: paste any question and answer. The critics ground the answer in the **knowledge base**.
  Add your own documents (`.md`, `.txt`, `.pdf`) with **Upload files** or **paste text**. For a one-off check use
  *Extra reference text for this run*.
* **Result page**: verdict, confidence, score per critic, the answer with sentences marked by problem severity,
  issues (severity, claim, fix), prioritised recommendations, and "working papers" (claim-by-claim table,
  retrieved passages, raw JSON).

### API example
```bash
curl -X POST http://127.0.0.1:8000/api/evaluate -H "Content-Type: application/json" -d '{
  "question": "How tall is the Eiffel Tower and what is it made of?",
  "answer": "The Eiffel Tower is 330 metres tall and made of steel."
}'
```
Other endpoints: `GET /api/health`, `GET/POST /api/documents`, `POST /api/documents/text`,
`DELETE /api/documents/{name}`, `POST /api/reindex`.

---

## How the ruling is produced

| Step | What happens | LangChain / tech |
|---|---|---|
| 1. Claim extraction | LLM turns the answer into atomic, checkable claims | prompt + `with_structured_output` |
| 2. Retrieval (RAG) | One search for the question and one per claim over Chroma (+ optional pasted context) | embeddings, `Chroma`, `InMemoryVectorStore`, text splitter |
| 3. Critics (parallel) | Each critic gets only the question, answer, claims and evidence | LangGraph fan-out, LCEL chains |
| 4. Arbiter | Weighted score, cross-critic checks, verdict rules, confidence, then an LLM writes explanation and recommendations | deterministic code + LCEL chain |

**Critics**
* **Accuracy**: each claim is `correct`, `incorrect` (with severity and correction) or `uncertain`, using evidence plus the model's own knowledge.
* **Grounding**: each claim is judged *only* against its retrieved passages: `supported`, `partially_supported`, `unsupported`, `contradicted`, or `no_evidence`. If nothing relevant is retrieved the critic reports "not evaluable" instead of guessing.
* **Consistency**: internal contradictions, plus a SelfCheck-style test: extra answers are sampled from the LLM and compared to each claim (`SELFCHECK_SAMPLES`, default 2, set 0 to save calls).
* **Completeness**: the question is decomposed into required aspects (including implicit ones such as "limitations" when benefits are asked) and each is `covered`, `partial` or `missing`.

**Arbiter rules** (all in `app/arbiter.py`, easy to change)
* Overall score = weighted average of the critics that could evaluate (weights in `.env`, renormalised).
* Score bands: 85+ `ACCEPT`, 70+ `ACCEPT_WITH_WARNINGS`, 50+ `REVISE`, otherwise `REJECT`.
* Severity caps: a *medium* issue caps at `ACCEPT_WITH_WARNINGS`, a *high* issue at `REVISE`, a *critical* issue at `REVISE`; two critical problems on different claims (or one critical with a score under 60) give `REJECT`.
* **Corroboration**: if two critics flag the same claim, its severity goes up one level.
* **Confidence** = 0.4 x agreement between critics + 0.4 x critics' own confidence + 0.2 x share of the panel that could evaluate.

---

## Configuration (`.env`)

| Setting | Meaning |
|---|---|
| `LLM_PROVIDER` | `groq` (default), `openai`, `anthropic`, `ollama`, or `offline` |
| `GROQ_MODEL` etc. | model name per provider |
| `EMBEDDING_PROVIDER` | `fastembed` (local, default), `openai`, or `hash` (offline fallback) |
| `SELFCHECK_SAMPLES` | independent answers sampled for the consistency critic (0 to 5) |
| `MAX_CLAIMS`, `RETRIEVAL_TOP_K`, `CLAIM_TOP_K` | pipeline size |
| `MAX_CONCURRENCY` | parallel LLM calls; lower it if you hit rate limits |
| `WEIGHT_*` | critic weights used by the arbiter |

Other providers: `pip install -r requirements-optional.txt` (or only the one you need), set `LLM_PROVIDER` and its key.
If you change `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`, documents are indexed into a fresh collection automatically
(click **Re-index sample documents**, and re-upload your own files).

## Troubleshooting

* **"Offline demo mode" banner**: `.env` is missing, or the key is empty. Add `GROQ_API_KEY` and restart.
* **`model_not_found` (404) from Groq**: the model in `GROQ_MODEL` is retired or not on your account. The banner lists the models your key can use; set one in `.env` (for example `GROQ_MODEL=openai/gpt-oss-120b`) and restart.
* **Rate-limit (429) or timeouts on a free tier**: set `SELFCHECK_SAMPLES=0` and `MAX_CONCURRENCY=2`, or use a smaller model.
* **Embedding model cannot download**: the app falls back to `hash` embeddings (works, but only matches similar words) and says so in the banner. Fix your internet or proxy and restart.
* **`pip install` fails on chromadb (Windows)**: use Python 3.11 or 3.12 and run `python -m pip install --upgrade pip` first.
* **Port 8000 in use**: `python -m uvicorn app.main:app --port 8001`.
* **Reset the vector DB**: stop the server, delete `storage/chroma`, start again.

## Project layout

```
app/
  main.py            FastAPI routes + serves the UI
  graph.py           LangGraph pipeline (claims -> retrieve -> 4 critics in parallel -> arbiter)
  arbiter.py         scoring, corroboration, verdict rules, confidence, recommendations
  critics/           accuracy.py, grounding.py, consistency.py, completeness.py
  rag/               embeddings.py, knowledge_base.py (Chroma), retriever.py (evidence builder)
  llm.py             LangChain provider factory, structured-output chains with JSON fallback
  prompts.py         every prompt in one file
  schemas.py         Pydantic models (API + LLM outputs)
  offline.py         heuristic stand-in model for demo mode and tests
frontend/            index.html, style.css, app.js
data/knowledge_base/ sample documents (add your own here or via the UI)
tests/               24 offline tests (pytest)
```

## Ideas to extend

Hybrid search (BM25 + vectors) and re-ranking in `rag/retriever.py`; a fifth critic (safety, tone, citation format);
run the panel on a batch CSV and export a report; use a different, stronger model for the arbiter than for the critics.
