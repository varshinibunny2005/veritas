import pytest
from fastapi.testclient import TestClient

from app.examples import EXAMPLES
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # runs startup: builds services + indexes sample docs
        yield c


def ex(name):
    return next(e for e in EXAMPLES if e["id"] == name)


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["llm_mode"] == "offline" and h["kb_chunks"] > 0


def test_frontend_served(client):
    assert "LLM Output Arbitration" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_bad_request_rejected(client):
    assert client.post("/api/evaluate", json={"question": "x", "answer": "y"}).status_code == 422


def test_bad_answer_is_not_accepted(client):
    e = ex("eiffel")
    r = client.post("/api/evaluate", json={"question": e["question"], "answer": e["answer"]}).json()
    assert r["verdict"] in ("REVISE", "REJECT")
    assert set(r["scores"]) == {"accuracy", "consistency", "completeness", "grounding"}
    assert 0 <= r["confidence"] <= 1 and 0 <= r["overall_score"] <= 100
    assert any("1899" in (i["claim"] or "") and i["severity"] in ("high", "critical") for i in r["issues"])
    assert r["recommendations"] and r["evidence"] and r["claims"]
    assert [i["id"] for i in r["issues"]] == [f"I{n}" for n in range(1, len(r["issues"]) + 1)]


def test_good_answer_is_accepted(client):
    e = ex("photosynthesis")
    r = client.post("/api/evaluate", json={"question": e["question"], "answer": e["answer"]}).json()
    assert r["verdict"] in ("ACCEPT", "ACCEPT_WITH_WARNINGS")


def test_self_contradiction_detected(client):
    e = ex("python")
    r = client.post("/api/evaluate", json={"question": e["question"], "answer": e["answer"]}).json()
    assert any(i["category"] == "internal_contradiction" for i in r["issues"])


def test_disabled_critic_is_skipped(client):
    e = ex("rag")
    body = {"question": e["question"], "answer": e["answer"], "options": {"critics": ["accuracy", "grounding"]}}
    r = client.post("/api/evaluate", json=body).json()
    assert r["scores"]["consistency"]["status"] == "skipped" and r["scores"]["accuracy"]["status"] == "ok"


def test_extra_context_is_used_as_evidence(client):
    body = {"question": "How many moons does Planet Zorb have?",
            "answer": "Planet Zorb has seven moons.",
            "extra_context": "Planet Zorb is a fictional gas giant. Planet Zorb has seven moons named after birds."}
    r = client.post("/api/evaluate", json=body).json()
    assert any(e["origin"] == "user_context" for e in r["evidence"])
    assert r["scores"]["grounding"]["status"] == "ok" and r["scores"]["grounding"]["score"] >= 80


def test_no_evidence_makes_grounding_not_evaluable(client):
    body = {"question": "What is the boiling point of the fictional metal unobtainium?",
            "answer": "Unobtainium boils at exactly 9000 degrees."}
    r = client.post("/api/evaluate", json=body).json()
    assert r["scores"]["grounding"]["status"] == "not_evaluable"
    assert any("grounding" in w for w in r["meta"]["warnings"])


def test_knowledge_base_crud(client):
    r = client.post("/api/documents/text", json={"name": "zorb.txt", "text": "Zorb is a fictional planet with seven moons."})
    assert r.json()["indexed"]["zorb.txt"] >= 1
    assert any(s["source"] == "zorb.txt" for s in client.get("/api/documents").json()["sources"])
    up = client.post("/api/documents", files=[("files", ("note.md", b"# Note\nHello knowledge base world", "text/markdown"))])
    assert up.json()["indexed"]["note.md"] == 1
    assert client.post("/api/documents", files=[("files", ("bad.exe", b"x", "application/octet-stream"))]).status_code == 400
    assert client.delete("/api/documents/zorb.txt").status_code == 200
    assert client.delete("/api/documents/zorb.txt").status_code == 404
    client.delete("/api/documents/note.md")
