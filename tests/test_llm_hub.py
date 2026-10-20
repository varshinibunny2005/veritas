"""Exercise the real-provider code path (structured output + JSON fallback) with a fake chat model."""
import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.llm import LLMHub
from app.prompts import CLAIM_HUMAN, CLAIM_SYSTEM
from app.schemas import AccuracyAssessment, ClaimExtraction


class FakeLLM(BaseChatModel):
    structured_broken: bool = True
    support_structured: bool = True

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        # returns JSON wrapped in a markdown fence, like many chat models do
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='```json\n{"claims": ["A is B."]}\n```'))])

    def with_structured_output(self, schema, **kwargs):
        if not self.support_structured:
            raise NotImplementedError
        def boom(_):
            raise ValueError("tool call failed")
        return RunnableLambda(boom)


def make_hub(**fake_kwargs) -> LLMHub:
    hub = LLMHub(Settings(llm_provider="groq", groq_api_key="dummy"))
    assert hub.mode == "groq"
    hub.chat = lambda temperature=0.0: FakeLLM(**fake_kwargs)  # type: ignore[method-assign]
    return hub


def run(hub):
    chain = hub.structured(ClaimExtraction, CLAIM_SYSTEM, CLAIM_HUMAN)
    return asyncio.run(hub.ainvoke(chain, {"max_claims": 5, "question": "q?", "answer": "A is B."}))


def test_falls_back_to_json_parsing_when_tool_call_fails():
    assert run(make_hub()).claims == ["A is B."]


def test_json_chain_used_when_model_has_no_structured_output():
    assert run(make_hub(support_structured=False)).claims == ["A is B."]


def test_missing_key_drops_to_offline_mode():
    hub = LLMHub(Settings(llm_provider="groq", groq_api_key=None))
    assert hub.mode == "offline" and "OFFLINE" in hub.warning


def test_real_provider_client_builds_without_network():
    hub = LLMHub(Settings(llm_provider="groq", groq_api_key="dummy"))
    assert type(hub.chat(0.0)).__name__ == "ChatGroq"


def test_llm_output_normalisation():
    a = AccuracyAssessment(claim_id=1, assessment="Wrong", severity="Major")
    assert a.assessment == "incorrect" and a.severity == "high"


def test_claim_ids_like_c2_are_accepted():
    from app.schemas import GroundingVerdict
    assert GroundingVerdict(claim_id="C2", verdict="Partial").claim_id == 2


def test_model_check_warns_when_model_missing(monkeypatch):
    import types, sys
    fake = types.SimpleNamespace(data=[types.SimpleNamespace(id="model-a"), types.SimpleNamespace(id="whisper-x")])
    class FakeGroq:
        def __init__(self, **kw): self.models = types.SimpleNamespace(list=lambda: fake)
    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    hub = LLMHub(Settings(llm_provider="groq", groq_api_key="dummy", groq_model="retired-model"))
    hub.check_model()
    assert "model-a" in hub.warning and "whisper" not in hub.warning


def test_llm_failure_gives_clear_error():
    import pytest
    from app.arbiter import ArbitrationError
    from app.config import Settings as S
    from app.services import build_services
    from app.schemas import EvaluationRequest
    import tempfile, pathlib
    sv = build_services(S(llm_provider="groq", groq_api_key="dummy", embedding_provider="hash",
                          chroma_dir=pathlib.Path(tempfile.mkdtemp())))
    sv.hub.chat = lambda temperature=0.0: FakeLLM(support_structured=False)  # type: ignore[method-assign]
    class Dead(FakeLLM):
        def _generate(self, *a, **k):
            raise RuntimeError("404 model_not_found")
    sv.hub.chat = lambda temperature=0.0: Dead(support_structured=False)  # type: ignore[method-assign]
    with pytest.raises(ArbitrationError, match="model_not_found"):
        asyncio.run(sv.evaluate(EvaluationRequest(question="What is X?", answer="X is a thing that exists.")))
