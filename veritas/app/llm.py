"""LLM access layer built on LangChain.

* `LLMHub.structured(...)` builds an LCEL chain:  ChatPromptTemplate | chat_model.with_structured_output(Schema)
  with an automatic fallback chain that asks for plain JSON and parses it with PydanticOutputParser.
* Provider is chosen with LLM_PROVIDER. If a cloud provider has no API key the hub drops to the
  built-in offline heuristic model (and says so in /api/health) so the app always starts.
"""
import asyncio
import logging
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import Settings
from app.offline import OfflineChatModel

logger = logging.getLogger(__name__)


class LLMHub:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sem: Optional[asyncio.Semaphore] = None
        self.warning = ""
        requested = settings.llm_provider.lower().strip()
        self.mode, self.model = self._resolve(requested)

    # ------------------------------------------------------------------ setup
    def _resolve(self, requested: str) -> tuple[str, str]:
        s = self.settings
        table = {
            "groq": (s.groq_api_key, s.groq_model),
            "openai": (s.openai_api_key, s.openai_model),
            "anthropic": (s.anthropic_api_key, s.anthropic_model),
            "ollama": ("not-needed", s.ollama_model),
        }
        if requested == "offline":
            return "offline", "heuristic-critics-v1"
        if requested not in table:
            self.warning = f"Unknown LLM_PROVIDER '{requested}'. Running in offline demo mode."
            logger.warning(self.warning)
            return "offline", "heuristic-critics-v1"
        key, model = table[requested]
        if not key:
            self.warning = (f"No API key for provider '{requested}'. Running in OFFLINE DEMO MODE "
                            f"(heuristic critics). Add the key to .env for real LLM critics.")
            logger.warning(self.warning)
            return "offline", "heuristic-critics-v1"
        return requested, model

    def check_model(self) -> None:
        """Groq only: warn early (instead of failing per request) if the model is not available to this key."""
        if self.mode != "groq":
            return
        try:
            from groq import Groq

            ids = [m.id for m in Groq(api_key=self.settings.groq_api_key, timeout=15).models.list().data]
        except Exception as exc:  # noqa: BLE001 - network hiccup: skip the check
            logger.info("Could not list Groq models (%s); skipping model check", exc)
            return
        if self.model not in ids:
            skip = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
            usable = [i for i in sorted(ids) if not any(x in i.lower() for x in skip)]
            self.warning = (f"Model '{self.model}' is not available to this Groq API key. Set GROQ_MODEL in .env to "
                            f"one of: {', '.join(usable)} and restart.")
            logger.warning(self.warning)

    def chat(self, temperature: float = 0.0) -> BaseChatModel:
        s = self.settings
        if self.mode == "offline":
            return OfflineChatModel()
        common = dict(temperature=temperature, timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)
        if self.mode == "groq":
            return init_chat_model(s.groq_model, model_provider="groq", api_key=s.groq_api_key, **common)
        if self.mode == "openai":
            return init_chat_model(s.openai_model, model_provider="openai", api_key=s.openai_api_key, **common)
        if self.mode == "anthropic":
            return init_chat_model(s.anthropic_model, model_provider="anthropic",
                                   api_key=s.anthropic_api_key, max_tokens=2048, **common)
        if self.mode == "ollama":
            return init_chat_model(s.ollama_model, model_provider="ollama",
                                   base_url=s.ollama_base_url, temperature=temperature)
        raise RuntimeError(f"Unsupported mode {self.mode}")

    # ----------------------------------------------------------------- chains
    def _with_schema(self, llm: BaseChatModel, schema: type[BaseModel]) -> Runnable:
        if self.mode == "openai":
            return llm.with_structured_output(schema, method="function_calling")
        return llm.with_structured_output(schema)

    def structured(self, schema: type[BaseModel], system: str, human: str, temperature: float = 0.0) -> Runnable:
        """prompt | llm.with_structured_output(schema)   (+ JSON-parser fallback)."""
        llm = self.chat(temperature)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        if self.mode == "offline":
            return prompt | self._with_schema(llm, schema)
        parser = PydanticOutputParser(pydantic_object=schema)
        fb_prompt = ChatPromptTemplate.from_messages(
            [("system", system + "\n\n{format_instructions}\nReturn ONLY the JSON object."), ("human", human)]
        ).partial(format_instructions=parser.get_format_instructions())
        json_chain = fb_prompt | llm | parser
        try:
            primary = prompt | self._with_schema(llm, schema)
        except (NotImplementedError, TypeError, ValueError):  # model has no native structured output
            return json_chain
        return primary.with_fallbacks([json_chain])

    def text(self, system: str, human: str, temperature: float = 0.7) -> Runnable:
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        return prompt | self.chat(temperature) | StrOutputParser()

    async def ainvoke(self, chain: Runnable, inputs: dict):
        """Run a chain with a global concurrency limit (protects free-tier rate limits)."""
        if self._sem is None:
            self._sem = asyncio.Semaphore(max(1, self.settings.max_concurrency))
        async with self._sem:
            return await chain.ainvoke(inputs)
