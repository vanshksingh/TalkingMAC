"""
ai/llm_manager.py — LLM backend manager.

Supports:
  • Ollama   — local models via the ollama Python client
  • Gemini   — Google Gemini via langchain-google-genai

Usage:
    mgr = LLMManager()
    response = mgr.chat("Tell me a joke")
"""

import logging
from typing import Generator

from config import (
    AI_BACKEND, SYSTEM_PROMPT,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)

log = logging.getLogger(__name__)


class LLMManager:
    """Unified LLM interface — choose backend in config.py."""

    def __init__(self):
        self._backend = AI_BACKEND.lower().strip()
        self._history: list[dict] = []   # simple in-memory chat history

        if self._backend == "ollama":
            self._init_ollama()
        elif self._backend == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown AI_BACKEND: '{self._backend}'. Use 'ollama' or 'gemini'.")

        log.info("LLM backend: %s", self._backend)

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_ollama(self):
        try:
            import ollama
            self._ollama = ollama
            # Quick health check
            self._ollama.list()
            log.info("Ollama connected at %s — model: %s", OLLAMA_BASE_URL, OLLAMA_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {OLLAMA_BASE_URL}.\n"
                f"Make sure Ollama is running (`ollama serve`).\nError: {exc}"
            ) from exc

    def _init_gemini(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to .env or set the environment variable."
            )
        try:
            import google.genai as genai
            self._genai_client = genai.Client(api_key=GEMINI_API_KEY)
            log.info("Gemini client ready — model: %s", GEMINI_MODEL)
        except ImportError:
            # Fallback to langchain-google-genai
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._lc_gemini = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
            )
            self._genai_client = None
            log.info("Gemini via LangChain — model: %s", GEMINI_MODEL)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Send a message and return the full response string.
        Maintains conversation history automatically.
        """
        self._history.append({"role": "user", "content": user_message})

        try:
            if self._backend == "ollama":
                response = self._chat_ollama()
            else:
                response = self._chat_gemini()

            self._history.append({"role": "assistant", "content": response})
            return response

        except Exception as exc:
            log.error("LLM error: %s", exc)
            return f"[Error: {exc}]"

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Streaming version — yields text chunks as they arrive.
        Full response is still added to history.
        """
        self._history.append({"role": "user", "content": user_message})
        full_response = ""

        try:
            if self._backend == "ollama":
                for chunk in self._stream_ollama():
                    full_response += chunk
                    yield chunk
            else:
                for chunk in self._stream_gemini():
                    full_response += chunk
                    yield chunk

            self._history.append({"role": "assistant", "content": full_response})

        except Exception as exc:
            log.error("LLM stream error: %s", exc)
            yield f"[Error: {exc}]"

    def clear_history(self):
        self._history.clear()

    @property
    def backend_name(self) -> str:
        if self._backend == "ollama":
            return f"Ollama/{OLLAMA_MODEL}"
        return f"Gemini/{GEMINI_MODEL}"

    # ── Ollama internals ──────────────────────────────────────────────────────

    def _build_ollama_messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs.extend(self._history)
        return msgs

    def _chat_ollama(self) -> str:
        resp = self._ollama.chat(
            model=OLLAMA_MODEL,
            messages=self._build_ollama_messages(),
        )
        return resp["message"]["content"]

    def _stream_ollama(self) -> Generator[str, None, None]:
        stream = self._ollama.chat(
            model=OLLAMA_MODEL,
            messages=self._build_ollama_messages(),
            stream=True,
        )
        for chunk in stream:
            text = chunk["message"]["content"]
            if text:
                yield text

    # ── Gemini internals ──────────────────────────────────────────────────────

    def _build_gemini_contents(self) -> list:
        """Convert history to Gemini API format."""
        parts = []
        for msg in self._history:
            role = "user" if msg["role"] == "user" else "model"
            parts.append({"role": role, "parts": [{"text": msg["content"]}]})
        return parts

    def _chat_gemini(self) -> str:
        if self._genai_client is not None:
            # google.genai native
            from google.genai import types as genai_types
            response = self._genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=self._build_gemini_contents(),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                )
            )
            return response.text
        else:
            # LangChain fallback
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            lc_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
            for msg in self._history:
                if msg["role"] == "user":
                    lc_msgs.append(HumanMessage(content=msg["content"]))
                else:
                    lc_msgs.append(AIMessage(content=msg["content"]))
            result = self._lc_gemini.invoke(lc_msgs)
            return result.content

    def _stream_gemini(self) -> Generator[str, None, None]:
        if self._genai_client is not None:
            from google.genai import types as genai_types
            for chunk in self._genai_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=self._build_gemini_contents(),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                )
            ):
                if chunk.text:
                    yield chunk.text
        else:
            # LangChain streaming fallback
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            lc_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
            for msg in self._history:
                if msg["role"] == "user":
                    lc_msgs.append(HumanMessage(content=msg["content"]))
                else:
                    lc_msgs.append(AIMessage(content=msg["content"]))
            for chunk in self._lc_gemini.stream(lc_msgs):
                if chunk.content:
                    yield chunk.content
