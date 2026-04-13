"""
ai/llm_manager.py — LLM backend manager.

Supports:
- Ollama (local models)
- Gemini (Google)
- Interaction modes:
  - chat  (current direct assistant replies)
  - tools (LangChain-style bounded tool-calling loop)
"""

from __future__ import annotations

import logging
import re
import traceback
from datetime import datetime
from typing import Any, Generator
from pydantic import SecretStr

from config import (
    AI_BACKEND,
    SYSTEM_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_INTERACTION_MODE,
    AGENT_MAX_STEPS,
)
from ai.tools import build_langchain_tools

log = logging.getLogger(__name__)


class LLMManager:
    """Unified LLM interface — backend and interaction mode are selected in config.py."""

    def __init__(self):
        self._backend = AI_BACKEND.lower().strip()
        self._interaction_mode = (LLM_INTERACTION_MODE or "chat").strip().lower()
        self._history: list[dict[str, str]] = []

        self._tool_mode_ready = False
        self._tool_llm = None
        self._tools: list = []
        self._tool_map: dict[str, Any] = {}

        if self._backend == "ollama":
            self._init_ollama()
        elif self._backend == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown AI_BACKEND: '{self._backend}'. Use 'ollama' or 'gemini'.")

        if self._interaction_mode == "tools":
            self._init_tool_mode()
        elif self._interaction_mode != "chat":
            log.warning("Unknown LLM_INTERACTION_MODE=%r, falling back to chat", self._interaction_mode)
            self._interaction_mode = "chat"

        log.info("LLM backend=%s mode=%s", self._backend, self._interaction_mode)

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_ollama(self):
        try:
            import ollama
            self._ollama = ollama
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
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._lc_gemini = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                api_key=SecretStr(GEMINI_API_KEY),
            )
            self._genai_client = None
            log.info("Gemini via LangChain — model: %s", GEMINI_MODEL)

    def _init_tool_mode(self):
        self._tools = build_langchain_tools()
        if not self._tools:
            log.warning("Tool mode enabled but no tools are active; reverting to chat mode")
            self._interaction_mode = "chat"
            return

        try:
            if self._backend == "ollama":
                try:
                    from langchain_ollama import ChatOllama
                except ImportError as exc:
                    raise RuntimeError(
                        "Ollama tool mode requires 'langchain-ollama'. "
                        "Install it with: pip install langchain-ollama"
                    ) from exc

                base_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI

                base_llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    api_key=SecretStr(GEMINI_API_KEY),
                    temperature=0,
                )

            self._tool_llm = base_llm.bind_tools(self._tools)
            self._tool_map = {tool.name: tool for tool in self._tools}
            self._tool_mode_ready = True
            log.info("Tool mode ready with tools: %s", ", ".join(self._tool_map.keys()))
        except Exception as exc:
            log.warning(
                "Could not enable tool mode (%s: %r). Falling back to chat mode.\n%s",
                type(exc).__name__,
                exc,
                traceback.format_exc(),
            )
            self._interaction_mode = "chat"
            self._tool_mode_ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a message and return the full response string."""
        self._history.append({"role": "user", "content": user_message})

        try:
            if self._interaction_mode == "tools" and self._tool_mode_ready:
                response = self._chat_tools()
            elif self._backend == "ollama":
                response = self._chat_ollama()
            else:
                response = self._chat_gemini()

            response = self._cleanup_tool_response(response) if self._interaction_mode == "tools" else response
            self._history.append({"role": "assistant", "content": response})
            self._log_response(response)
            return response

        except Exception as exc:
            log.error("LLM error: %s", exc)
            return f"[Error: {exc}]"

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Streaming version — yields text chunks as they arrive."""
        self._history.append({"role": "user", "content": user_message})
        full_response = ""

        try:
            if self._interaction_mode == "tools" and self._tool_mode_ready:
                # LangChain tool loop is executed step-wise internally; emit as one chunk.
                response = self._chat_tools()
                full_response = self._cleanup_tool_response(response)
                if response:
                    yield full_response
            elif self._backend == "ollama":
                for chunk in self._stream_ollama():
                    full_response += chunk
                    yield chunk
            else:
                for chunk in self._stream_gemini():
                    full_response += chunk
                    yield chunk

            full_response = self._cleanup_tool_response(full_response) if self._interaction_mode == "tools" else full_response
            self._history.append({"role": "assistant", "content": full_response})
            self._log_response(full_response)

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

    # ── Tool mode internals ───────────────────────────────────────────────────

    @staticmethod
    def _build_tools_system_prompt() -> str:
        now = datetime.now().astimezone()
        now_block = (
            f"Current local datetime: {now.isoformat()}\n"
            f"Current local date: {now.strftime('%Y-%m-%d')}\n"
            f"Current local year: {now.year}\n"
            f"Timezone: {now.tzname() or 'local'}"
        )

        tool_rules = (
            "Tools-mode rules:\n"
            "1) For any time-sensitive request (today, current year/date/time, latest, recently, now), "
            "use current_datetime and/or web_search before finalizing.\n"
            "2) Do not guess temporal facts from memory when a tool can verify them.\n"
            "3) If tools fail, say so clearly and provide a best-effort answer labeled uncertain.\n"
            "4) Synthesize tool results into natural conversational sentences; do not paste raw tool output.\n"
            "5) Do not output markdown separators, raw dumps, or long URL lists unless the user explicitly asks.\n"
            "6) For web answers, mention source names naturally and say things like 'The Verge says...' or 'According to Wired...'.\n"
            "7) Never say 'link' or 'links' unless the user explicitly asks for links. Never show raw URLs in the spoken answer."
        )

        return f"{SYSTEM_PROMPT}\n\n{now_block}\n\n{tool_rules}"

    def _chat_tools(self) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        messages: list[Any] = [SystemMessage(content=self._build_tools_system_prompt())]
        for msg in self._history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        for _ in range(AGENT_MAX_STEPS):
            ai_msg = self._tool_llm.invoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                text = self._coerce_content(ai_msg.content)
                clean = self._cleanup_tool_response(text)
                return clean or "I could not produce a response."

            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {})
                tool_call_id = call.get("id", "")
                tool_impl = self._tool_map.get(name)

                if tool_impl is None:
                    result = f"Tool '{name}' is not available."
                else:
                    try:
                        payload = args if isinstance(args, dict) else args
                        result = tool_impl.invoke(payload)
                    except Exception as exc:
                        result = f"Tool '{name}' failed: {exc}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

        return "I reached my tool-step limit before finishing. Please ask me to continue."

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _cleanup_tool_response(text: str) -> str:
        """Make tool-mode output sound natural for chat/TTS by removing raw formatting artifacts."""
        if not text:
            return ""

        out = text.replace("\r\n", "\n")
        out = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", out)
        out = re.sub(r"(?m)^\s*[-=*_]{3,}\s*$", "", out)  # markdown separators like ---
        out = re.sub(r"(?m)^\s*[-=*_]{2,}\s*(https?://\S+|www\.\S+)\s*$", lambda m: f"Source: {LLMManager._friendly_source_name(m.group(1))}", out)
        out = re.sub(r"(?m)^\s*(https?://\S+|www\.\S+)\s*$", lambda m: f"Source: {LLMManager._friendly_source_name(m.group(1))}", out)
        out = re.sub(r"(?i)https?://\S+|www\.\S+", lambda m: LLMManager._friendly_source_name(m.group(0)), out)
        out = re.sub(r"(?i)\blinks?\b", "", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = re.sub(r"\s{2,}", " ", out)
        return out.strip()

    @staticmethod
    def _friendly_source_name(url: str) -> str:
        cleaned = re.sub(r"^https?://", "", (url or "").strip(), flags=re.I)
        cleaned = re.sub(r"^www\.", "", cleaned, flags=re.I)
        cleaned = cleaned.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if not cleaned:
            return "Unknown source"

        first_label = cleaned.split(".", 1)[0]
        parts = [p for p in re.split(r"[-_.]+", first_label) if p]
        if not parts:
            return "Unknown source"
        if parts[0].lower() == "the" and len(parts) > 1:
            return "The " + " ".join(p.capitalize() for p in parts[1:])
        return " ".join(p.capitalize() for p in parts)

    @staticmethod
    def _log_response(response: str):
        if not response:
            log.info("Assistant response: <empty>")
            return
        preview = response if len(response) <= 1000 else response[:1000] + "…"
        log.info("Assistant response (%d chars): %s", len(response), preview)

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
        parts = []
        for msg in self._history:
            role = "user" if msg["role"] == "user" else "model"
            parts.append({"role": role, "parts": [{"text": msg["content"]}]})
        return parts

    def _chat_gemini(self) -> str:
        if self._genai_client is not None:
            from google.genai import types as genai_types

            response = self._genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=self._build_gemini_contents(),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                ),
            )
            return response.text

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_msgs: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
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
                ),
            ):
                if chunk.text:
                    yield chunk.text
            return

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_msgs: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in self._history:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        for chunk in self._lc_gemini.stream(lc_msgs):
            if chunk.content:
                yield chunk.content

