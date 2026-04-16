"""
TalkingMAC — Retro Macintosh AI Assistant
==========================================

Entry point. Wires all subsystems together.

Run from terminal:
    python3 main.py
    AI_BACKEND=gemini python3 main.py
    LLM_INTERACTION_MODE=tools python3 main.py
    UI_WINDOW_MODE=windowed UI_WINDOW_WIDTH=1200 UI_WINDOW_HEIGHT=760 python3 main.py

Run from PyCharm:
    Set `main.py` as run target, then set Environment variables, for example:
    AI_BACKEND=ollama;LLM_INTERACTION_MODE=chat;UI_WINDOW_MODE=fullscreen
"""

import logging
import os
import queue
import re
import sys
import threading
import time

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("talkingmac")

# Ensure project root on path (needed for PyCharm + terminal)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AI_BACKEND, IDLE_SLEEP_TIMEOUT_SECS, WAKE_WORD_INTERRUPTION_ENABLED
from ui.app import TalkingMACApp
from ui.expressions import Expression
from ai.llm_manager import LLMManager
from ai.mcp_client import MCPClient
from voice.audio_capture import AudioCapture
from voice.tts import TTSEngine
from voice.stt import STTEngine
from voice.wake_word import WakeWordDetector


class TalkingMACAssistant:
    """
    Top-level orchestrator.

    State machine:
        IDLE
          ↓  wake word spoken OR text typed
        LISTENING (face listens, STT records)
          ↓
        THINKING (LLM generating)
          ↓  first chunk received
        TALKING (TTS starts while stream continues)
        TALKING  ← face animates mouth while audio plays
          ↓  TTS finishes
        IDLE
    """

    def __init__(self):
        log.info("Booting TalkingMAC — backend: %s", AI_BACKEND)

        # Shared mic capture (started before STT / wake word)
        self._capture = AudioCapture()

        # UI first so we can show errors
        self._ui  = TalkingMACApp()
        self._tts = TTSEngine()
        self._stt = STTEngine(self._capture)
        self._ww  = WakeWordDetector(self._capture, on_wake=self._on_wake_word)
        self._llm = self._init_llm()
        self._mcp = MCPClient()

        # Wire TTS → face animation (this is the key sync fix)
        self._tts.on_speaking_start = self._on_tts_start
        self._tts.on_speaking_end   = self._on_tts_end

        # Wire UI callbacks
        self._ui.on_text_input = self._handle_text_input
        self._ui.on_quit       = self._shutdown

        # Prevent overlapping queries
        self._busy = threading.Lock()
        self._wake_handler_lock = threading.Lock()
        self._interrupt_requested = threading.Event()
        self._wake_lock = threading.Lock()
        self._wake_pause_depth = 0
        self._activity_lock = threading.Lock()
        self._last_activity_ts = time.monotonic()
        self._sleeping = False
        self._sleep_timeout_secs = max(0.0, IDLE_SLEEP_TIMEOUT_SECS)
        self._stop_event = threading.Event()

        # Streaming TTS chunking tuned for low-latency but natural phrasing.
        self._tts_stream_sentence_split = re.compile(r"(?<=[.!?])\s+|(?<=[,;:])\s+(?=[A-Z0-9])")
        self._tts_stream_soft_chunk_chars = 120
        self._tts_stream_hard_chunk_chars = 170
        self._tts_stream_min_first_chunk_chars = 48
        self._tts_stream_min_partial_chunk_chars = 32
        self._tts_stream_partial_flush_interval_secs = 0.35
        self._tts_stream_coalesce_window_secs = 0.12
        self._tts_stream_coalesce_target_chars = 210

    # ── Boot ──────────────────────────────────────────────────────────────────

    def run(self):
        # Start microphone FIRST (both wake word and STT need it)
        try:
            self._capture.start()
        except Exception as exc:
            log.error("Microphone failed: %s", exc)
            self._ui.show_status(f"Mic error: {exc} — text input still works", ttl=12)

        # MCP init in background (non-blocking)
        threading.Thread(target=self._init_mcp, daemon=True).start()

        # Wake word detector
        self._ww.start()

        # Greeting in background
        threading.Thread(target=self._greet, daemon=True).start()

        # Auto-sleep face after idle timeout
        threading.Thread(target=self._idle_sleep_loop, daemon=True).start()

        self._ui.set_mode_label("IDLE")
        log.info("TalkingMAC ready")

        # Blocks until ESC / window close
        self._ui.run()
        self._shutdown()

    # ── TTS sync callbacks ────────────────────────────────────────────────────

    def _on_tts_start(self):
        """Called by TTS worker thread the moment audio starts playing."""
        self._ww.set_tts_active(True)
        self._ui.set_expression(Expression.TALKING)
        self._ui.set_mode_label("TALKING")

    def _on_tts_end(self):
        """Called by TTS worker thread when audio finishes."""
        self._ww.set_tts_active(False)
        self._ui.set_expression(Expression.IDLE)
        self._ui.set_mode_label("IDLE")

    # ── Wake word ─────────────────────────────────────────────────────────────

    def _on_wake_word(self):
        # Keep wake detection thread responsive: handle wake work asynchronously.
        threading.Thread(target=self._handle_wake_word_event, daemon=True).start()

    def _handle_wake_word_event(self):
        wake_lock_held = self._wake_handler_lock.acquire(blocking=False)
        if not wake_lock_held:
            return  # drop duplicate wake events while one is being handled

        busy_now = self._busy.locked()
        barge_in = WAKE_WORD_INTERRUPTION_ENABLED and (busy_now or self._tts.is_speaking)

        if barge_in:
            log.info("Wake word during active response: requesting interruption")
            self._interrupt_requested.set()
            self._tts.stop()
            self._busy.acquire()
        elif not self._busy.acquire(blocking=False):
            self._wake_handler_lock.release()
            return  # already handling something

        try:
            self._mark_activity()
            self._interrupt_requested.clear()
            self._ui.set_look_target(0.0, -0.3)
            self._ui.set_expression(Expression.HAPPY)
            time.sleep(0.15)
            self._set_state(Expression.LISTENING, "LISTENING")
            self._ui.set_look_target(0.0, 0.45)
            if barge_in:
                self._ui.show_status("Interrupted. Listening…", ttl=10)
            else:
                self._ui.show_status("Listening…", ttl=10)

            # Pause wake detection only while capturing the user's utterance.
            self._pause_wake_detection()
            try:
                text = self._stt.listen()
            finally:
                self._resume_wake_detection()

            if not text:
                self._ui.show_status("Didn't catch that — say 'hey mac' again", ttl=4)
                self._set_state(Expression.IDLE, "IDLE")
                return

            # Allow future wake events while we are thinking/talking so barge-in works.
            self._wake_handler_lock.release()
            wake_lock_held = False
            self._process_query(text)
        finally:
            self._busy.release()
            if wake_lock_held:
                self._wake_handler_lock.release()

    # ── Text input from keyboard ───────────────────────────────────────────────

    def _handle_text_input(self, text: str):
        clean = text.strip()
        if not clean:
            return

        if clean.lower() in {"/s", "/sleep"}:
            self._set_sleep_face()
            return

        self._mark_activity()
        if not self._busy.acquire(blocking=False):
            self._ui.show_status("Still thinking… wait a moment", ttl=3)
            return
        try:
            self._process_query(clean)
        finally:
            self._busy.release()

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _process_query(self, user_text: str):
        log.info("Query: %r", user_text)
        self._mark_activity()
        self._interrupt_requested.clear()

        if self._llm is None:
            self._set_state(Expression.ERROR, "ERROR")
            self._ui.show_status("No LLM — check config / Ollama", ttl=8)
            return

        # ── Think (LLM generates full response) ──────────────────────────────
        self._ui.set_look_target(0.0, -0.12)
        self._set_state(Expression.THINKING, "THINKING")
        self._ui.show_status("", ttl=0)

        tts_queue: queue.Queue[str | None] = queue.Queue()
        tts_worker = threading.Thread(
            target=self._stream_tts_worker,
            args=(tts_queue,),
            daemon=True,
            name="StreamingTTS",
        )
        tts_worker.start()
        tts_buffer = ""
        has_spoken_chunk = False
        last_tts_emit_ts = time.monotonic()

        try:
            full_response = ""
            for chunk in self._llm.chat_stream(user_text):
                if self._interrupt_requested.is_set():
                    log.info("Response interrupted during thinking")
                    self._set_state(Expression.IDLE, "IDLE")
                    return
                full_response += chunk

                tts_buffer += chunk
                ready_chunks, tts_buffer = self._extract_tts_chunks(tts_buffer)
                if not ready_chunks:
                    now = time.monotonic()
                    buffered = tts_buffer.strip()
                    min_chars = (
                        self._tts_stream_min_partial_chunk_chars
                        if has_spoken_chunk
                        else self._tts_stream_min_first_chunk_chars
                    )
                    can_force_emit = (
                        buffered
                        and len(buffered) >= min_chars
                        and (now - last_tts_emit_ts) >= self._tts_stream_partial_flush_interval_secs
                        and tts_queue.qsize() <= 2
                    )
                    if can_force_emit:
                        ready_chunks = [buffered]
                        tts_buffer = ""

                for spoken_chunk in ready_chunks:
                    tts_queue.put(spoken_chunk)
                    has_spoken_chunk = True
                    last_tts_emit_ts = time.monotonic()

            tail_chunks, tts_buffer = self._extract_tts_chunks(tts_buffer, flush=True)
            for spoken_chunk in tail_chunks:
                tts_queue.put(spoken_chunk)
                has_spoken_chunk = True
                last_tts_emit_ts = time.monotonic()

            if not full_response.strip():
                raise ValueError("Empty LLM response")

            log.info("Response ready (%d chars)", len(full_response))

        except Exception as exc:
            log.error("LLM error: %s", exc)
            self._set_state(Expression.ERROR, "ERROR")
            self._ui.show_status(f"Error: {exc}", ttl=8)
            time.sleep(2)
            self._set_state(Expression.IDLE, "IDLE")
            return
        finally:
            tts_queue.put(None)
            tts_worker.join(timeout=60)

        # ── Streaming speak complete ─────────────────────────────────────────
        if self._interrupt_requested.is_set():
            log.info("Response interrupted before speaking")
            self._set_state(Expression.IDLE, "IDLE")
            return
        # on_tts_end callback already set face back to IDLE

    def _extract_tts_chunks(self, text_buffer: str, flush: bool = False) -> tuple[list[str], str]:
        """Split model stream text into speakable chunks without waiting for full completion."""
        normalized = text_buffer.replace("\r\n", "\n").replace("\n\n", ". ")
        parts = self._tts_stream_sentence_split.split(normalized)
        chunks: list[str] = []

        if len(parts) > 1:
            chunks.extend(p.strip() for p in parts[:-1] if p.strip())
            remainder = parts[-1]
        else:
            remainder = normalized

        remainder_clean = remainder.strip()
        if flush:
            if remainder_clean:
                chunks.append(remainder_clean)
            return chunks, ""

        # Fallback chunking for long punctuation-free outputs.
        if len(remainder_clean) >= self._tts_stream_hard_chunk_chars:
            split_at = remainder_clean.rfind(" ", 0, self._tts_stream_soft_chunk_chars)
            if split_at <= 0:
                split_at = self._tts_stream_soft_chunk_chars
            chunks.append(remainder_clean[:split_at].strip())
            remainder = remainder_clean[split_at:].lstrip()

        return chunks, remainder

    def _stream_tts_worker(self, tts_queue: "queue.Queue[str | None]"):
        stop_after_speak = False
        while not self._interrupt_requested.is_set():
            try:
                payload = tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if payload is None:
                return
            if not payload.strip() or self._interrupt_requested.is_set():
                continue

            # Coalesce tiny neighboring chunks to avoid choppy start/stop prosody.
            merged = payload.strip()
            deadline = time.monotonic() + self._tts_stream_coalesce_window_secs
            while (
                len(merged) < self._tts_stream_coalesce_target_chars
                and not re.search(r"[.!?][\"')\]]?\s*$", merged)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = tts_queue.get(timeout=remaining)
                except queue.Empty:
                    break

                if nxt is None:
                    stop_after_speak = True
                    break

                nxt = nxt.strip()
                if not nxt:
                    continue
                if merged and not merged.endswith((" ", "\n")) and not nxt.startswith((".", ",", "!", "?", ";", ":")):
                    merged += " "
                merged += nxt

            self._tts.speak(merged)
            if stop_after_speak:
                return

    # ── Greeting ──────────────────────────────────────────────────────────────

    def _greet(self):
        time.sleep(1.0)
        msg = "Hello! I'm Mac. Say the wake word to talk, or just type below."
        self._set_state(Expression.HAPPY, "HAPPY")
        self._tts.speak(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_state(self, expr: Expression, label: str):
        self._ui.set_expression(expr)
        self._ui.set_mode_label(label)

    def _set_sleep_face(self):
        with self._activity_lock:
            self._sleeping = True
            self._last_activity_ts = time.monotonic()
        self._ui.set_look_target(0.0, 0.0)
        self._set_state(Expression.SLEEPING, "SLEEP")

    def _mark_activity(self):
        should_wake_visual = False
        with self._activity_lock:
            self._last_activity_ts = time.monotonic()
            if self._sleeping:
                self._sleeping = False
                should_wake_visual = True

        # Wake the face only if we're not currently in another active state.
        if should_wake_visual and not self._busy.locked() and not self._tts.is_speaking:
            self._set_state(Expression.IDLE, "IDLE")

    def _idle_sleep_loop(self):
        while not self._stop_event.is_set():
            if self._sleep_timeout_secs <= 0:
                time.sleep(0.5)
                continue

            if self._busy.locked() or self._tts.is_speaking:
                time.sleep(0.25)
                continue

            with self._activity_lock:
                idle_for = time.monotonic() - self._last_activity_ts
                should_sleep = idle_for >= self._sleep_timeout_secs and not self._sleeping

            if should_sleep:
                self._set_sleep_face()

            time.sleep(0.25)

    def _pause_wake_detection(self):
        with self._wake_lock:
            self._wake_pause_depth += 1
            if self._wake_pause_depth == 1:
                self._ww.pause()

    def _resume_wake_detection(self):
        with self._wake_lock:
            if self._wake_pause_depth == 0:
                return
            self._wake_pause_depth -= 1
            if self._wake_pause_depth == 0:
                self._ww.resume()

    def _init_llm(self) -> LLMManager | None:
        try:
            return LLMManager()
        except Exception as exc:
            log.error("LLM init failed: %s", exc)
            self._ui.show_status(f"LLM Error: {exc}", ttl=15)
            return None

    def _init_mcp(self):
        import asyncio
        asyncio.run(self._mcp.start())
        if self._mcp.enabled:
            log.info("MCP tools: %s", self._mcp.tool_names())

    def _shutdown(self):
        log.info("Shutting down…")
        self._stop_event.set()
        self._ww.stop()
        self._capture.stop()
        self._tts.shutdown()
        if self._ui.is_running():
            self._ui.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    TalkingMACAssistant().run()


if __name__ == "__main__":
    main()
