"""
TalkingMAC — Retro Macintosh AI Assistant
==========================================

Entry point.  Wires all subsystems together.

Run:
    python3 main.py
    AI_BACKEND=gemini python3 main.py

PyCharm: set main.py as run target.
"""

import logging
import os
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

from config import AI_BACKEND
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
          ↓  first token received
        [wait for LLM to finish streaming]
          ↓  TTS starts playing
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

        self._ui.set_mode_label("IDLE")
        log.info("TalkingMAC ready")

        # Blocks until ESC / window close
        self._ui.run()
        self._shutdown()

    # ── TTS sync callbacks ────────────────────────────────────────────────────

    def _on_tts_start(self):
        """Called by TTS worker thread the moment audio starts playing."""
        self._ui.set_expression(Expression.TALKING)
        self._ui.set_mode_label("TALKING")

    def _on_tts_end(self):
        """Called by TTS worker thread when audio finishes."""
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
        barge_in = busy_now or self._tts.is_speaking

        if barge_in:
            log.info("Wake word during active response: requesting interruption")
            self._interrupt_requested.set()
            self._tts.stop()
            self._busy.acquire()
        elif not self._busy.acquire(blocking=False):
            self._wake_handler_lock.release()
            return  # already handling something

        try:
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
        if not self._busy.acquire(blocking=False):
            self._ui.show_status("Still thinking… wait a moment", ttl=3)
            return
        try:
            self._process_query(text)
        finally:
            self._busy.release()

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _process_query(self, user_text: str):
        log.info("Query: %r", user_text)
        self._interrupt_requested.clear()

        if self._llm is None:
            self._set_state(Expression.ERROR, "ERROR")
            self._ui.show_status("No LLM — check config / Ollama", ttl=8)
            return

        # ── Think (LLM generates full response) ──────────────────────────────
        self._set_state(Expression.THINKING, "THINKING")
        self._ui.show_status("", ttl=0)

        try:
            full_response = ""
            for chunk in self._llm.chat_stream(user_text):
                if self._interrupt_requested.is_set():
                    log.info("Response interrupted during thinking")
                    self._set_state(Expression.IDLE, "IDLE")
                    return
                full_response += chunk

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

        # ── Speak (TTS callbacks drive TALKING ↔ IDLE transitions) ───────────
        # Face goes TALKING exactly when audio starts via on_tts_start callback
        if self._interrupt_requested.is_set():
            log.info("Response interrupted before speaking")
            self._set_state(Expression.IDLE, "IDLE")
            return
        self._tts.speak(full_response)
        # on_tts_end callback already set face back to IDLE

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
