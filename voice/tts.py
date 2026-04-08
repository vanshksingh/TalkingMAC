"""
voice/tts.py — Text-to-Speech engine.

Uses pyttsx3 (offline).  Runs in a dedicated thread.

on_speaking_start / on_speaking_end callbacks let the UI sync
the mouth animation exactly with actual audio playback.
"""

import logging
import queue
import threading
from typing import Callable, Optional

from config import TTS_RATE, TTS_VOLUME

log = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self._q        = queue.Queue()
        self._speaking = threading.Event()
        self._done     = threading.Event()
        self._thread   = threading.Thread(target=self._worker, daemon=True, name="TTS")
        self._thread.start()

        # Sync callbacks — set these before calling speak()
        self.on_speaking_start: Optional[Callable] = None
        self.on_speaking_end:   Optional[Callable] = None

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def speak(self, text: str):
        """Enqueue and block until spoken."""
        if not text.strip():
            return
        self._done.clear()
        self._q.put(("speak", text))
        self._done.wait()

    def speak_async(self, text: str):
        """Enqueue and return immediately."""
        if text.strip():
            self._q.put(("speak", text))

    def stop(self):
        self._q.put(("stop", None))

    def shutdown(self):
        self._q.put(("quit", None))
        self._thread.join(timeout=4)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
            log.info("TTS ready (pyttsx3, rate=%d)", TTS_RATE)
        except Exception as exc:
            log.error("TTS init failed: %s", exc)
            return

        while True:
            try:
                cmd, payload = self._q.get(timeout=1)
            except queue.Empty:
                continue

            if cmd == "quit":
                break

            elif cmd == "stop":
                try:
                    engine.stop()
                except Exception:
                    pass
                self._speaking.clear()
                self._done.set()
                if self.on_speaking_end:
                    try:
                        self.on_speaking_end()
                    except Exception:
                        pass

            elif cmd == "speak":
                self._speaking.set()
                if self.on_speaking_start:
                    try:
                        self.on_speaking_start()
                    except Exception:
                        pass
                try:
                    engine.say(payload)
                    engine.runAndWait()
                except Exception as exc:
                    log.warning("TTS speak error: %s", exc)
                finally:
                    self._speaking.clear()
                    self._done.set()
                    if self.on_speaking_end:
                        try:
                            self.on_speaking_end()
                        except Exception:
                            pass
