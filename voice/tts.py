"""
voice/tts.py — Text-to-Speech engine.

Uses pyttsx3 (offline).  Runs in a dedicated thread.

on_speaking_start / on_speaking_end callbacks let the UI sync
the mouth animation exactly with actual audio playback.
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from config import TTS_RATE, TTS_VOLUME, TTS_VOICE_NAME, TTS_VOICE_TYPE

log = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self._q        = queue.Queue()
        self._speaking = threading.Event()
        self._done     = threading.Event()
        self._stop_requested = threading.Event()
        self._proc_lock = threading.Lock()
        self._current_proc: subprocess.Popen | None = None
        self._voice_type = TTS_VOICE_TYPE.strip().lower() or "default"
        self._voice_name = TTS_VOICE_NAME.strip()
        self._backends = self._select_backends()
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
        self._stop_requested.clear()
        self._done.clear()
        self._q.put(("speak", text))
        self._done.wait()

    def speak_async(self, text: str):
        """Enqueue and return immediately."""
        if text.strip():
            self._q.put(("speak", text))

    def stop(self):
        self._stop_requested.set()
        self._terminate_current_process()
        self._q.put(("stop", None))

    def shutdown(self):
        self._stop_requested.set()
        self._terminate_current_process()
        self._q.put(("quit", None))
        self._thread.join(timeout=4)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _select_backends(self) -> list[str]:
        forced = os.getenv("TTS_BACKEND", "auto").strip().lower()
        if forced and forced != "auto":
            return [forced] if forced == "pyttsx3" else [forced, "pyttsx3"]

        if sys.platform == "darwin":
            return ["say", "pyttsx3"]

        if sys.platform.startswith("linux"):
            return ["spd-say", "espeak-ng", "espeak", "pyttsx3"]

        return ["pyttsx3"]

    def _set_current_proc(self, proc: subprocess.Popen | None):
        with self._proc_lock:
            self._current_proc = proc

    def _terminate_current_process(self):
        with self._proc_lock:
            proc = self._current_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
        finally:
            self._set_current_proc(None)

    def _worker(self):
        engine = None
        log.info("TTS backend order: %s", " -> ".join(self._backends))
        if "pyttsx3" in self._backends:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", TTS_RATE)
                engine.setProperty("volume", TTS_VOLUME)
                self._configure_pyttsx3_voice(engine)
                log.info("TTS ready (pyttsx3, rate=%d)", TTS_RATE)
            except Exception as exc:
                log.warning("TTS pyttsx3 unavailable: %s", exc)
                engine = None

        while True:
            try:
                cmd, payload = self._q.get(timeout=1)
            except queue.Empty:
                continue

            if cmd == "quit":
                break

            elif cmd == "stop":
                was_speaking = self._speaking.is_set()
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                self._terminate_current_process()
                self._speaking.clear()
                self._done.set()
                if was_speaking and self.on_speaking_end:
                    try:
                        self.on_speaking_end()
                    except Exception:
                        pass

            elif cmd == "speak":
                self._speaking.set()
                try:
                    spoken = False
                    for backend in self._backends:
                        if self._stop_requested.is_set():
                            break

                        if backend == "pyttsx3":
                            if engine is None:
                                continue
                            log.info("TTS speaking (%d chars, backend=pyttsx3)", len(payload))
                            if self.on_speaking_start:
                                try:
                                    self.on_speaking_start()
                                except Exception:
                                    pass
                            self._speak_with_pyttsx3(engine, payload)
                            spoken = True
                            break

                        try:
                            command = self._native_command(backend, payload, self._native_voice_for_backend(backend))
                        except ValueError:
                            continue

                        try:
                            log.info("TTS speaking (%d chars, backend=%s)", len(payload), backend)
                            proc = subprocess.Popen(
                                command,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            self._set_current_proc(proc)
                            if self.on_speaking_start:
                                try:
                                    self.on_speaking_start()
                                except Exception:
                                    pass
                            while proc.poll() is None:
                                if self._stop_requested.is_set():
                                    self._terminate_current_process()
                                    break
                                time.sleep(0.05)
                            spoken = True
                            break
                        except FileNotFoundError:
                            continue

                    if not spoken:
                        log.warning("TTS has no working backend")
                except Exception as exc:
                    log.warning("TTS speak error: %s", exc)
                finally:
                    self._terminate_current_process()
                    self._speaking.clear()
                    self._done.set()
                    if self.on_speaking_end:
                        try:
                            self.on_speaking_end()
                        except Exception:
                            pass

    def _speak_with_pyttsx3(self, engine, text: str):
        """Run pyttsx3 in small iterations so stop() can interrupt mid-utterance."""
        engine.say(text)
        started_loop = False
        try:
            try:
                engine.startLoop(False)
                started_loop = True
            except Exception:
                # Fallback path if driver loop control is unavailable.
                engine.runAndWait()
                return

            while True:
                if self._stop_requested.is_set():
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    break

                try:
                    engine.iterate()
                except Exception:
                    break

                try:
                    if not engine.isBusy():
                        break
                except Exception:
                    # If driver cannot report busy state, keep iterating briefly.
                    pass
                time.sleep(0.01)
        finally:
            if started_loop:
                try:
                    engine.endLoop()
                except Exception:
                    pass

    @staticmethod
    def _native_command(backend: str, text: str, voice_name: str = "") -> list[str]:
        if backend == "say":
            cmd = ["say", "-r", str(TTS_RATE)]
            if voice_name:
                cmd.extend(["-v", voice_name])
            return cmd + [text]
        if backend == "spd-say":
            cmd = ["spd-say", "-r", str(TTS_RATE)]
            if voice_name:
                cmd.extend(["-l", voice_name])
            return cmd + [text]
        if backend in {"espeak", "espeak-ng"}:
            cmd = [backend, "-s", str(TTS_RATE)]
            if voice_name:
                cmd.extend(["-v", voice_name])
            return cmd + [text]
        raise ValueError(f"Unsupported native backend: {backend}")

    def _native_voice_for_backend(self, backend: str) -> str:
        if self._voice_name:
            return self._voice_name

        if backend == "say":
            mac_map = {
                "default": "",
                "female": "Samantha",
                "male": "Alex",
                "classic": "Fred",
                "robot": "Zarvox",
                "whisper": "Whisper",
            }
            return mac_map.get(self._voice_type, "")

        if backend in {"espeak", "espeak-ng"}:
            linux_map = {
                "default": "",
                "female": "en+f3",
                "male": "en+m3",
                "classic": "en",
                "robot": "en+m1",
                "whisper": "en+f2",
            }
            return linux_map.get(self._voice_type, "")

        if backend == "spd-say":
            spd_map = {
                "default": "",
                "female": "en",
                "male": "en",
                "classic": "en",
                "robot": "en",
                "whisper": "en",
            }
            return spd_map.get(self._voice_type, "")

        return ""

    def _configure_pyttsx3_voice(self, engine):
        target = self._voice_name.strip().lower()
        if not target:
            target = {
                "default": "",
                "female": "samantha",
                "male": "alex",
                "classic": "fred",
                "robot": "zira",
                "whisper": "victoria",
            }.get(self._voice_type, "")
        if not target:
            return

        try:
            voices = engine.getProperty("voices") or []
        except Exception:
            return

        for voice in voices:
            blob = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            if target in blob:
                try:
                    engine.setProperty("voice", voice.id)
                    log.info("TTS voice selected: %s", getattr(voice, "name", voice.id))
                except Exception:
                    pass
                return

