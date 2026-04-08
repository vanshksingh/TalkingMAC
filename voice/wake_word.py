"""
voice/wake_word.py — Wake-word detector with rolling window.

Maintains a persistent subscription to AudioCapture and keeps a
rolling deque of the last WINDOW_SECS seconds of audio.
Every STEP_SECS seconds it sends the window to Google STT and checks
for the wake phrase.  75% overlap means the phrase is never split
across two non-overlapping chunks.
"""

import collections
import io
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from config import WAKE_WORD
from voice.audio_capture import AudioCapture, SAMPLE_RATE

log = logging.getLogger(__name__)

WINDOW_SECS   = 3.0                          # rolling window size
STEP_SECS     = 1.0                          # how often we run STT
WINDOW_SAMPLES = int(WINDOW_SECS * SAMPLE_RATE)
ENERGY_THRESH  = 0.003                       # skip STT on near-silence


class WakeWordDetector:
    _whisper_model = None   # lazy-loaded, shared across instances

    def __init__(self, capture: AudioCapture, on_wake: Callable[[], None]):
        self._capture  = capture
        self._on_wake  = on_wake
        self._wake_kw  = WAKE_WORD.lower().strip()
        self._running  = False
        self._paused   = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="WakeWord")
        self._thread.start()
        log.info("Wake-word detector started — phrase: %r", self._wake_kw)

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        import speech_recognition as sr
        rec = sr.Recognizer()

        # Lazy-load whisper.tiny once for offline, reliable transcription.
        # SSL bypass is needed on macOS for the one-time model weight download.
        if WakeWordDetector._whisper_model is None:
            try:
                import ssl, whisper
                _orig = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context
                try:
                    WakeWordDetector._whisper_model = whisper.load_model("tiny")
                finally:
                    ssl._create_default_https_context = _orig
                log.info("Wake: loaded whisper.tiny model")
            except Exception as e:
                log.warning("Wake: whisper unavailable (%s), falling back to Google STT", e)

        # Subscribe once — keep the queue alive for the detector's lifetime
        q = self._capture.subscribe(maxsize=600)
        # Rolling buffer: deque of chunks, total length ≈ WINDOW_SAMPLES
        buf: collections.deque = collections.deque()
        buf_len = 0   # total samples currently in buf

        next_check = time.time() + STEP_SECS

        try:
            while self._running:
                # Drain available audio into rolling buffer
                while True:
                    try:
                        chunk = q.get_nowait()
                        buf.append(chunk)
                        buf_len += len(chunk)
                        # Trim to window size
                        while buf_len > WINDOW_SAMPLES:
                            dropped = buf.popleft()
                            buf_len -= len(dropped)
                    except Exception:
                        break   # queue empty

                if self._paused or time.time() < next_check:
                    time.sleep(0.05)
                    continue

                next_check = time.time() + STEP_SECS

                if buf_len < SAMPLE_RATE * 0.5:
                    continue   # not enough audio yet

                # Check energy before paying for an STT round-trip
                audio = np.concatenate(list(buf))[-WINDOW_SAMPLES:]
                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms < ENERGY_THRESH:
                    continue

                log.debug("Wake: checking window, rms=%.4f", rms)
                text = self._transcribe(rec, audio)
                if text and self._wake_kw in text.lower():
                    log.info("Wake word detected: %r", text)
                    self._on_wake()
                    # Clear buffer so we don't re-trigger immediately
                    buf.clear()
                    buf_len = 0
                    next_check = time.time() + STEP_SECS

        finally:
            self._capture.unsubscribe(q)

    @classmethod
    def _transcribe(cls, rec, audio_np: np.ndarray) -> str:
        if cls._whisper_model is not None:
            try:
                result = cls._whisper_model.transcribe(audio_np, language="en", fp16=False)
                return result["text"]
            except Exception as e:
                log.warning("Wake: whisper transcribe error: %s", e)
                return ""

        # Fallback: Google STT with visible error logging
        try:
            import speech_recognition as sr
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            sf.write(buf, audio_int16, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            buf.seek(0)
            with sr.AudioFile(buf) as source:
                audio_data = rec.record(source)
            return rec.recognize_google(audio_data)
        except Exception as e:
            log.warning("Wake: Google STT error: %s", e)
            return ""
