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
import queue
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf

from config import WAKE_WORD
from voice.audio_capture import AudioCapture, SAMPLE_RATE

log = logging.getLogger(__name__)

WINDOW_SECS   = 2.0                          # rolling window size (lower latency)
STEP_SECS     = 0.4                          # how often we run STT
WINDOW_SAMPLES = int(WINDOW_SECS * SAMPLE_RATE)
ENERGY_THRESH  = 0.0015                      # skip STT on near-silence
WAKE_MIN_TRIGGER_INTERVAL_SECS = 1.2         # debounce duplicate/echo triggers


class WakeWordDetector:
    _whisper_model = None   # lazy-loaded, shared across instances

    def __init__(self, capture: AudioCapture, on_wake: Callable[[], None]):
        self._capture  = capture
        self._on_wake  = on_wake
        self._wake_kw  = WAKE_WORD.lower().strip()
        self._running  = False
        self._paused   = False
        self._reset_pending = False
        self._thread: Optional[threading.Thread] = None
        self._last_wake_ts = 0.0
        self._tts_active = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._reset_pending = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="WakeWord")
        self._thread.start()
        log.info("Wake-word detector started — phrase: %r", self._wake_kw)

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True
        self._reset_pending = True

    def resume(self):
        self._paused = False
        self._reset_pending = True

    def set_tts_active(self, active: bool):
        """Called by orchestrator to tighten matching while assistant audio is playing."""
        self._tts_active = bool(active)

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
                if self._reset_pending:
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    buf.clear()
                    buf_len = 0
                    next_check = time.time() + STEP_SECS
                    self._reset_pending = False

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
                    time.sleep(0.02)
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
                text = self._transcribe(rec, audio)  # pyright: ignore[reportGeneralTypeIssues]
                if text:
                    log.info("Wake transcript: %r", text)

                if text and self._matches_wake_phrase(text):
                    now = time.time()
                    if now - self._last_wake_ts < WAKE_MIN_TRIGGER_INTERVAL_SECS:
                        continue
                    self._last_wake_ts = now
                    log.info("Wake word detected: %r", text)
                    self._on_wake()

        finally:
            self._capture.unsubscribe(q)

    @classmethod
    def _transcribe(cls, rec: Any, audio_np: np.ndarray) -> str:
        if cls._whisper_model is not None:
            try:
                result = cls._whisper_model.transcribe(
                    audio_np,
                    language="en",
                    fp16=False,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt="hey mac",
                )
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
                audio_data = getattr(rec, "record")(source)
                return getattr(rec, "recognize_google")(audio_data)
        except Exception as e:
            log.warning("Wake: Google STT error: %s", e)
            return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()

    def _matches_wake_phrase(self, text: str) -> bool:
        normalized_text = self._normalize_text(text)
        normalized_wake = self._normalize_text(self._wake_kw)

        if not normalized_text or not normalized_wake:
            return False

        text_tokens = normalized_text.split()
        wake_tokens = normalized_wake.split()

        if normalized_wake in normalized_text:
            # During TTS, require phrase near beginning or as a short utterance to reduce speaker echo triggers.
            return self._wake_phrase_position_ok(text_tokens, wake_tokens, strict=self._tts_active)

        if len(wake_tokens) == 2:
            first, second = wake_tokens
            first_min = 0.86 if self._tts_active else 0.74
            second_min = 0.82 if self._tts_active else 0.63
            max_gap = 1 if self._tts_active else 3
            for i, token in enumerate(text_tokens):
                if not self._token_close(token, first, min_ratio=first_min):
                    continue
                if self._tts_active and i > 1:
                    continue
                window = text_tokens[i + 1 : i + 1 + max_gap]
                if any(
                    token2 == second or SequenceMatcher(None, token2, second).ratio() >= second_min
                    for token2 in window
                ):
                    return True

            return False

        return SequenceMatcher(None, normalized_text, normalized_wake).ratio() >= 0.84

    @staticmethod
    def _token_close(a: str, b: str, min_ratio: float) -> bool:
        return a == b or SequenceMatcher(None, a, b).ratio() >= min_ratio

    @staticmethod
    def _wake_phrase_position_ok(text_tokens: list[str], wake_tokens: list[str], strict: bool) -> bool:
        if not text_tokens or not wake_tokens:
            return False
        n = len(wake_tokens)
        for i in range(0, max(0, len(text_tokens) - n + 1)):
            if text_tokens[i:i + n] != wake_tokens:
                continue
            if not strict:
                return True
            # Strict mode (TTS active): only accept short utterances or phrase near the beginning.
            return len(text_tokens) <= n + 2 or i <= 1
        return False

