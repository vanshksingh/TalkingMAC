"""
voice/stt.py — Speech-to-Text engine.

Reads from a shared AudioCapture queue (no stream conflicts).
Records until trailing silence detected, then transcribes via
Google Web Speech API or local Whisper.
"""

import io
import logging
from typing import Optional

import numpy as np
import soundfile as sf

from config import (
    LISTEN_TIMEOUT,
    STT_ENGINE,
    STT_WHISPER_INITIAL_PROMPT,
    STT_WHISPER_LANGUAGE,
    STT_WHISPER_MODEL,
)
from voice.audio_capture import AudioCapture, SAMPLE_RATE

log = logging.getLogger(__name__)

SILENCE_LEVEL = 0.008   # RMS below this = silence
SILENCE_SECS  = 1.6     # seconds of trailing silence before stopping
MAX_SECS      = LISTEN_TIMEOUT + 2


class STTEngine:
    def __init__(self, capture: AudioCapture):
        self._capture = capture
        self._engine  = STT_ENGINE.lower()
        self._whisper = None
        self._whisper_model_name = STT_WHISPER_MODEL
        self._whisper_language = STT_WHISPER_LANGUAGE.strip() or None
        self._whisper_prompt = STT_WHISPER_INITIAL_PROMPT.strip()
        if self._engine == "whisper":
            self._load_whisper()
        log.info("STT engine: %s", self._engine)

    # ── Public ────────────────────────────────────────────────────────────────

    def listen(self, timeout: float = LISTEN_TIMEOUT) -> str:
        """
        Subscribe to audio, record until silence/timeout, return transcript.
        Returns "" if nothing heard or transcription failed.
        """
        log.info("STT: listening…")
        audio = self._record(timeout)
        if audio is None or len(audio) < int(SAMPLE_RATE * 0.4):
            log.info("STT: no speech detected")
            return ""
        return self._transcribe(audio)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record(self, timeout: float) -> Optional[np.ndarray]:
        import time

        q = self._capture.subscribe()
        frames: list[np.ndarray] = []
        silence_samples = 0
        silence_thresh  = int(SILENCE_SECS * SAMPLE_RATE)
        max_samples     = int(MAX_SECS * SAMPLE_RATE)
        total_samples   = 0
        speech_started  = False
        noise_floor     = 0.0
        noise_samples   = 0
        speech_thresh   = SILENCE_LEVEL
        trailing_thresh = SILENCE_LEVEL * 0.75
        deadline        = time.time() + timeout

        try:
            while time.time() < deadline and total_samples < max_samples:
                try:
                    chunk = q.get(timeout=0.5)
                except Exception:
                    continue

                frames.append(chunk)
                total_samples += len(chunk)
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if not speech_started:
                    noise_samples += 1
                    noise_floor += (rms - noise_floor) / noise_samples
                    speech_thresh = max(SILENCE_LEVEL, noise_floor * 2.5 + 0.002)
                    trailing_thresh = max(SILENCE_LEVEL * 0.75, noise_floor * 1.5 + 0.001)

                if rms > speech_thresh:
                    speech_started = True
                    silence_samples = 0
                elif speech_started:
                    if rms < trailing_thresh:
                        silence_samples += len(chunk)
                        if silence_samples >= silence_thresh:
                            break   # trailing silence — done
                    else:
                        silence_samples = 0

        finally:
            self._capture.unsubscribe(q)

        if not frames or not speech_started:
            return None

        return np.concatenate(frames)

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._engine == "whisper":
            return self._transcribe_whisper(audio)
        return self._transcribe_google(audio)

    def _transcribe_google(self, audio: np.ndarray) -> str:
        try:
            import speech_recognition as sr

            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            sf.write(buf, audio_int16, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            buf.seek(0)

            rec = sr.Recognizer()
            with sr.AudioFile(buf) as source:
                audio_data = rec.record(source)

            text = rec.recognize_google(audio_data)  # type: ignore[attr-defined]
            log.info("STT result: %r", text)
            return text

        except Exception as exc:
            log.warning("Google STT error: %s", exc)
            return ""

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        try:
            audio_f32 = np.asarray(audio, dtype=np.float32)
            audio_f32 = np.clip(audio_f32, -1.0, 1.0)
            result = self._whisper.transcribe(
                audio_f32,
                fp16=False,
                task="transcribe",
                language=self._whisper_language,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=self._whisper_prompt,
                beam_size=5,
                best_of=5,
                no_speech_threshold=0.35,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                verbose=False,
            )
            text   = result.get("text", "").strip()
            log.info("Whisper result: %r", text)
            return text
        except Exception as exc:
            log.warning("Whisper STT error: %s", exc)
            return ""

    def _load_whisper(self):
        try:
            import ssl, whisper
            log.info("Loading Whisper %r model…", self._whisper_model_name)
            # macOS Python often has SSL cert issues when downloading model weights;
            # disable verification only for this one-time download — cached after that.
            _orig = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                self._whisper = whisper.load_model(self._whisper_model_name)
            finally:
                ssl._create_default_https_context = _orig
        except Exception as exc:
            log.warning("Whisper unavailable (%s), using google", exc)
            self._engine = "google"
