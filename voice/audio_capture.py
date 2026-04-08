"""
voice/audio_capture.py — Single shared microphone InputStream.

Both the wake-word detector and the STT engine subscribe to this.
One InputStream → callback → per-subscriber queues.
This prevents the conflict that occurs when two parts of the code
each try to open their own sounddevice stream against the same device.
"""

import logging
import queue
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS    = 1
BLOCK_SIZE  = 1024   # frames per callback


class AudioCapture:
    """
    Singleton mic capture.  Call start() once at boot.
    Components call subscribe() to get a Queue[np.ndarray] of audio chunks.
    """

    def __init__(self):
        self._stream: sd.InputStream | None = None
        self._lock      = threading.Lock()
        self._listeners: list[queue.Queue] = []
        self._running   = False

    def start(self):
        if self._running:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=self._callback,
                latency="low",
            )
            self._stream.start()
            self._running = True
            log.info("AudioCapture started (sr=%d, block=%d)", SAMPLE_RATE, BLOCK_SIZE)
        except Exception as exc:
            log.error(
                "AudioCapture failed: %s\n"
                "  → Check System Settings → Privacy & Security → Microphone\n"
                "    and grant access to Terminal / Python / PyCharm.",
                exc,
            )
            raise   # caller decides whether to fall back to text-only

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._running = False

    def subscribe(self, maxsize: int = 300) -> queue.Queue:
        """Return a Queue that will receive np.ndarray chunks."""
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._listeners.remove(q)
            except ValueError:
                pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            log.debug("AudioCapture status: %s", status)
        chunk = indata[:, 0].copy()   # flatten to 1-D
        with self._lock:
            for q in self._listeners:
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    # Drop oldest chunk to make room
                    try:
                        q.get_nowait()
                        q.put_nowait(chunk)
                    except Exception:
                        pass
