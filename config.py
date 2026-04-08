"""
TalkingMAC — Central configuration
All user-tunable settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── AI Backend ──────────────────────────────────────────────────────────────
AI_BACKEND: str = os.getenv("AI_BACKEND", "ollama")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT: str = (
    "You are Mac, a friendly retro Macintosh assistant from 1984. "
    "You speak with nostalgic charm and wit. Keep responses concise and helpful. "
    "Occasionally reference the classic Mac era. Never break character."
)

# ── Voice ────────────────────────────────────────────────────────────────────
WAKE_WORD: str      = os.getenv("WAKE_WORD", "hey mac")
WAKE_WORD_INTERRUPTION_ENABLED: bool = os.getenv("WAKE_WORD_INTERRUPTION_ENABLED", "1").lower() not in {"0", "false", "no"}
STT_ENGINE: str     = os.getenv("STT_ENGINE", "whisper")
STT_WHISPER_MODEL: str = os.getenv("STT_WHISPER_MODEL", "small.en")
STT_WHISPER_LANGUAGE: str = os.getenv("STT_WHISPER_LANGUAGE", "en")
STT_WHISPER_INITIAL_PROMPT: str = os.getenv(
    "STT_WHISPER_INITIAL_PROMPT",
    "Short spoken commands for TalkingMAC, including names, places, and everyday objects.",
)
STT_TRAILING_SILENCE_SECS: float = float(os.getenv("STT_TRAILING_SILENCE_SECS", "1.0"))
STT_WHISPER_BEAM_SIZE: int = int(os.getenv("STT_WHISPER_BEAM_SIZE", "6"))
STT_WHISPER_BEST_OF: int = int(os.getenv("STT_WHISPER_BEST_OF", "6"))
STT_WHISPER_DUAL_PASS: bool = os.getenv("STT_WHISPER_DUAL_PASS", "1").lower() not in {"0", "false", "no"}
TTS_VOICE_TYPE: str = os.getenv("TTS_VOICE_TYPE", "default")
TTS_VOICE_NAME: str = os.getenv("TTS_VOICE_NAME", "")
TTS_RATE: int       = int(os.getenv("TTS_RATE", "165"))
TTS_VOLUME: float   = float(os.getenv("TTS_VOLUME", "0.9"))
AMBIENT_DURATION: float = 1.0
LISTEN_TIMEOUT: int = 8

# ── UI ───────────────────────────────────────────────────────────────────────
WINDOW_TITLE: str = "TalkingMAC"
TARGET_FPS: int   = 60

# ── Colour palette — retro Mac pixel art ─────────────────────────────────────
# Background: near-black (old Mac screen)
COLOR_BG            = (16,  12,  10)

# Face element colours per expression — warm monochrome retro Mac tones
COLOR_FACE_IDLE     = (236, 220, 188)   # warm cream (classic Mac beige)
COLOR_FACE_LISTEN   = (200, 230, 255)   # pale cool blue — attentive
COLOR_FACE_THINK    = (180, 220, 180)   # pale green — processing
COLOR_FACE_TALK     = (255, 255, 255)   # pure white — active/bright
COLOR_FACE_HAPPY    = (255, 240, 160)   # warm golden white
COLOR_FACE_ERROR    = (255,  80,  60)   # red
COLOR_FACE_SLEEP    = ( 80,  70,  60)   # dim warm grey

# Pupil / inner square colour (dark = same as bg for crisp look)
COLOR_PUPIL         = (16,  12,  10)

# No glow — square pixels only
COLOR_GLOW_INTENSITY: float = 0.0

# ── Face animation ────────────────────────────────────────────────────────────
FLOAT_AMPLITUDE: int   = 4      # subtle float (reduced from 10)
FLOAT_SPEED: float     = 0.9    # radians/sec

# ── MCP ──────────────────────────────────────────────────────────────────────
MCP_SERVER_CONFIG: dict | None = None
