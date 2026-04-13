"""
TalkingMAC — Central configuration
All user-tunable settings live here.

How to run with params:
- Terminal (one-off): AI_BACKEND=gemini LLM_INTERACTION_MODE=tools python3 main.py
- Terminal (.env): put values in .env, then run python3 main.py
- PyCharm: Run/Debug Configurations -> Environment variables, e.g.
  AI_BACKEND=ollama;LLM_INTERACTION_MODE=tools;UI_WINDOW_MODE=fullscreen
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


# ── AI Backend ──────────────────────────────────────────────────────────────
# AI_BACKEND: "ollama" | "gemini"
AI_BACKEND: str = os.getenv("AI_BACKEND", "ollama")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# LLM mode toggle:
# - "chat"  = current conversational mode (default)
# - "tools" = LangChain-style tool-calling loop (bounded by AGENT_MAX_STEPS)
LLM_INTERACTION_MODE: str = os.getenv("LLM_INTERACTION_MODE", "tools").strip().lower()
AGENT_MAX_STEPS: int = max(1, int(os.getenv("AGENT_MAX_STEPS", "4")))

SYSTEM_PROMPT: str = (
    "You are Mac, a friendly retro Macintosh assistant from 1984. "
    "You speak with nostalgic charm and wit. Keep responses concise and helpful. "
    "Occasionally reference the classic Mac era. Never break character."
)

# ── Voice ────────────────────────────────────────────────────────────────────
WAKE_WORD: str = os.getenv("WAKE_WORD", "hey mac")
WAKE_WORD_INTERRUPTION_ENABLED: bool = _env_bool("WAKE_WORD_INTERRUPTION_ENABLED", "1")
STT_ENGINE: str = os.getenv("STT_ENGINE", "whisper")
STT_WHISPER_MODEL: str = os.getenv("STT_WHISPER_MODEL", "small.en")
STT_WHISPER_LANGUAGE: str = os.getenv("STT_WHISPER_LANGUAGE", "en")
STT_WHISPER_INITIAL_PROMPT: str = os.getenv(
    "STT_WHISPER_INITIAL_PROMPT",
    "Short spoken commands for TalkingMAC, including names, places, and everyday objects.",
)
STT_TRAILING_SILENCE_SECS: float = float(os.getenv("STT_TRAILING_SILENCE_SECS", "1.0"))
STT_WHISPER_BEAM_SIZE: int = int(os.getenv("STT_WHISPER_BEAM_SIZE", "6"))
STT_WHISPER_BEST_OF: int = int(os.getenv("STT_WHISPER_BEST_OF", "6"))
STT_WHISPER_DUAL_PASS: bool = _env_bool("STT_WHISPER_DUAL_PASS", "1")

# Wake-word sensitivity tuning (affects barge-in behavior too):
# Higher WAKE_ENERGY_THRESH and higher WAKE_MIN_TRIGGER_INTERVAL_SECS => fewer false triggers.
# Higher *_TOKEN_MIN values => stricter phrase matching.
WAKE_CHECK_STEP_SECS: float = max(0.1, float(os.getenv("WAKE_CHECK_STEP_SECS", "0.4")))
WAKE_ENERGY_THRESH: float = max(0.0, float(os.getenv("WAKE_ENERGY_THRESH", "0.0015")))
WAKE_MIN_TRIGGER_INTERVAL_SECS: float = max(0.0, float(os.getenv("WAKE_MIN_TRIGGER_INTERVAL_SECS", "1.2")))

WAKE_IDLE_FIRST_TOKEN_MIN: float = min(0.99, max(0.5, float(os.getenv("WAKE_IDLE_FIRST_TOKEN_MIN", "0.74"))))
WAKE_IDLE_SECOND_TOKEN_MIN: float = min(0.99, max(0.5, float(os.getenv("WAKE_IDLE_SECOND_TOKEN_MIN", "0.63"))))
WAKE_TTS_FIRST_TOKEN_MIN: float = min(0.99, max(0.5, float(os.getenv("WAKE_TTS_FIRST_TOKEN_MIN", "0.86"))))
WAKE_TTS_SECOND_TOKEN_MIN: float = min(0.99, max(0.5, float(os.getenv("WAKE_TTS_SECOND_TOKEN_MIN", "0.82"))))

WAKE_IDLE_MAX_GAP_TOKENS: int = max(1, int(os.getenv("WAKE_IDLE_MAX_GAP_TOKENS", "3")))
WAKE_TTS_MAX_GAP_TOKENS: int = max(1, int(os.getenv("WAKE_TTS_MAX_GAP_TOKENS", "1")))
# While TTS is speaking, wake phrase must start within this many leading tokens.
WAKE_TTS_PREFIX_TOKEN_LIMIT: int = max(0, int(os.getenv("WAKE_TTS_PREFIX_TOKEN_LIMIT", "1")))

# Voice output style presets available in TTSEngine:
# "default" | "female" | "male" | "classic" | "robot" | "whisper"
TTS_VOICE_TYPE: str = os.getenv("TTS_VOICE_TYPE", "default")
# Optional explicit voice name from the OS voice list (overrides TTS_VOICE_TYPE mapping)
TTS_VOICE_NAME: str = os.getenv("TTS_VOICE_NAME", "")
# Optional backend override: "auto" | "say" | "pyttsx3" | "spd-say" | "espeak" | "espeak-ng"
TTS_BACKEND: str = os.getenv("TTS_BACKEND", "auto")
TTS_RATE: int = int(os.getenv("TTS_RATE", "165"))
TTS_VOLUME: float = float(os.getenv("TTS_VOLUME", "0.9"))
AMBIENT_DURATION: float = 1.0
LISTEN_TIMEOUT: int = 8
IDLE_SLEEP_TIMEOUT_SECS: float = float(os.getenv("IDLE_SLEEP_TIMEOUT_SECS", "60"))

# ── Tool-calling mode ────────────────────────────────────────────────────────
# Enable/disable tools when LLM_INTERACTION_MODE=tools
TOOL_REPL_ENABLED: bool = _env_bool("TOOL_REPL_ENABLED", "1")
TOOL_TERMINAL_ENABLED: bool = _env_bool("TOOL_TERMINAL_ENABLED", "1")
TOOL_WEB_SEARCH_ENABLED: bool = _env_bool("TOOL_WEB_SEARCH_ENABLED", "1")
TOOL_TERMINAL_TIMEOUT_SECS: float = max(1.0, float(os.getenv("TOOL_TERMINAL_TIMEOUT_SECS", "8")))
TOOL_WEB_SEARCH_MAX_RESULTS: int = max(1, int(os.getenv("TOOL_WEB_SEARCH_MAX_RESULTS", "5")))

# ── UI ───────────────────────────────────────────────────────────────────────
WINDOW_TITLE: str = "TalkingMAC"
TARGET_FPS: int = 60

# Startup window mode:
# - "fullscreen"            -> native exclusive fullscreen
# - "windowed"              -> normal resizable window
# - "borderless"            -> frameless window (uses UI_WINDOW_WIDTH/HEIGHT)
# - "floating"              -> small frameless window preset (best-effort, platform dependent)
# - "fullscreen_borderless" -> no-border fullscreen (desktop-sized borderless)
UI_WINDOW_MODE: str = os.getenv("UI_WINDOW_MODE", "fullscreen").strip().lower()
# If true, overrides UI_WINDOW_MODE and starts in fullscreen_borderless
UI_FORCE_FULLSCREEN: bool = _env_bool("UI_FORCE_FULLSCREEN", "0")
UI_WINDOW_WIDTH: int = max(320, int(os.getenv("UI_WINDOW_WIDTH", "1280")))
UI_WINDOW_HEIGHT: int = max(240, int(os.getenv("UI_WINDOW_HEIGHT", "800")))

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
