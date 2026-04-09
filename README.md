# TalkingMAC

A retro Macintosh AI assistant with a pixel-art face, voice I/O, and a 1984 personality.

Say **"hey mac"** — the face wakes up, listens, thinks, and talks back.

---

## Features

- **Pixel-art animated face** — 7 expressions (idle, listening, thinking, talking, happy, error, sleep)
- **Wake word + barge-in** — say "hey mac" to trigger speech; can interrupt active replies
- **Voice input** — local Whisper STT (default) or Google Speech Recognition
- **Voice output** — pyttsx3/native backends with presets: `default`, `female`, `male`, `classic`, `robot`, `whisper`
- **Dual AI backends** — Ollama (default) or Google Gemini
- **Dual interaction modes** — `chat` (current behavior) or `tools` (LangChain-style tool loop)
- **Built-in tools** (tool mode) — `repl`, `terminal`, `web_search`
- **Configurable startup window mode** — fullscreen, windowed, borderless, floating, or fullscreen-borderless
- **MCP support** — optional LangChain MCP integration

---

## Requirements

- Python 3.10+
- macOS (tested) — should work on Linux with minor audio config changes
- A microphone
- [Ollama](https://ollama.com) running locally **or** a Gemini API key

Install dependencies:

```bash
pip install -r requirements.txt
```

For local Whisper STT (downloads a model on first run):

```bash
pip install openai-whisper
```

---

## Setup

### 1. Ollama (default backend)

```bash
ollama pull mistral:7b-instruct
ollama serve
```

### 2. Gemini (cloud backend)

Get a key at [aistudio.google.com](https://aistudio.google.com) and set it in `.env`:

```env
AI_BACKEND=gemini
GEMINI_API_KEY=your_key_here
```

### 3. Run from terminal

```bash
python3 main.py
AI_BACKEND=gemini python3 main.py
LLM_INTERACTION_MODE=tools python3 main.py
UI_WINDOW_MODE=windowed UI_WINDOW_WIDTH=1200 UI_WINDOW_HEIGHT=760 python3 main.py
```

### 4. Run from PyCharm

- Run target: `main.py`
- In **Run/Debug Configurations -> Environment variables**, set values like:
  - `AI_BACKEND=ollama;LLM_INTERACTION_MODE=chat;UI_WINDOW_MODE=fullscreen`
  - `TTS_VOICE_TYPE=classic;TTS_BACKEND=auto`

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `AI_BACKEND` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_MODEL` | `mistral:7b-instruct` | Any local Ollama model |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `LLM_INTERACTION_MODE` | `chat` | `chat` or `tools` |
| `AGENT_MAX_STEPS` | `4` | Max LangChain tool loop iterations |
| `TOOL_REPL_ENABLED` | `1` | Enable `repl` tool in tools mode |
| `TOOL_TERMINAL_ENABLED` | `1` | Enable `terminal` tool in tools mode |
| `TOOL_WEB_SEARCH_ENABLED` | `1` | Enable `web_search` tool in tools mode |
| `TOOL_TERMINAL_TIMEOUT_SECS` | `8` | Timeout for terminal tool commands |
| `TOOL_WEB_SEARCH_MAX_RESULTS` | `5` | Maximum web result lines |
| `WAKE_WORD` | `hey mac` | Phrase that triggers listening |
| `STT_ENGINE` | `whisper` | `whisper` or `google` |
| `TTS_VOICE_TYPE` | `default` | `default`, `female`, `male`, `classic`, `robot`, `whisper` |
| `TTS_VOICE_NAME` | _(empty)_ | Explicit OS voice name (overrides type mapping) |
| `TTS_BACKEND` | `auto` | `auto`, `say`, `pyttsx3`, `spd-say`, `espeak`, `espeak-ng` |
| `TTS_RATE` | `165` | Speech rate |
| `TTS_VOLUME` | `0.9` | Volume 0.0 to 1.0 |
| `UI_WINDOW_MODE` | `fullscreen` | `fullscreen`, `windowed`, `borderless`, `floating`, `fullscreen_borderless` |
| `UI_FORCE_FULLSCREEN` | `0` | Force startup in `fullscreen_borderless` |
| `UI_WINDOW_WIDTH` | `1280` | Window width (windowed/borderless/floating) |
| `UI_WINDOW_HEIGHT` | `800` | Window height (windowed/borderless/floating) |
| `IDLE_SLEEP_TIMEOUT_SECS` | `60` | Seconds before sleep face; `0` disables |

### Example `.env`

```env
AI_BACKEND=ollama
LLM_INTERACTION_MODE=tools
AGENT_MAX_STEPS=4
TOOL_REPL_ENABLED=1
TOOL_TERMINAL_ENABLED=1
TOOL_WEB_SEARCH_ENABLED=1
TTS_VOICE_TYPE=classic
UI_WINDOW_MODE=windowed
UI_WINDOW_WIDTH=1200
UI_WINDOW_HEIGHT=760
```

---

## Keyboard shortcuts

- `Enter` -> submit text input
- `Esc` -> quit
- `/sleep` or `/s` -> manual sleep face

---

## Architecture

```text
main.py  (TalkingMACAssistant orchestrator)
├── ui/
│   ├── app.py          — pygame loop with selectable startup window mode
│   ├── mac_face.py     — pixel-art Mac face
│   └── expressions.py  — expression grids and animation constants
├── ai/
│   ├── llm_manager.py  — backend + interaction mode router (chat/tools)
│   ├── tools.py        — repl / terminal / web_search tools
│   └── mcp_client.py   — optional MCP multi-server client
└── voice/
    ├── audio_capture.py
    ├── stt.py
    ├── tts.py
    └── wake_word.py
```

---

## Notes

- In `floating` mode, pygame creates a frameless smaller window, but always-on-top behavior is platform dependent.
- `fullscreen_borderless` gives a no-border full-screen look without exclusive fullscreen mode.
