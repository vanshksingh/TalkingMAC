# TalkingMAC

A full-screen retro Macintosh AI assistant with a pixel-art face, voice I/O, and a 1984 personality.

Say **"hey mac"** — the face wakes up, listens, thinks, and talks back. All rendered in a dark full-screen pygame window with warm retro beige tones.

---

## Features

- **Pixel-art animated face** — 7 expressions (idle, listening, thinking, talking, happy, error, sleep) with smooth color blending and a gentle floating animation
- **Wake word** — say "hey mac" to trigger a voice query; interrupts any in-progress response (barge-in)
- **Voice input** — local Whisper STT (default) or Google Speech Recognition
- **Voice output** — pyttsx3 TTS with configurable voice, rate, and volume
- **Dual AI backends** — local Ollama (default) or Google Gemini via LangChain
- **Text input** — type directly in the chat box at any time
- **Auto-sleep** — face dims and sleeps after configurable idle timeout; wakes on activity
- **MCP support** — optional LangChain MCP tool integration (disabled by default)

---

## Expressions

| Expression  | Trigger                        | Face color  |
|-------------|--------------------------------|-------------|
| IDLE        | Waiting                        | Warm cream  |
| LISTENING   | Wake word detected, recording  | Pale blue   |
| THINKING    | LLM generating response        | Pale green  |
| TALKING     | TTS playing, mouth animating   | Pure white  |
| HAPPY       | Greeting / wake word ack       | Warm gold   |
| ERROR       | LLM or config failure          | Red         |
| SLEEPING    | Idle timeout reached           | Dim grey    |

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

For local Whisper STT (downloads a ~500 MB model on first run):

```bash
pip install openai-whisper
```

---

## Setup

### 1. Ollama (default backend)

```bash
# Install Ollama: https://ollama.com
ollama pull mistral:7b-instruct
ollama serve
```

### 2. Gemini (cloud backend)

Get a key at [aistudio.google.com](https://aistudio.google.com) and set it in your `.env`:

```env
AI_BACKEND=gemini
GEMINI_API_KEY=your_key_here
```

### 3. Run

```bash
python3 main.py

# Override backend on the fly:
AI_BACKEND=gemini python3 main.py
```

PyCharm: set `main.py` as the run target — no extra config needed.

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `AI_BACKEND` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_MODEL` | `mistral:7b-instruct` | Any model available in Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `GEMINI_API_KEY` | _(none)_ | Required for Gemini backend |
| `WAKE_WORD` | `hey mac` | Phrase that triggers listening |
| `WAKE_WORD_INTERRUPTION_ENABLED` | `1` | Barge-in: say wake word mid-response to interrupt |
| `STT_ENGINE` | `whisper` | `whisper` or `google` |
| `STT_WHISPER_MODEL` | `small.en` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `TTS_RATE` | `165` | Speech rate (words per minute) |
| `TTS_VOLUME` | `0.9` | TTS volume (0.0 – 1.0) |
| `TTS_VOICE_NAME` | _(system default)_ | pyttsx3 voice name |
| `IDLE_SLEEP_TIMEOUT_SECS` | `60` | Seconds of inactivity before sleep face; `0` to disable |

Example `.env`:

```env
AI_BACKEND=ollama
OLLAMA_MODEL=llama3:8b
WAKE_WORD=hey mac
STT_WHISPER_MODEL=base.en
TTS_RATE=155
IDLE_SLEEP_TIMEOUT_SECS=120
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit text input |
| `Esc` | Quit |
| `/sleep` or `/s` | Manually trigger sleep face |

---

## Architecture

```
main.py  (TalkingMACAssistant — orchestrator, state machine)
├── ui/
│   ├── app.py          — pygame full-screen loop, chat history, text input
│   ├── mac_face.py     — pixel-art Mac face with floating animation
│   └── expressions.py  — Expression enum, eye/mouth pixel grids, animation constants
├── ai/
│   ├── llm_manager.py  — Ollama or Gemini via LangChain, streaming
│   └── mcp_client.py   — optional LangChain MCP multi-server client
└── voice/
    ├── audio_capture.py — shared sounddevice mic capture
    ├── stt.py           — Whisper or Google STT
    ├── tts.py           — pyttsx3 in dedicated thread, speaking callbacks
    └── wake_word.py     — background STT pass checking for wake phrase
```

**State machine:**

```
IDLE → (wake word / text input) → LISTENING → THINKING → TALKING → IDLE
                                                    ↑
                                         barge-in: wake word mid-response
                                         interrupts TTS and restarts
```

---

## Troubleshooting

**No audio input detected**
- Check microphone permissions (macOS: System Settings > Privacy > Microphone)
- `sounddevice` is used instead of `pyaudio` — no PortAudio headers needed

**Wake word not triggering**
- Speak clearly: "hey mac" (default phrase)
- Try a quieter environment or a larger Whisper model (`STT_WHISPER_MODEL=medium.en`)

**Ollama connection refused**
- Make sure Ollama is running: `ollama serve`
- Confirm the model is pulled: `ollama list`

**LangChain version conflicts**
- `langchain-mcp` 0.2.x requires `langchain-core ~0.3.x` — do not upgrade to 1.x
- Pin is already set in `requirements.txt`

---

## Character

Mac is a friendly retro Macintosh assistant from 1984. Concise, helpful, occasionally nostalgic. The system prompt can be customized in `config.py`:

```python
SYSTEM_PROMPT = (
    "You are Mac, a friendly retro Macintosh assistant from 1984. "
    "You speak with nostalgic charm and wit. Keep responses concise and helpful. "
    "Occasionally reference the classic Mac era. Never break character."
)
```
