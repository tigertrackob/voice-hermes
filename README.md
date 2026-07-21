# Voice-Hermes Agent 🎙️🤖

Headless voice interface for [Hermes Agent](https://hermes-agent.nousresearch.com). Fully local, CPU-only, no GPU required.

## Features

- **Wake word triggered** — say "Hey Jarvis" (or your custom wake word) to activate
- **Speech-to-text** — Whisper.cpp (tiny/small model) transcribes your query
- **AI agent** — query is sent to Hermes Agent for tool-calling and reasoning
- **Text-to-speech** — Piper TTS reads the response aloud with selectable .onnx voice profiles
- **Interruptible** — say "stop" to cut off TTS mid-playback
- **Always-listening** — idle until wake word, auto-closes after silence
- **Systemd service** — runs headless, logs to journalctl

## Quick Start

```bash
# Install dependencies
./setup.sh

# Start in foreground (testing)
source .venv/bin/activate
python -m voice_hermes start

# Or install as systemd service
sudo cp voice-hermes.service /etc/systemd/system/
sudo systemctl enable --now voice-hermes
sudo journalctl -u voice-hermes -f
```

## Project Structure

```
voice-hermes/
├── voice_hermes/        # Python package
│   ├── audio.py         # Audio capture & playback
│   ├── wake_word.py     # OpenWakeWord listener
│   ├── stt.py           # Whisper.cpp STT
│   ├── tts.py           # Piper TTS
│   ├── agent_bridge.py  # Hermes subprocess bridge
│   ├── interrupt.py     # Keyword interruption
│   ├── orchestrator.py  # State machine
│   ├── config.py        # Config loader
│   └── cli.py           # CLI entry point
├── models/              # Downloaded model files (gitignored)
├── config.toml          # User configuration (gitignored)
├── setup.sh             # One-shot install script
└── voice-hermes.service # systemd unit
```

## License

MIT
