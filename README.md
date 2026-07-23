# Voice-Hermes Agent 🎙️🤖

Headless voice interface for [Hermes Agent](https://hermes-agent.nousresearch.com). Fully local, CPU-only, no GPU required.

## Features

- **Wake word triggered** — say "Hey Jarvis" (or your custom wake word) to activate
- **Speech-to-text** — Whisper.cpp (tiny/small model) transcribes your query
- **AI agent** — query is sent to Hermes Agent for tool-calling and reasoning
- **Text-to-speech** — Piper TTS reads the response aloud with selectable .onnx voice profiles
- **Interruptible** — say "stop" to cut off TTS mid-playback
- **Always-listening** — idle until wake word, auto-closes after silence
- **Systemd/launchd service** — runs headless

## Quick Start

### Linux
```bash
# Install dependencies and models
./setup.sh

# Start in foreground (testing)
source .venv/bin/activate
python -m voice_hermes start

# Or install as systemd service
sudo cp voice-hermes.service /etc/systemd/system/
sudo systemctl enable --now voice-hermes
sudo journalctl -u voice-hermes -f
```

### macOS (Apple Silicon & Intel)
```bash
# macOS-specific setup via Homebrew
bash scripts/setup_macos.sh

# Start in foreground
source .venv/bin/activate
python -m voice_hermes start

# Or install as launchd service
cp scripts/voice-hermes.plist ~/Library/LaunchAgents/
# Edit the plist to set your correct WorkingDirectory and python path
launchctl load ~/Library/LaunchAgents/voice-hermes.plist
tail -f /tmp/voice-hermes.log
```

> **Note:** macOS requires microphone permissions in System Settings → Privacy & Security → Microphone.

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
