#!/usr/bin/env bash
# Voice-Hermes macOS Setup Script
# Requires: macOS 13+ (Ventura) with Homebrew
set -euo pipefail

echo "=== Voice-Hermes — macOS Setup ==="
echo ""

# Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "ERROR: Homebrew is required. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

# Check for Python 3.11
PYTHON=""
for py in python3.11 python3; do
    if command -v "$py" &>/dev/null && "$py" --version 2>&1 | grep -q "3.11"; then
        PYTHON="$py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.11 is required. Installing via Homebrew..."
    brew install python@3.11
    PYTHON="python3.11"
fi

echo "Using: $($PYTHON --version 2>&1)"

# Detect architecture for Piper binary
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    PIPER_ARCH="aarch64"
elif [ "$ARCH" = "x86_64" ]; then
    PIPER_ARCH="x64"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Step 1: Install system dependencies
echo ""
echo "[1/7] Installing system packages via Homebrew..."
brew install portaudio libsndfile sox ffmpeg cmake espeak-ng pulseaudio 2>/dev/null || true

# Step 2: Create virtual environment
echo ""
echo "[2/7] Creating Python 3.11 virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate

# Step 3: Install Python deps
echo ""
echo "[3/7] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install openwakeword>=0.6.0 -q
pip install scikit-learn -q

# Step 4: Download OpenWakeWord models
echo ""
echo "[4/7] Downloading OpenWakeWord pre-trained models..."
$PYTHON << 'PYEOF'
import os, urllib.request, pathlib
# Find openwakeword package path
import importlib.util
spec = importlib.util.find_spec("openwakeword")
if spec is None:
    print("ERROR: openwakeword not installed")
    exit(1)

models_dir = pathlib.Path(spec.origin).parent / "resources" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

downloads = {
    "embedding_model.tflite":    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
    "melspectrogram.tflite":     "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
    "silero_vad.onnx":           "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx",
    "alexa_v0.1.tflite":         "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.tflite",
    "hey_mycroft_v0.1.tflite":   "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.tflite",
    "hey_jarvis_v0.1.tflite":    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite",
    "hey_rhasspy_v0.1.tflite":   "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_rhasspy_v0.1.tflite",
    "timer_v0.1.tflite":         "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/timer_v0.1.tflite",
    "weather_v0.1.tflite":       "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/weather_v0.1.tflite",
}

for fname, url in downloads.items():
    dest = models_dir / fname
    if not dest.exists():
        print(f"   Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"   {fname} already exists")

print("   OpenWakeWord models ready")
PYEOF

# Step 5: Download Piper binary
echo ""
echo "[5/7] Downloading Piper TTS binary (macOS ${PIPER_ARCH})..."
mkdir -p models/piper
if [ ! -f models/piper/piper ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_${PIPER_ARCH}.tar.gz"
    curl -sL "$PIPER_URL" -o /tmp/piper-macos.tar.gz
    tar xzf /tmp/piper-macos.tar.gz -C models/piper/ --strip-components=1
    rm /tmp/piper-macos.tar.gz
    chmod +x models/piper/piper
    echo "   Piper binary installed at models/piper/piper"
fi

# Step 6: Download Whisper model
echo ""
echo "[6/7] Downloading Whisper tiny.en model..."
mkdir -p models/whisper
if [ ! -f models/whisper/ggml-tiny.en.bin ]; then
    curl -sL "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" \
        -o models/whisper/ggml-tiny.en.bin
fi

# Step 7: Download Piper voice model
echo ""
echo "[7/7] Downloading Piper voice model (en_US-lessac-medium)..."
if [ ! -f models/piper/en_US-lessac-medium.onnx ]; then
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        -o models/piper/en_US-lessac-medium.onnx
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        -o models/piper/en_US-lessac-medium.onnx.json
fi

echo ""
echo "=========================================="
echo "  macOS setup complete!"
echo "=========================================="
echo ""
echo "  Activate: source .venv/bin/activate"
echo "  Run:      python -m voice_hermes start"
echo ""
echo "  NOTE: On macOS, microphone permissions must be"
echo "  granted to Terminal/IDE in System Settings →"
echo "  Privacy & Security → Microphone."
echo "=========================================="
