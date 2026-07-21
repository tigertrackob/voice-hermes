#!/usr/bin/env bash
set -euo pipefail

echo "=== Voice-Hermes — System Setup ==="

# System dependencies
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    portaudio19-dev \
    python3-pyaudio \
    libsndfile1-dev \
    libopenblas-dev \
    libsox-fmt-all \
    sox \
    ffmpeg \
    build-essential \
    cmake \
    espeak-ng

# Create virtual environment with Python 3.11 (required for package compatibility)
cd "$(dirname "$0")"
echo "[2/6] Creating Python 3.11 virtual environment..."
if ! command -v python3.11 &> /dev/null; then
    echo "ERROR: python3.11 not found. Install it first:"
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "  sudo apt-get install python3.11 python3.11-dev python3.11-venv"
    exit 1
fi
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[3/6] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

pip install openwakeword>=0.6.0 -q
# scikit-learn is needed by openwakeword's internal preprocessing
pip install scikit-learn -q

# Download OpenWakeWord model files (not bundled in the pip package)
echo "[4/7] Downloading OpenWakeWord pre-trained models..."
python3 << 'PYEOF'
import os, urllib.request, pathlib, openwakeword

models_dir = pathlib.Path(openwakeword.__file__).parent / "resources" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Models that need to be downloaded (from openwakeword/__init__.py)
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

# Download Piper binary
echo "[5/7] Downloading Piper TTS binary..."
mkdir -p models/piper
if [ ! -f models/piper/piper ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
    wget -q "$PIPER_URL" -O /tmp/piper.tar.gz
    tar xzf /tmp/piper.tar.gz -C models/piper/ --strip-components=1
    rm /tmp/piper.tar.gz
    chmod +x models/piper/piper
    echo "   Piper binary installed at models/piper/piper"
fi

# Download Whisper.cpp model
echo "[6/7] Downloading Whisper tiny.en model..."
mkdir -p models/whisper
if [ ! -f models/whisper/ggml-tiny.en.bin ]; then
    wget -q https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin \
        -O models/whisper/ggml-tiny.en.bin
fi

# Download Piper voice model
echo "[7/7] Downloading Piper voice model (en_US-lessac-medium)..."
if [ ! -f models/piper/en_US-lessac-medium.onnx ]; then
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
        -O models/piper/en_US-lessac-medium.onnx
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
        -O models/piper/en_US-lessac-medium.onnx.json
fi

echo ""
echo "=== Setup complete! ==="
echo "Activate: source .venv/bin/activate"
echo "Run:     python -m voice_hermes start"
