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

# Create virtual environment
cd "$(dirname "$0")"
echo "[2/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[3/6] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Install tflite-runtime (no Python 3.12 wheel — use cp311, it's ABI-compatible)
pip install tflite-runtime==2.14.0 --only-binary :all: \
    --platform manylinux2014_x86_64 --python-version 3.11 \
    --no-deps -q 2>/dev/null || true

# Install openwakeword (tflite-runtime already handled above)
pip install openwakeword>=0.6.0 -q
# scikit-learn is needed by openwakeword's internal preprocessing
pip install scikit-learn -q

# Download Piper binary
echo "[4/6] Downloading Piper TTS binary..."
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
echo "[5/6] Downloading Whisper tiny.en model..."
mkdir -p models/whisper
if [ ! -f models/whisper/ggml-tiny.en.bin ]; then
    wget -q https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin \
        -O models/whisper/ggml-tiny.en.bin
fi

# Download Piper voice model
echo "[6/6] Downloading Piper voice model (en_US-lessac-medium)..."
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
