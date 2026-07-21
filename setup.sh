#!/usr/bin/env bash
set -euo pipefail

echo "=== Voice-Hermes — System Setup ==="

# System dependencies
echo "[1/5] Installing system packages..."
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
    cmake

# Create virtual environment
cd "$(dirname "$0")"
echo "[2/5] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[3/5] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Download Whisper.cpp model
echo "[4/5] Downloading Whisper tiny.en model..."
mkdir -p models/whisper
if [ ! -f models/whisper/ggml-tiny.en.bin ]; then
    wget -q https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin \
        -O models/whisper/ggml-tiny.en.bin
fi

# Download Piper voice model
echo "[5/5] Downloading Piper voice model (en_US-lessac-medium)..."
mkdir -p models/piper
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
