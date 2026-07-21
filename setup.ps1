<#
.SYNOPSIS
    Voice-Hermes — Windows Setup Script
.DESCRIPTION
    Sets up the Voice-Hermes environment on Windows:
    - Creates Python virtual environment
    - Installs Python dependencies
    - Downloads OpenWakeWord pre-trained models
    - Downloads Piper TTS Windows binary + voice model
    - Downloads Whisper.cpp model
.NOTES
    Run from PowerShell as: .\setup.ps1
    Requires: Python 3.11 or 3.12
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Faster downloads

Write-Host "=== Voice-Hermes — Windows Setup ===" -ForegroundColor Cyan

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
Write-Host "[1/7] Checking Python..." -ForegroundColor Green

# Try python3 first (some Windows installs have it), fall back to python
$python = $null
foreach ($cmd in @("python3", "python")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $ver -match "(\d+\.\d+)") {
            $pyVer = [version]$Matches[1]
            if ($pyVer -ge [version]"3.10") {
                $python = $cmd
                Write-Host "   Found $ver"
                break
            }
        }
    } catch { continue }
}

if (-not $python) {
    Write-Error "Python 3.10+ not found. Install it from https://www.python.org/downloads/"
    exit 1
}

# Check for common missing components
try {
    & $python -c "import ensurepip" 2>$null
} catch {
    Write-Warning "   ensurepip module missing — pip may not be available."
}

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
Write-Host "[2/7] Creating virtual environment..." -ForegroundColor Green

if (-not (Test-Path ".venv")) {
    & $python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    Write-Host "   .venv created"
} else {
    Write-Host "   .venv already exists"
}

# Activate via the full path to pip
$pip = "$ScriptDir\.venv\Scripts\pip.exe"
$py = "$ScriptDir\.venv\Scripts\python.exe"

if (-not (Test-Path $pip)) {
    Write-Error "pip not found in virtual environment."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Install Python packages
# ---------------------------------------------------------------------------
Write-Host "[3/7] Installing Python packages..." -ForegroundColor Green

& $pip install --upgrade pip -q
& $pip install -r requirements.txt -q

# scikit-learn is needed by openwakeword's internal preprocessing
& $pip install scikit-learn -q

Write-Host "   Packages installed"

# ---------------------------------------------------------------------------
# 4. Download OpenWakeWord pre-trained models
# ---------------------------------------------------------------------------
Write-Host "[4/7] Downloading OpenWakeWord pre-trained models..." -ForegroundColor Green

# Find openwakeword package directory
$owwDir = & $py -c "import pathlib, openwakeword; print(pathlib.Path(openwakeword.__file__).parent / 'resources' / 'models')"
$modelsDir = [System.Environment]::ExpandEnvironmentVariables($owwDir)
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$models = @{
    "embedding_model.tflite"  = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"
    "melspectrogram.tflite"   = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"
    "silero_vad.onnx"         = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx"
    "alexa_v0.1.tflite"       = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.tflite"
    "hey_mycroft_v0.1.tflite" = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.tflite"
    "hey_jarvis_v0.1.tflite"  = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite"
    "hey_rhasspy_v0.1.tflite" = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_rhasspy_v0.1.tflite"
    "timer_v0.1.tflite"       = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/timer_v0.1.tflite"
    "weather_v0.1.tflite"     = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/weather_v0.1.tflite"
}

foreach ($fname in $models.Keys) {
    $dest = Join-Path $modelsDir $fname
    if (-not (Test-Path $dest)) {
        Write-Host "   Downloading $fname..."
        Invoke-WebRequest -Uri $models[$fname] -OutFile $dest
    } else {
        Write-Host "   $fname already exists"
    }
}
Write-Host "   OpenWakeWord models ready"

# ---------------------------------------------------------------------------
# 5. Download Piper TTS Windows binary
# ---------------------------------------------------------------------------
Write-Host "[5/7] Downloading Piper TTS Windows binary..." -ForegroundColor Green

New-Item -ItemType Directory -Force -Path "models\piper" | Out-Null
$piperExe = "models\piper\piper.exe"

if (-not (Test-Path $piperExe)) {
    $zipUrl = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
    $zipPath = "$env:TEMP\piper_windows.zip"

    Write-Host "   Downloading Piper Windows binary..."
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

    Write-Host "   Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath "models\piper_temp" -Force

    # Move piper.exe + DLLs from piper_temp\piper\* to models\piper\
    if (Test-Path "models\piper_temp\piper\piper.exe") {
        # Move everything from the nested piper dir
        Copy-Item -Path "models\piper_temp\piper\*" -Destination "models\piper\" -Recurse -Force
    }
    Remove-Item -Path "models\piper_temp" -Recurse -Force
    Remove-Item -Path $zipPath -Force

    Write-Host "   Piper binary installed at $piperExe"
} else {
    Write-Host "   Piper binary already exists"
}

# ---------------------------------------------------------------------------
# 6. Download Whisper.cpp model
# ---------------------------------------------------------------------------
Write-Host "[6/7] Downloading Whisper tiny.en model..." -ForegroundColor Green

New-Item -ItemType Directory -Force -Path "models\whisper" | Out-Null
$whisperModel = "models\whisper\ggml-tiny.en.bin"

if (-not (Test-Path $whisperModel)) {
    Write-Host "   Downloading ggml-tiny.en.bin (~75 MB)..."
    Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" `
        -OutFile $whisperModel
    Write-Host "   Whisper model ready"
} else {
    Write-Host "   Whisper model already exists"
}

# ---------------------------------------------------------------------------
# 7. Download Piper voice model
# ---------------------------------------------------------------------------
Write-Host "[7/7] Downloading Piper voice model (en_US-lessac-medium)..." -ForegroundColor Green

$onnxFile = "models\piper\en_US-lessac-medium.onnx"
$jsonFile = "models\piper\en_US-lessac-medium.onnx.json"

if (-not (Test-Path $onnxFile)) {
    Write-Host "   Downloading en_US-lessac-medium.onnx..."
    Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" `
        -OutFile $onnxFile
    Write-Host "   Downloading voice config..."
    Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" `
        -OutFile $jsonFile
    Write-Host "   Piper voice model ready"
} else {
    Write-Host "   Piper voice model already exists"
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Cyan
Write-Host "Activate: .venv\Scripts\activate"
Write-Host "Run:     python -m voice_hermes start"
Write-Host ""
Write-Host "Audio devices on Windows are auto-detected by sounddevice."
Write-Host "If you have issues, run: python -m voice_hermes list-devices"
