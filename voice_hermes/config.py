"""
Configuration loader for Voice-Hermes.

Loads TOML config from config.toml with sensible defaults.
Search order: ./config.toml → ~/.voice-hermes/config.toml → /etc/voice-hermes/config.toml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class WakeWordConfig:
    engine: str = "openwakeword"
    model_path: str = ""
    wake_words: list = field(default_factory=lambda: ["hey jarvis"])
    sensitivity: float = 0.5
    frame_length: int = 1280


@dataclass
class STTConfig:
    engine: str = "whisper_cpp"
    model_path: str = "models/whisper/ggml-tiny.en.bin"
    model_size: str = "tiny"
    language: str = "en"
    silence_timeout: float = 1.5
    vad_mode: int = 2
    sample_rate: int = 16000


@dataclass
class TTSConfig:
    engine: str = "piper"
    model_path: str = "models/piper/en_US-lessac-medium.onnx"
    config_path: str = "models/piper/en_US-lessac-medium.onnx.json"
    speaker_id: int = 0
    rate: float = 1.0
    volume: float = 1.0
    device: str = "cpu"


@dataclass
class AgentConfig:
    type: str = "hermes"
    hermes_command: str = "hermes"
    hermes_profile: str = "default"


@dataclass
class InterruptConfig:
    enabled: bool = True
    stop_words: list = field(default_factory=lambda: ["stop", "silence"])
    cooldown: float = 0.5


@dataclass
class AudioConfig:
    input_device: str = ""
    output_device: str = ""
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 1024


@dataclass
class GeneralConfig:
    debug: bool = True
    log_level: str = "info"


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    interrupt: InterruptConfig = field(default_factory=InterruptConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _merge_config(base: dict, overlay: dict) -> dict:
    """Deep merge overlay into base."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from file, merging defaults with user config."""
    search_paths = [
        Path("config.toml"),
        Path.home() / ".voice-hermes" / "config.toml",
        Path("/etc/voice-hermes/config.toml"),
    ]

    if path:
        search_paths.insert(0, Path(path))

    merged: dict = {}
    for sp in search_paths:
        if sp.exists():
            merged = _merge_config(merged, _load_toml(sp))

    if not merged:
        return Config()

    cfg = Config()

    if "general" in merged:
        cfg.general = GeneralConfig(**merged["general"])
    if "wake_word" in merged:
        cfg.wake_word = WakeWordConfig(**merged["wake_word"])
    if "stt" in merged:
        cfg.stt = STTConfig(**merged["stt"])
    if "tts" in merged:
        cfg.tts = TTSConfig(**merged["tts"])
    if "agent" in merged:
        cfg.agent = AgentConfig(**merged["agent"])
    if "interrupt" in merged:
        cfg.interrupt = InterruptConfig(**merged["interrupt"])
    if "audio" in merged:
        cfg.audio = AudioConfig(**merged["audio"])

    return cfg
