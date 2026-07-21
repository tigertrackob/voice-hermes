"""
Text-to-speech via Piper TTS.

Converts agent response text to speech using Piper ONNX models.
Supports switching voice profiles (.onnx files).
"""

import numpy as np


class TTSEngine:
    """Text-to-speech using Piper TTS with ONNX voice profiles."""

    def __init__(self, model_path: str = "", config_path: str = "",
                 device: str = "cpu"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio (returns audio array + sample rate)."""
        raise NotImplementedError

    def play(self, audio: np.ndarray, sample_rate: int):
        """Play audio (non-blocking)."""
        raise NotImplementedError

    def stop(self):
        """Stop current playback immediately."""
        raise NotImplementedError

    def is_playing(self) -> bool:
        """Check if TTS is currently playing."""
        raise NotImplementedError

    def set_voice(self, onnx_path: str, config_path: str):
        """Switch voice profile."""
        raise NotImplementedError
