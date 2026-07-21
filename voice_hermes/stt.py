"""
Speech-to-text via Whisper.cpp.

Transcribes recorded audio to text using Whisper tiny/small models.
"""

import numpy as np


class STTEngine:
    """Speech-to-text engine using Whisper.cpp."""

    def __init__(self, model_path: str = "", model_size: str = "tiny",
                 language: str = "en", sample_rate: int = 16000):
        self.model_path = model_path
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> str:
        """Transcribe audio to text."""
        raise NotImplementedError
