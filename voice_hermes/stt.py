"""
Speech-to-text via Whisper.cpp.

Transcribes recorded audio to text using Whisper tiny/small models.
Supports two backends: pywhispercpp (preferred) or subprocess whisper.cpp.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# try importing pywhispercpp
_has_pywhispercpp = False
try:
    import pywhispercpp
    _has_pywhispercpp = True
except ImportError:
    pass


class STTEngine:
    """
    Speech-to-text engine using Whisper.cpp.

    Automatically selects the best available backend:
    1. pywhispercpp (Python bindings) — fastest
    2. Subprocess whisper.cpp binary — fallback

    Typical usage::

        stt = STTEngine(model_size="tiny")
        text = stt.transcribe(audio_array)
        print(text)
    """

    def __init__(
        self,
        model_path: str = "",
        model_size: str = "tiny",
        language: str = "en",
        sample_rate: int = 16000,
    ):
        """
        Args:
            model_path: Path to GGML model file. If empty, derived from model_size.
            model_size: Model size (tiny, base, small).
            language: Language code for transcription.
            sample_rate: Expected audio sample rate.
        """
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate

        # Resolve model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = (
                Path("models/whisper") / f"ggml-{model_size}.en.bin"
            )

        self._model = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available backend: pywhispercpp or subprocess."""
        if _has_pywhispercpp:
            logger.debug("STT backend: pywhispercpp")
            return "pywhispercpp"
        logger.debug("STT backend: subprocess (whisper.cpp binary)")
        return "subprocess"

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {self.model_path}\n"
                f"Run setup.sh to download models."
            )

        if self._backend == "pywhispercpp":
            from pywhispercpp.model import Model
            self._model = Model(
                str(self.model_path),
                n_threads=os.cpu_count() or 4,
            )
            logger.info("Whisper model loaded (backend=pywhispercpp, model=%s)",
                        self.model_path.name)
        else:
            # Subprocess backend — no model object to load
            logger.info("Whisper model ready (backend=subprocess, model=%s)",
                        self.model_path.name)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array (range -1..1).
            sample_rate: Sample rate of the audio (defaults to self.sample_rate).

        Returns:
            Transcribed text string.
        """
        sr = sample_rate or self.sample_rate
        self._load_model()

        if self._backend == "pywhispercpp":
            return self._transcribe_pywhispercpp(audio, sr)
        else:
            return self._transcribe_subprocess(audio, sr)

    def _transcribe_pywhispercpp(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe using pywhispercpp bindings."""
        from pywhispercpp.model import Model

        # Ensure 16kHz mono
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
            sample_rate = 16000

        # pywhispercpp accepts float32 audio directly
        segments = self._model.transcribe(audio, sample_rate=sample_rate)
        text = " ".join(s.text.strip() for s in segments)
        logger.debug("Transcription: %.60s...", text)
        return text.strip() or ""

    def _transcribe_subprocess(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe by writing a temp WAV and running whisper.cpp binary."""
        import soundfile as sf

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
            sample_rate = 16000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")

        try:
            # Determine whisper.cpp binary path
            whisper_bin = self._find_whisper_binary()
            result = subprocess.run(
                [
                    whisper_bin,
                    "-m", str(self.model_path),
                    "-f", tmp_path,
                    "-oj",  # JSON output
                    "-nt", str(os.cpu_count() or 4),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error("whisper.cpp error: %s", result.stderr)
                return ""

            import json
            data = json.loads(result.stdout)
            text = " ".join(s["text"].strip() for s in data.get("segments", []))
            return text.strip() or ""
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Simple linear resampling. For production, use scipy.signal.resample."""
        if orig_rate == target_rate:
            return audio
        import scipy.signal
        number_of_samples = int(len(audio) * target_rate / orig_rate)
        return scipy.signal.resample(audio, number_of_samples)

    @staticmethod
    def _find_whisper_binary() -> str:
        """Locate the whisper.cpp binary."""
        candidates = [
            "whisper.cpp/main",
            "whisper.cpp/build/bin/whisper-cli",
            "whisper.cpp/build/main",
            "/usr/local/bin/whisper-cli",
            # Check if it's in PATH
            "whisper-cli",
        ]
        for candidate in candidates:
            if os.path.isfile(candidate) or (
                not "/" in candidate and (
                    subprocess.run(
                        ["which", candidate], capture_output=True
                    ).returncode == 0
                )
            ):
                return candidate
        # Default fallback
        return "whisper.cpp/main"
