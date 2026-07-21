"""
Text-to-speech via Piper TTS.

Converts agent response text to speech using Piper ONNX models.
Supports switching voice profiles (.onnx files) at runtime.
Non-blocking playback with immediate stop for interruption.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-speech engine using Piper TTS with ONNX voice profiles.

    Supports:
    - Non-blocking playback via background thread
    - Immediate stop for interruption
    - Runtime voice switching (.onnx + .json config)
    - Queueing multiple utterances

    Typical usage::

        tts = TTSEngine("models/piper/en_US-lessac-medium.onnx")
        tts.speak("Hello, how can I help you?")
        # ... later, if interrupted:
        tts.stop()
    """

    def __init__(
        self,
        model_path: str = "models/piper/en_US-lessac-medium.onnx",
        config_path: str = "",
        device: str = "cpu",
        output_device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to Piper .onnx voice model.
            config_path: Path to Piper .onnx.json config. Auto-derived if empty.
            device: Inference device ("cpu" or "cuda").
            output_device: Audio output device name/index for playback.
        """
        self.model_path = Path(model_path)
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.model_path.with_suffix(".onnx.json")
        self.device = device
        self.output_device = output_device

        self._voice = None
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """
        Synthesize *text* and play it (non-blocking).

        If already speaking, the new text is queued and played after
        the current utterance finishes.
        """
        if not text.strip():
            return
        self._queue.put(text.strip())
        if self._playback_thread is None or not self._playback_thread.is_alive():
            self._start_playback()

    def stop(self):
        """Stop current and queued playback immediately (for interruption)."""
        self._stop_event.set()
        self._clear_queue()
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=3)
            self._playback_thread = None
        logger.debug("TTS stopped (interrupted)")

    def is_speaking(self) -> bool:
        """Check if TTS is currently playing audio."""
        return (
            self._playback_thread is not None
            and self._playback_thread.is_alive()
            and not self._queue.empty()
        )

    def is_busy(self) -> bool:
        """Check if TTS has pending utterances."""
        return self.is_speaking() or not self._queue.empty()

    def set_voice(self, onnx_path: str, config_path: str = ""):
        """
        Switch voice profile at runtime.

        Args:
            onnx_path: Path to new .onnx voice model.
            config_path: Path to corresponding .json config.
        """
        self.model_path = Path(onnx_path)
        self.config_path = Path(config_path) if config_path else self.model_path.with_suffix(".onnx.json")
        self._voice = None  # Force reload on next speak
        logger.info("TTS voice switched to %s", self.model_path.name)

    def wait_until_done(self):
        """Block until all queued speech is finished."""
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=60)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_voice(self):
        """Lazy-load the Piper voice model."""
        if self._voice is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {self.model_path}\n"
                f"Run setup.sh to download models."
            )

        import piper
        logger.info("Loading Piper voice: %s", self.model_path.name)
        self._voice = piper.load_voice(
            str(self.model_path),
            config_path=str(self.config_path) if self.config_path.exists() else None,
        )
        logger.debug("Piper voice loaded (sample_rate=%d)",
                     self._voice.config.sample_rate)

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio array."""
        self._load_voice()
        import piper
        audio = piper.synthesize(text, self._voice)
        return audio, self._voice.config.sample_rate

    def _start_playback(self):
        """Start the playback thread."""
        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="tts-playback",
            daemon=True,
        )
        self._playback_thread.start()

    def _playback_loop(self):
        """Background loop: synthesize + play queued texts."""
        import sounddevice as sd

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            try:
                # Synthesize
                audio, sample_rate = self._synthesize(text)
                if len(audio) == 0:
                    continue

                # Play (blocking within this thread, but non-blocking for caller)
                sd.play(audio, samplerate=sample_rate, device=self.output_device)
                sd.wait()
            except Exception as e:
                logger.error("TTS playback error: %s", e)

    def _clear_queue(self):
        """Drain all pending items from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
