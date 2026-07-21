"""
OpenWakeWord-based wake word detector.

Always-on listener that detects configured wake words from microphone input.
Runs in a separate thread so it doesn't block the main event loop.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WakeWordEvent:
    """Fired when a wake word is detected."""
    word: str
    confidence: float
    timestamp: float


class WakeWordDetector:
    """
    Always-on wake word detection using OpenWakeWord.

    Listens on a microphone stream in a background thread and fires
    callbacks when a configured wake word is detected above the
    confidence threshold.

    Typical usage::

        detector = WakeWordDetector(
            wake_words=["hey jarvis"],
            sensitivity=0.5,
        )

        def on_wake(event: WakeWordEvent):
            print(f"Wake word detected: {event.word} ({event.confidence:.2f})")

        detector.on_detected = on_wake
        detector.start()
        # ... run main loop ...
        detector.stop()
    """

    def __init__(
        self,
        wake_words: list[str] | None = None,
        model_path: str = "",
        sensitivity: float = 0.5,
        sample_rate: int = 16000,
        frame_length: int = 1280,
        input_device: Optional[str] = None,
    ):
        """
        Args:
            wake_words: List of wake word phrases to detect.
            model_path: Path to custom OpenWakeWord .onnx model (empty = built-in).
            sensitivity: Detection threshold 0.0 (strict) to 1.0 (lenient).
            sample_rate: Audio sample rate (must be 16kHz for OpenWakeWord).
            frame_length: Audio frame size in samples (default: 1280 = 80ms).
            input_device: Audio input device name/index.
        """
        self.wake_words = wake_words or ["hey jarvis"]
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.input_device = input_device

        self.on_detected: Optional[Callable[[WakeWordEvent], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._model = None
        self._stream = None

    def start(self):
        """Start the wake word detector in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WakeWordDetector already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="wake-word-detector",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "WakeWordDetector started (words=%s, sensitivity=%.2f)",
            self.wake_words, self.sensitivity,
        )

    def stop(self):
        """Stop the wake word detector."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._close_stream()
        logger.info("WakeWordDetector stopped")

    def _open_model(self):
        """Lazy-load the OpenWakeWord model."""
        if self._model is not None:
            return
        from openwakeword import Model
        if self.model_path:
            self._model = Model(
                wakeword_models=[self.model_path],
                enable_speex_noise_suppression=True,
            )
        else:
            # Use built-in pre-trained wake words (e.g. "hey jarvis", "alexa")
            self._model = Model(
                wakeword_models=[],
                enable_speex_noise_suppression=True,
            )
        logger.debug("OpenWakeWord model loaded")

    def _open_stream(self):
        """Open the audio input stream."""
        if self._stream is not None:
            return
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_length,
            device=self.input_device,
            dtype="float32",
        )
        self._stream.start()

    def _close_stream(self):
        """Close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _run(self):
        """Main detection loop (runs in background thread)."""
        try:
            self._open_model()
            self._open_stream()
        except Exception as e:
            logger.error("Failed to initialize wake word detector: %s", e)
            return

        while not self._stop_event.is_set():
            try:
                chunk, _ = self._stream.read(self.frame_length)
                audio_flat = chunk.flatten()
            except Exception as e:
                logger.warning("Audio read error in wake word detector: %s", e)
                continue

            # Run OpenWakeWord inference
            prediction = self._model.predict(audio_flat)

            # Check each configured wake word
            for word in self.wake_words:
                confidence = prediction.get(word, 0.0)
                if confidence >= self.sensitivity:
                    logger.info(
                        "Wake word detected: %s (confidence=%.3f)",
                        word, confidence,
                    )
                    if self.on_detected:
                        import time
                        event = WakeWordEvent(
                            word=word,
                            confidence=float(confidence),
                            timestamp=time.time(),
                        )
                        self.on_detected(event)

        self._close_stream()

    def is_running(self) -> bool:
        """Check if the detector thread is alive."""
        return self._thread is not None and self._thread.is_alive()
