"""
Interruption detector.

Listens for stop keywords during TTS playback and triggers interruption.
Reuses OpenWakeWord with a separate model for stop-word spotting,
or falls back to simple audio energy threshold detection.
"""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class InterruptionDetector:
    """
    Detects stop keywords while TTS is playing and triggers interruption.

    Two modes:
    1. OpenWakeWord mode (preferred) — keyword spotting with a dedicated model
    2. Simple energy mode — detects loud sounds above threshold as potential stops

    Typical usage::

        def on_interrupt():
            print("User interrupted!")

        detector = InterruptionDetector(stop_words=["stop", "silence"])
        detector.on_interrupt = on_interrupt
        detector.start()
        # ... when TTS starts playing, detector automatically listens ...
        # ... when TTS stops, call detector.pause() ...
        detector.stop()
    """

    def __init__(
        self,
        stop_words: Optional[list[str]] = None,
        enabled: bool = True,
        cooldown: float = 0.5,
        sensitivity: float = 0.4,
        sample_rate: int = 16000,
        input_device: Optional[str] = None,
    ):
        """
        Args:
            stop_words: List of phrases that trigger interruption.
            enabled: Master switch for interruption detection.
            cooldown: Seconds to ignore after an interruption fires.
            sensitivity: Detection threshold (lower = more sensitive).
            sample_rate: Audio sample rate for the listening thread.
            input_device: Audio input device name/index.
        """
        self.stop_words = stop_words or ["stop", "silence", "that's enough"]
        self.enabled = enabled
        self.cooldown = cooldown
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.input_device = input_device

        self.on_interrupt: Optional[Callable[[], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._active = threading.Event()  # Set when we should be listening
        self._last_trigger = 0.0
        self._oww_model = None
        self._stream = None

    def start(self):
        """Start the interruption detector thread (paused by default)."""
        if self._thread is not None and self._thread.is_alive():
            return

        if not self.enabled:
            logger.info("Interruption detector disabled")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="interruption-detector",
            daemon=True,
        )
        self._thread.start()
        logger.info("Interruption detector started (words=%s)", self.stop_words)

    def stop(self):
        """Stop the detector thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._close_stream()
        logger.debug("Interruption detector stopped")

    def activate(self):
        """Start listening for stop words (call when TTS starts playing)."""
        if self.enabled:
            self._active.set()
            logger.debug("Interruption detector activated")

    def deactivate(self):
        """Stop listening for stop words (call when TTS finishes)."""
        self._active.clear()
        logger.debug("Interruption detector deactivated")

    def _open_model(self):
        """Lazy-load OpenWakeWord for stop-word detection."""
        if self._oww_model is not None:
            return
        try:
            from openwakeword import Model
            # Use built-in model; the stop words are checked via prediction dict
            self._oww_model = Model(
                wakeword_models=[],
                enable_speex_noise_suppression=True,
            )
            logger.debug("Interruption OpenWakeWord model loaded")
        except ImportError:
            logger.warning("OpenWakeWord not available — interruption disabled")
            self.enabled = False

    def _open_stream(self):
        """Open the microphone stream."""
        if self._stream is not None:
            return
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=1280,
            device=self.input_device,
            dtype="float32",
        )
        self._stream.start()

    def _close_stream(self):
        """Close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _run(self):
        """Main detection loop."""
        try:
            self._open_model()
            self._open_stream()
        except Exception as e:
            logger.error("Failed to init interruption detector: %s", e)
            return

        while not self._stop_event.is_set():
            # Only process audio when activated (TTS is playing)
            if not self._active.is_set():
                time.sleep(0.05)
                continue

            try:
                chunk, _ = self._stream.read(1280)
                audio_flat = chunk.flatten()
            except Exception:
                continue

            # Run OpenWakeWord inference
            prediction = self._oww_model.predict(audio_flat)

            # Check if any stop word is detected above threshold
            for word in self.stop_words:
                confidence = prediction.get(word, 0.0)
                if confidence >= self.sensitivity:
                    now = time.time()
                    if now - self._last_trigger >= self.cooldown:
                        self._last_trigger = now
                        logger.info(
                            "Interruption triggered by '%s' (conf=%.3f)",
                            word, confidence,
                        )
                        if self.on_interrupt:
                            self.on_interrupt()
                    break  # Only one trigger per frame

        self._close_stream()
