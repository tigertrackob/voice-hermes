"""
OpenWakeWord-based wake word detector with integrated stop-word detection.

Always-on listener that detects configured wake words from microphone input.
Also handles stop-word detection during TTS playback (single mic stream
avoids ALSA device contention).

Runs in a separate thread so it doesn't block the main event loop.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WakeWordEvent:
    """Fired when a wake word or stop word is detected."""
    word: str
    confidence: float
    timestamp: float


class WakeWordDetector:
    """
    Always-on wake word detection using OpenWakeWord.

    Also handles stop-word spotting when stop mode is active
    (activated by orchestrator during TTS playback).
    Single mic stream — no ALSA device contention.

    Typical usage::

        detector = WakeWordDetector(wake_words=["hey jarvis"])
        detector.on_detected = lambda e: print(f"Wake: {e.word}")
        detector.on_interrupt = lambda: print("Stop word heard!")
        detector.start()
        # ... during TTS playback:
        detector.set_stop_mode(True)
        # ... after TTS finishes:
        detector.set_stop_mode(False)
        detector.stop()
    """

    def __init__(
        self,
        wake_words: list[str] | None = None,
        stop_words: list[str] | None = None,
        model_path: str = "",
        sensitivity: float = 0.5,
        stop_sensitivity: float = 0.4,
        stop_cooldown: float = 0.5,
        sample_rate: int = 16000,
        frame_length: int = 1280,
        input_device: Optional[str] = None,
    ):
        self.wake_words = wake_words or ["hey jarvis"]
        self.stop_words = stop_words or ["stop", "silence"]
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.stop_sensitivity = stop_sensitivity
        self.stop_cooldown = stop_cooldown
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.input_device = input_device

        self.on_detected: Optional[Callable[[WakeWordEvent], None]] = None
        self.on_interrupt: Optional[Callable[[], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._model = None
        self._stream = None

        # Stop-word mode (activated during TTS playback)
        self._stop_mode = False
        self._last_stop_trigger = 0.0

    def start(self):
        """Start the detector in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WakeWordDetector already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="wake-word-detector", daemon=True,
        )
        self._thread.start()
        logger.info("WakeWordDetector started (wake=%s, stop=%s)",
                     self.wake_words, self.stop_words)

    def stop(self):
        """Stop the detector thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._close_stream()
        logger.info("WakeWordDetector stopped")

    def set_stop_mode(self, enabled: bool):
        """Enable/disable stop-word detection during TTS playback."""
        self._stop_mode = enabled
        logger.debug("Stop-word detection %s", "ACTIVATED" if enabled else "deactivated")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_model(self):
        if self._model is not None:
            return
        from openwakeword import Model
        if self.model_path:
            self._model = Model(
                wakeword_models=[self.model_path],
                enable_speex_noise_suppression=True,
            )
        else:
            self._model = Model(
                wakeword_models=[],
                enable_speex_noise_suppression=True,
            )
        logger.debug("OpenWakeWord model loaded")

    def _open_stream(self):
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
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _run(self):
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
                logger.warning("Audio read error: %s", e)
                continue

            try:
                prediction = self._model.predict(audio_flat)
            except Exception as e:
                logger.warning("Inference error: %s", e)
                continue

            # Always check wake words (IDLE mode)
            for word in self.wake_words:
                confidence = prediction.get(word, 0.0)
                if confidence >= self.sensitivity:
                    logger.info("Wake word: %s (conf=%.3f)", word, confidence)
                    if self.on_detected:
                        event = WakeWordEvent(
                            word=word, confidence=float(confidence),
                            timestamp=time.time(),
                        )
                        self.on_detected(event)

            # Check stop words only when in stop mode (SPEAKING state)
            if self._stop_mode:
                for word in self.stop_words:
                    confidence = prediction.get(word, 0.0)
                    if confidence >= self.stop_sensitivity:
                        now = time.time()
                        if now - self._last_stop_trigger >= self.stop_cooldown:
                            self._last_stop_trigger = now
                            logger.info("Stop word: %s (conf=%.3f)", word, confidence)
                            if self.on_interrupt:
                                self.on_interrupt()

        self._close_stream()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
