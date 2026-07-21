"""
Interruption detector — thin wrapper that controls stop-word mode
on the wake word detector (avoids separate mic stream contention).

During TTS playback, the orchestrator activates stop mode on the
wake word detector, which checks for stop words on the same audio
frames it's already processing.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InterruptionDetector:
    """
    Controls stop-word detection on the WakeWordDetector (no own mic stream).

    Typical usage::

        detector = InterruptionDetector(wake_word_detector, stop_words=["stop"])
        detector.on_interrupt = my_callback
        detector.start()
        # ... when TTS starts:
        detector.activate()
        # ... when TTS finishes:
        detector.deactivate()
        detector.stop()
    """

    def __init__(
        self,
        wake_word_detector=None,
        stop_words: Optional[list[str]] = None,
        enabled: bool = True,
    ):
        """
        Args:
            wake_word_detector: The WakeWordDetector instance to control.
            stop_words: List of stop phrases (passed to wake word detector).
            enabled: Master switch for interruption detection.
        """
        self._wwd = wake_word_detector
        self.enabled = enabled
        self.on_interrupt: Optional[callable] = None

    def start(self):
        """Wire up to the wake word detector's interrupt callback."""
        if not self.enabled:
            logger.info("Interruption detector disabled")
            return
        if self._wwd is None:
            logger.warning("No wake word detector provided — interruption unavailable")
            return
        self._wwd.on_interrupt = self._on_interrupt
        logger.info("Interruption detector ready (stop words=%s)",
                     getattr(self._wwd, 'stop_words', []))

    def stop(self):
        """Disconnect from wake word detector."""
        if self._wwd is not None:
            self._wwd.set_stop_mode(False)
            self._wwd.on_interrupt = None
        logger.debug("Interruption detector stopped")

    def activate(self):
        """Enable stop-word detection (call when TTS starts playing)."""
        if self.enabled and self._wwd is not None:
            self._wwd.set_stop_mode(True)

    def deactivate(self):
        """Disable stop-word detection (call when TTS finishes)."""
        if self._wwd is not None:
            self._wwd.set_stop_mode(False)

    def _on_interrupt(self):
        """Called by wake word detector when a stop word is heard."""
        logger.info("Interruption triggered by stop word")
        if self.on_interrupt:
            self.on_interrupt()
