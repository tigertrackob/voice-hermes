"""
OpenWakeWord-based wake word detector.

Always-on listener that detects configured wake words from microphone input.
"""


class WakeWordDetector:
    """Always-on wake word detection using OpenWakeWord."""

    def __init__(self, model_path: str = "", sensitivity: float = 0.5,
                 wake_words: list[str] | None = None):
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.wake_words = wake_words or ["hey jarvis"]

    def start(self):
        """Start listening for wake words (blocking or thread)."""
        raise NotImplementedError

    def stop(self):
        """Stop the detector."""
        raise NotImplementedError
