"""
Interruption detector.

Listens for stop keywords during TTS playback and triggers interruption.
"""


class InterruptionDetector:
    """Detects stop keywords while TTS is playing."""

    def __init__(self, stop_words: list[str] | None = None,
                 enabled: bool = True, cooldown: float = 0.5):
        self.stop_words = stop_words or ["stop", "silence"]
        self.enabled = enabled
        self.cooldown = cooldown

    def start(self):
        """Start listening for stop keywords."""
        raise NotImplementedError

    def stop(self):
        """Stop listening."""
        raise NotImplementedError
