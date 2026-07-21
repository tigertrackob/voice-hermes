"""
Audio capture, playback, and voice activity detection.

Uses sounddevice (PortAudio) for cross-platform audio I/O.
"""

from typing import Optional


def list_devices() -> list[dict]:
    """List available audio input/output devices."""
    import sounddevice as sd
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        devices.append({
            "index": i,
            "name": dev["name"],
            "inputs": dev["max_input_channels"],
            "outputs": dev["max_output_channels"],
            "sample_rate": dev["default_samplerate"],
        })
    return devices


class AudioCapture:
    """Microphone capture with VAD silence detection."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 blocksize: int = 1024, device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._stream = None

    def start(self):
        """Start the capture stream."""
        raise NotImplementedError

    def stop(self):
        """Stop the capture stream."""
        raise NotImplementedError

    def read(self, frames: int) -> bytes:
        """Read audio frames from the microphone."""
        raise NotImplementedError


class AudioPlayback:
    """Non-blocking audio playback with stop/clear."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._stream = None

    def play(self, audio, sample_rate: int):
        """Play audio data (non-blocking)."""
        raise NotImplementedError

    def stop(self):
        """Stop current playback immediately."""
        raise NotImplementedError

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        raise NotImplementedError
