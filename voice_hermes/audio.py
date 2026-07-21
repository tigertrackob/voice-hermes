"""
Audio capture, playback, and voice activity detection.

Uses sounddevice (PortAudio) for cross-platform audio I/O.
Voice activity detection uses energy-based thresholding (no external deps).

AudioCapture: microphone capture with silence-gated recording
AudioPlayback: non-blocking playback with immediate stop (for interruption)
"""

import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_BLOCKSIZE = 1024
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = 480       # 16kHz × 30ms = 480 samples
VAD_ENERGY_THRESHOLD = 0.008  # RMS energy threshold for speech detection


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_devices() -> list[dict]:
    """List available audio input/output devices with metadata."""
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


def get_default_input_device() -> Optional[int]:
    """Return the index of the default input device, or None."""
    try:
        dev = sd.query_devices(kind="input")
        return dev["index"]
    except (ValueError, KeyError):
        return None


def get_default_output_device() -> Optional[int]:
    """Return the index of the default output device, or None."""
    try:
        dev = sd.query_devices(kind="output")
        return dev["index"]
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection) utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection) — energy-based, no external deps
# ---------------------------------------------------------------------------

class VAD:
    """Energy-based voice activity detection.

    Detects speech by measuring RMS energy against a threshold.
    No external dependencies — works on all Python versions.
    """

    def __init__(
        self,
        mode: int = 2,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        energy_threshold: float = VAD_ENERGY_THRESHOLD,
    ):
        """
        Args:
            mode: Aggressiveness 0–3 (higher = more aggressive).
                  Maps to higher energy thresholds: 0→0.004, 1→0.006,
                  2→0.008 (default), 3→0.012.
            sample_rate: Audio sample rate (affects frame size).
            energy_threshold: Override threshold directly (overrides mode).
        """
        self.sample_rate = sample_rate
        self._threshold = energy_threshold
        # Map mode to threshold if not explicitly set
        if energy_threshold == VAD_ENERGY_THRESHOLD:
            mode_map = {0: 0.004, 1: 0.006, 2: 0.008, 3: 0.012}
            self._threshold = mode_map.get(mode, 0.008)

    def is_speech(self, frame: bytes) -> bool:
        """Return True if the 30ms PCM16 frame contains speech."""
        # Convert bytes to int16, compute RMS
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return False
        rms = float(np.sqrt(np.mean(samples ** 2))) / 32767.0
        return rms >= self._threshold

    def is_speech_np(self, audio_chunk: np.ndarray) -> bool:
        """Convenience: accept float32 numpy array (range -1..1)."""
        if len(audio_chunk) == 0:
            return False
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        return rms >= self._threshold

    def frame_count(self, audio: np.ndarray) -> int:
        """Number of complete VAD frames in *audio*."""
        samples_per_frame = self.sample_rate // 1000 * VAD_FRAME_MS
        return len(audio) // samples_per_frame


# ---------------------------------------------------------------------------
# Audio Capture
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Microphone capture with VAD-based silence detection.

    Typical usage::

        cap = AudioCapture()
        cap.start()
        audio = cap.record_until_silence(silence_duration=1.5)
        cap.stop()
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        blocksize: int = DEFAULT_BLOCKSIZE,
        device: Optional[str] = None,
        vad_mode: int = 2,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self.vad = VAD(mode=vad_mode, sample_rate=sample_rate)
        self._stream: Optional[sd.InputStream] = None
        self._buffer: list[bytes] = []

    def start(self):
        """Open the input stream."""
        if self._stream is not None:
            return
        self._buffer = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            device=self.device,
            dtype="float32",
        )
        self._stream.start()
        logger.debug("AudioCapture started (rate=%d, device=%s)",
                      self.sample_rate, self.device)

    def stop(self):
        """Close the input stream."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.debug("AudioCapture stopped")

    def read(self, frames: Optional[int] = None) -> np.ndarray:
        """
        Read a chunk of audio as a float32 numpy array (range -1..1).

        If *frames* is None, uses the blocksize set at construction.
        """
        if self._stream is None:
            raise RuntimeError("Capture not started. Call start() first.")
        chunk, _ = self._stream.read(frames or self.blocksize)
        return chunk.flatten()

    def record_until_silence(
        self,
        timeout: float = 30.0,
        silence_duration: float = 1.5,
        min_duration: float = 0.2,
    ) -> np.ndarray:
        """
        Record audio until *silence_duration* seconds of silence is detected
        (or *timeout* seconds elapses).

        Returns the full recorded audio as float32 numpy array.

        Args:
            timeout: Max recording time in seconds.
            silence_duration: Seconds of continuous silence to stop recording.
            min_duration: Minimum recording length before silence can stop capture.
        """
        if self._stream is None:
            raise RuntimeError("Capture not started. Call start() first.")

        vad_frame_samples = self.sample_rate // 1000 * VAD_FRAME_MS  # 480 @ 16kHz
        frames_per_vad = max(1, vad_frame_samples // self.blocksize)

        silence_frames_needed = int(silence_duration / VAD_FRAME_MS * 1000)
        max_frames = int(timeout / VAD_FRAME_MS * 1000)
        min_frames = int(min_duration / VAD_FRAME_MS * 1000)

        audio_chunks: list[np.ndarray] = []
        silence_run = 0
        total_frames = 0

        logger.info("Recording... (silence timeout=%.1fs, max=%.1fs)",
                     silence_duration, timeout)

        while total_frames < max_frames:
            chunk = self.read()
            audio_chunks.append(chunk)

            # VAD checks are done at 30ms frame granularity
            total_frames += len(chunk) // vad_frame_samples

            # Only start checking for silence after min_duration
            if total_frames < min_frames:
                continue

            # Check the latest frame for speech
            # Convert the last vad_frame_samples to PCM bytes for VAD
            frame_end = chunk  # take last chunk
            if len(frame_end) >= vad_frame_samples:
                frame_bytes = (frame_end[:vad_frame_samples] * 32767).astype(
                    np.int16
                ).tobytes()
                is_speech = self.vad.is_speech(frame_bytes)
            else:
                is_speech = True  # conservative

            if is_speech:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= silence_frames_needed:
                    logger.info("Silence detected — stopping recording")
                    break

        result = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
        logger.info("Recording finished: %.2fs", len(result) / self.sample_rate)
        return result


# ---------------------------------------------------------------------------
# Audio Playback
# ---------------------------------------------------------------------------

class AudioPlayback:
    """
    Non-blocking audio playback that supports immediate stop (for interruption).

    Typical usage::

        play = AudioPlayback()
        play.play(audio_array, sample_rate)
        # ... later, if interrupted:
        play.stop()
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._stream: Optional[sd.OutputStream] = None

    def play(self, audio: np.ndarray, sample_rate: int):
        """
        Start (or queue) non-blocking playback of *audio*.

        If audio is already playing, the new audio replaces the current
        buffer (simple approach — stop current, start new).
        """
        self.stop()  # flush any existing playback

        if audio.size == 0:
            return

        audio = np.asarray(audio, dtype=np.float32).flatten()

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            device=self.device,
            dtype="float32",
        )
        self._stream.start()
        self._stream.write(audio)
        logger.debug("Playback started (%.2fs)", len(audio) / sample_rate)

    def wait(self):
        """Block until current playback finishes."""
        if self._stream is not None:
            self._stream.wait()
            self._stream.close()
            self._stream = None

    def stop(self):
        """Stop playback immediately (for interruption)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.debug("Playback stopped (interrupted)")

    def is_playing(self) -> bool:
        """Check if playback is active."""
        return self._stream is not None and self._stream.active


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def play_audio(audio: np.ndarray, sample_rate: int, device: Optional[str] = None):
    """Simple blocking playback (fire-and-forget)."""
    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()
