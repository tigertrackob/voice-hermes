"""
Text-to-speech via Piper TTS (subprocess mode).

Uses the Piper command-line binary to synthesize speech from text.
The binary is downloaded separately (see setup.sh).
Falls back to espeak-ng if Piper binary is unavailable.
"""

import logging
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# Piper binary names per platform
PIPER_BIN_NAMES = {
    "linux": "piper",
    "linux_aarch64": "piper",
    "darwin": "piper",
    "win32": "piper.exe",
}

ESPEAK_AVAILABLE = os.system("which espeak-ng >/dev/null 2>&1") == 0


class TTSEngine:
    """
    Text-to-speech engine using Piper TTS (subprocess) with espeak-ng fallback.

    Supports:
    - Non-blocking playback via background thread + queue
    - Immediate stop for interruption
    - Runtime voice switching (.onnx files)

    Typical usage::

        tts = TTSEngine()
        tts.speak("Hello, how can I help you?")
        # ... later:
        tts.stop()
    """

    def __init__(
        self,
        model_path: str = "models/piper/en_US-lessac-medium.onnx",
        config_path: str = "",
        output_device: Optional[str] = None,
        piper_bin: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else self.model_path.with_suffix(".onnx.json")
        self.output_device = output_device
        self._piper_bin = piper_bin

        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """Synthesize *text* and play it (non-blocking)."""
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
        )

    def is_busy(self) -> bool:
        """Check if TTS has pending utterances."""
        return self.is_speaking() or not self._queue.empty()

    def set_voice(self, onnx_path: str, config_path: str = ""):
        """Switch voice profile at runtime."""
        self.model_path = Path(onnx_path)
        self.config_path = Path(config_path) if config_path else self.model_path.with_suffix(".onnx.json")
        logger.info("TTS voice switched to %s", self.model_path.name)

    def wait_until_done(self):
        """Block until all queued speech is finished."""
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=60)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_piper_binary(self) -> Optional[str]:
        """Locate the Piper binary."""
        if self._piper_bin and os.path.isfile(self._piper_bin):
            return self._piper_bin

        candidates = [
            "piper",
            "./piper",
            "./piper/piper",
            "/usr/local/bin/piper",
            str(Path("models/piper/piper")),
        ]
        for c in candidates:
            if os.path.isfile(c) or os.path.isfile(os.path.expanduser(c)):
                return c
            # Check PATH
            if "/" not in c:
                result = subprocess.run(
                    ["which", c], capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        return None

    def _synthesize_piper(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize using the Piper binary via subprocess."""
        piper_bin = self._find_piper_binary()
        if piper_bin is None:
            logger.warning("Piper binary not found")
            return None

        if not self.model_path.exists():
            logger.error("Piper voice model not found: %s", self.model_path)
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                piper_bin,
                "--model", str(self.model_path),
                "--output-raw",
            ]
            if self.config_path.exists():
                cmd.extend(["--config", str(self.config_path)])

            # Run piper: feed text via stdin, get raw audio via stdout
            proc = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=False,  # binary mode
                timeout=60,
            )

            if proc.returncode != 0:
                logger.error("Piper error: %s",
                             proc.stderr.decode(errors="replace")[:200])
                return None

            # Raw audio is 16-bit PCM at Piper's sample rate (usually 22050)
            raw_audio = np.frombuffer(proc.stdout, dtype=np.int16)
            if len(raw_audio) == 0:
                return None

            audio_float = raw_audio.astype(np.float32) / 32767.0

            # Piper default sample rate is 22050
            return audio_float, 22050

        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out")
            return None
        except Exception as e:
            logger.error("Piper synthesis error: %s", e)
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _synthesize_espeak(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Fallback: synthesize using espeak-ng via subprocess."""
        if not ESPEAK_AVAILABLE:
            logger.warning("espeak-ng not available")
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                ["espeak-ng", "-w", tmp_path, text],
                capture_output=True,
                timeout=30,
                check=True,
            )
            audio, sample_rate = sf.read(tmp_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # mono
            return audio.astype(np.float32), sample_rate
        except Exception as e:
            logger.error("espeak-ng error: %s", e)
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _synthesize(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize text to audio using best available backend."""
        # Try Piper first
        result = self._synthesize_piper(text)
        if result is not None:
            return result
        # Fallback to espeak-ng
        result = self._synthesize_espeak(text)
        if result is not None:
            logger.info("Using espeak-ng fallback for TTS")
            return result
        logger.error("No TTS backend available")
        return None

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
                result = self._synthesize(text)
                if result is None:
                    continue
                audio, sample_rate = result
                if len(audio) == 0:
                    continue

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
