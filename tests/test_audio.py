"""
Unit tests for the audio module (VAD, config, utility functions).

These tests do NOT require actual microphone/speaker hardware.
Hardware-dependent tests (capture, playback) are marked as integration tests.
"""

from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# VAD tests
# ---------------------------------------------------------------------------

class TestVAD:
    """Test the energy-based VAD class."""

    def test_init_defaults(self):
        from voice_hermes.audio import VAD
        vad = VAD()
        assert vad.sample_rate == 16000
        assert vad._threshold == 0.008

    def test_init_mode_mapping(self):
        from voice_hermes.audio import VAD
        thresholds = {0: 0.004, 1: 0.006, 2: 0.008, 3: 0.012}
        for mode, expected in thresholds.items():
            vad = VAD(mode=mode)
            assert vad._threshold == expected, f"Mode {mode}: expected {expected}"

    def test_init_custom_threshold(self):
        from voice_hermes.audio import VAD
        vad = VAD(energy_threshold=0.02)
        assert vad._threshold == 0.02

    def test_silence_frame_is_not_speech(self):
        """An all-zero frame should be classified as non-speech."""
        from voice_hermes.audio import VAD
        vad = VAD()
        frame = b"\x00\x00" * 240  # 480 samples, 16-bit = 960 bytes
        assert not vad.is_speech(frame)

    def test_loud_frame_is_speech(self):
        """A full-scale frame should be classified as speech."""
        from voice_hermes.audio import VAD
        vad = VAD()
        # Max amplitude int16 = 32767
        frame = b"\xff\x7f" * 240  # 480 samples of max positive
        assert vad.is_speech(frame)

    def test_is_speech_np_silence(self):
        """is_speech_np should return False for silence."""
        from voice_hermes.audio import VAD
        vad = VAD()
        audio = np.zeros(480, dtype=np.float32)
        assert not vad.is_speech_np(audio)

    def test_is_speech_np_speech(self):
        """is_speech_np should return True for loud audio."""
        from voice_hermes.audio import VAD
        vad = VAD(energy_threshold=0.01)
        audio = np.ones(480, dtype=np.float32) * 0.5
        assert vad.is_speech_np(audio)

    def test_frame_count(self):
        from voice_hermes.audio import VAD, VAD_FRAME_MS
        vad = VAD(sample_rate=16000)
        samples_per_frame = 16000 // 1000 * VAD_FRAME_MS  # 480
        audio = np.zeros(samples_per_frame * 5, dtype=np.float32)
        assert vad.frame_count(audio) == 5

    def test_empty_frame_not_speech(self):
        from voice_hermes.audio import VAD
        vad = VAD()
        assert not vad.is_speech(b"")

    def test_empty_np_not_speech(self):
        from voice_hermes.audio import VAD
        vad = VAD()
        assert not vad.is_speech_np(np.array([], dtype=np.float32))


# ---------------------------------------------------------------------------
# List devices test
# ---------------------------------------------------------------------------

class TestListDevices:
    """Test the device listing utility."""

    @patch("voice_hermes.audio.sd.query_devices")
    def test_list_devices(self, mock_query):
        from voice_hermes.audio import list_devices

        mock_query.return_value = [
            {"name": "hw:0,0", "max_input_channels": 2, "max_output_channels": 0,
             "default_samplerate": 48000},
            {"name": "hw:0,1", "max_input_channels": 0, "max_output_channels": 2,
             "default_samplerate": 48000},
        ]

        devices = list_devices()
        assert len(devices) == 2
        assert devices[0]["name"] == "hw:0,0"
        assert devices[0]["inputs"] == 2
        assert devices[0]["outputs"] == 0
        assert devices[1]["outputs"] == 2


# ---------------------------------------------------------------------------
# Config loader integration
# ---------------------------------------------------------------------------

class TestConfigAudioDefaults:
    """Verify that default audio config values are reasonable."""

    def test_default_audio_config(self):
        from voice_hermes.config import load_config
        cfg = load_config()  # No file exists — returns defaults
        assert cfg.audio.sample_rate == 16000
        assert cfg.audio.channels == 1
        assert cfg.audio.blocksize == 1024
        assert cfg.stt.sample_rate == 16000  # Must match audio sample_rate


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestAudioUtilities:

    def test_get_default_input_device(self):
        from voice_hermes.audio import get_default_input_device
        with patch("voice_hermes.audio.sd.query_devices") as mock_q:
            mock_q.return_value = {"index": 1, "name": "default"}
            assert get_default_input_device() == 1

    def test_get_default_input_device_fallback(self):
        from voice_hermes.audio import get_default_input_device
        with patch("voice_hermes.audio.sd.query_devices",
                   side_effect=ValueError("bad index")):
            assert get_default_input_device() is None
