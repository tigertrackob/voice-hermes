"""
Unit tests for the audio module (VAD, config, utility functions).

These tests do NOT require actual microphone/speaker hardware.
Hardware-dependent tests (capture, playback) are marked as integration tests.
"""

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# VAD tests
# ---------------------------------------------------------------------------

class TestVAD:
    """Test the VAD wrapper class."""

    def test_init_valid_sample_rates(self):
        from voice_hermes.audio import VAD
        for rate in (8000, 16000, 32000, 48000):
            vad = VAD(mode=2, sample_rate=rate)
            assert vad.sample_rate == rate

    def test_init_invalid_sample_rate(self):
        from voice_hermes.audio import VAD
        with pytest.raises(ValueError, match="Unsupported.*sample rate"):
            VAD(sample_rate=12345)

    def test_init_valid_modes(self):
        from voice_hermes.audio import VAD
        for mode in range(4):
            vad = VAD(mode=mode)
            assert vad._vad is not None

    def test_is_speech_empty_frame(self):
        """An all-zero frame should generally be classified as non-speech."""
        from voice_hermes.audio import VAD
        vad = VAD(sample_rate=16000)
        frame = b"\x00\x00" * 240  # 480 samples, 16-bit = 960 bytes
        result = vad.is_speech(frame)
        assert isinstance(result, bool)

    def test_is_speech_wrong_frame_size(self):
        from voice_hermes.audio import VAD
        vad = VAD(sample_rate=16000)
        with pytest.raises(ValueError, match="VAD frame must be"):
            vad.is_speech(b"\x00" * 100)

    def test_is_speech_np_conversion(self):
        """is_speech_np should accept float32 array and return bool."""
        from voice_hermes.audio import VAD
        vad = VAD(sample_rate=16000)
        audio = np.zeros(480, dtype=np.float32)
        result = vad.is_speech_np(audio)
        assert isinstance(result, bool)

    def test_frame_count(self):
        from voice_hermes.audio import VAD, VAD_FRAME_MS
        vad = VAD(sample_rate=16000)
        samples_per_frame = 16000 // 1000 * VAD_FRAME_MS  # 480
        audio = np.zeros(samples_per_frame * 5, dtype=np.float32)
        assert vad.frame_count(audio) == 5


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
