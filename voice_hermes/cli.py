"""
Command-line interface for Voice-Hermes.

Subcommands:
  start         Launch the orchestrator (foreground)
  status        Check if voice-hermes is running
  list-devices  List audio input/output devices
  test-mic      Test microphone: record and show audio levels
  test-speaker  Play a test tone through speakers
"""

import argparse
import logging
import sys

import numpy as np

from voice_hermes import __version__

logger = logging.getLogger("voice_hermes.cli")


def _setup_logging(debug: bool = False):
    """Configure console logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def cmd_start(args):
    """Start the voice-hermes orchestrator (foreground)."""
    from voice_hermes.config import load_config
    from voice_hermes.orchestrator import VoiceHermesOrchestrator

    _setup_logging(debug=args.verbose)
    config = load_config(args.config)
    orch = VoiceHermesOrchestrator(config)

    try:
        import asyncio
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)


def cmd_status(args):
    """Check if voice-hermes is running."""
    import os
    import subprocess

    try:
        result = subprocess.run(
            ["systemctl", "is-active", "voice-hermes"],
            capture_output=True, text=True, timeout=5,
        )
        status = result.stdout.strip()
        if status == "active":
            print("✅ voice-hermes is RUNNING")
            # Show PID from running unit
            pid_result = subprocess.run(
                ["systemctl", "show", "voice-hermes", "--property", "MainPID"],
                capture_output=True, text=True, timeout=5,
            )
            if pid_result.stdout:
                print(f"   PID: {pid_result.stdout.strip().split('=')[1]}")
            # Show recent logs
            log_result = subprocess.run(
                ["journalctl", "-u", "voice-hermes", "-n", "5", "--no-pager"],
                capture_output=True, text=True, timeout=5,
            )
            if log_result.stdout:
                print("   Recent logs:")
                for line in log_result.stdout.strip().split("\n")[-5:]:
                    print(f"     {line}")
        elif status == "inactive":
            print("⏹️  voice-hermes is STOPPED (systemd unit exists)")
            print("   Start with: sudo systemctl start voice-hermes")
        else:
            print(f"⚠️  voice-hermes status: {status}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  systemd not available or unit not found")
        print("   Run in foreground: python -m voice_hermes start")


def cmd_list_devices(args):
    """List available audio devices."""
    from voice_hermes.audio import list_devices
    _setup_logging(debug=False)
    devices = list_devices()
    if not devices:
        print("No audio devices found.")
        return
    print(f"Found {len(devices)} audio device(s):\n")
    for d in devices:
        io = []
        if d["inputs"]:
            io.append(f"{d['inputs']} in")
        if d["outputs"]:
            io.append(f"{d['outputs']} out")
        print(f"  [{d['index']}] {d['name']}")
        print(f"       ({', '.join(io)}) @ {d['sample_rate']:.0f} Hz")
    print()

    # Show defaults
    from voice_hermes.audio import get_default_input_device, get_default_output_device
    default_in = get_default_input_device()
    default_out = get_default_output_device()
    if default_in is not None:
        print(f"   → Default input device: [{default_in}]")
    if default_out is not None:
        print(f"   → Default output device: [{default_out}]")


def cmd_test_mic(args):
    """Record a short sample and report audio levels."""
    from voice_hermes.audio import AudioCapture
    _setup_logging(debug=False)

    duration = args.duration or 3
    print(f"🎤 Recording for up to {duration}s... (speak to test)")
    print("   Press Ctrl+C to stop early.\n")

    cap = AudioCapture()
    cap.start()
    try:
        audio = cap.record_until_silence(
            timeout=duration,
            silence_duration=0.3,
            min_duration=0.1,
        )
    except KeyboardInterrupt:
        audio = np.array([], dtype=np.float32)
        print()
    cap.stop()

    if len(audio) == 0:
        print("No audio captured.")
        return

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Captured {len(audio) / cap.sample_rate:.2f}s of audio")
    print(f"Peak level: {peak:.4f}  RMS: {rms:.4f}")

    if peak < 0.01:
        print("⚠️  Very quiet — check your microphone gain or input device.")
    elif peak > 0.9:
        print("⚠️  Very loud — possible clipping.")
    else:
        print("✅ Mic level looks good.")


def cmd_test_speaker(args):
    """Play a test tone through speakers."""
    from voice_hermes.audio import AudioPlayback
    _setup_logging(debug=False)

    sample_rate = 16000
    duration = args.duration or 1.0
    freq = args.freq or 440  # A4

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fade in/out to avoid clicks
    fade_samples = int(sample_rate * 0.05)
    fade = np.minimum(
        np.linspace(0, 1, fade_samples),
        np.linspace(1, 0, fade_samples),
    )
    fade = np.clip(fade, 0, 1)
    if duration > 0.1:
        envelope = np.ones_like(t)
        envelope[:fade_samples] = fade[:fade_samples]
        envelope[-fade_samples:] = fade[-fade_samples:]
    else:
        envelope = np.ones_like(t)

    tone = (0.3 * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)

    print(f"🔊 Playing {freq}Hz tone for {duration}s...")
    play = AudioPlayback()
    play.play(tone, sample_rate)
    play.wait()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="voice-hermes",
        description="Voice-Hermes Agent — Headless voice interface for Hermes Agent",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"voice-hermes {__version__}",
    )
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    # start
    sub.add_parser("start", help="Launch the orchestrator (foreground)")

    # status
    sub.add_parser("status", help="Check if voice-hermes is running")

    # list-devices
    sub.add_parser("list-devices", help="List available audio devices")

    # test-mic
    mic_parser = sub.add_parser("test-mic", help="Record sample & report audio levels")
    mic_parser.add_argument("-d", "--duration", type=float, default=3.0,
                            help="Max recording duration in seconds (default: 3)")

    # test-speaker
    spk_parser = sub.add_parser("test-speaker", help="Play a test tone")
    spk_parser.add_argument("-d", "--duration", type=float, default=1.0,
                            help="Tone duration in seconds (default: 1)")
    spk_parser.add_argument("-f", "--freq", type=float, default=440,
                            help="Tone frequency in Hz (default: 440)")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "list-devices":
        cmd_list_devices(args)
    elif args.command == "test-mic":
        cmd_test_mic(args)
    elif args.command == "test-speaker":
        cmd_test_speaker(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
