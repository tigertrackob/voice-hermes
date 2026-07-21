"""
Command-line interface for Voice-Hermes.

Subcommands:
  start         Launch the orchestrator
  list-devices  List audio input/output devices
  test-mic      Test microphone: record and show audio levels
  test-speaker  Play a test tone through speakers
"""

import argparse
import sys
import time

import numpy as np


def cmd_start(args):
    """Start the voice-hermes orchestrator."""
    from voice_hermes.config import load_config
    from voice_hermes.orchestrator import VoiceHermesOrchestrator
    config = load_config(args.config)
    orch = VoiceHermesOrchestrator(config)
    try:
        import asyncio
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        orch.shutdown()


def cmd_list_devices(args):
    """List available audio devices."""
    from voice_hermes.audio import list_devices
    devices = list_devices()
    if not devices:
        print("No audio devices found.")
        return
    for d in devices:
        io = []
        if d["inputs"]:
            io.append(f"{d['inputs']} in")
        if d["outputs"]:
            io.append(f"{d['outputs']} out")
        print(f"[{d['index']}] {d['name']}  ({', '.join(io)}) @ {d['sample_rate']:.0f} Hz")


def cmd_test_mic(args):
    """Record a short sample and report audio levels."""
    from voice_hermes.audio import AudioCapture

    duration = args.duration or 3
    print(f"Recording for {duration}s... (speak to test)")
    cap = AudioCapture()
    cap.start()
    audio = cap.record_until_silence(
        timeout=duration,
        silence_duration=0.3,
        min_duration=0.1,
    )
    cap.stop()

    if len(audio) == 0:
        print("No audio captured.")
        return

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"Captured {len(audio) / cap.sample_rate:.2f}s of audio")
    print(f"Peak level: {peak:.4f}  RMS: {rms:.4f}")
    if peak < 0.01:
        print("⚠️  Very quiet — check your microphone gain or input device.")
    elif peak > 0.9:
        print("⚠️  Very loud — possible clipping.")
    else:
        print("✅ Mic level looks good.")


def cmd_test_speaker(args):
    """Play a test tone / chirp through speakers."""
    from voice_hermes.audio import AudioPlayback

    sample_rate = 16000
    duration = args.duration or 1.0
    freq = args.freq or 440  # A4

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fade in/out to avoid clicks
    fade = np.minimum(
        np.linspace(0, 1, int(sample_rate * 0.05)),
        np.linspace(1, 0, int(sample_rate * 0.05)),
    )
    fade = np.clip(fade, 0, 1)
    pad = np.zeros(int(sample_rate * (duration - len(fade)))) if duration > 0.1 else np.array([])
    envelope = np.concatenate([fade[:len(fade)//2], pad, fade[len(fade)//2:]])

    tone = (0.3 * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)

    print(f"Playing {freq}Hz tone for {duration}s...")
    play = AudioPlayback()
    play.play(tone, sample_rate)
    play.wait()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Voice-Hermes Agent")
    parser.add_argument("--config", "-c", help="Path to config file")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Launch the orchestrator")

    dev_parser = sub.add_parser("list-devices", help="List audio input/output devices")

    mic_parser = sub.add_parser("test-mic", help="Test microphone recording")
    mic_parser.add_argument("-d", "--duration", type=float, default=3.0,
                            help="Recording duration in seconds (default: 3)")

    spk_parser = sub.add_parser("test-speaker", help="Test speaker playback")
    spk_parser.add_argument("-d", "--duration", type=float, default=1.0,
                            help="Tone duration in seconds (default: 1)")
    spk_parser.add_argument("-f", "--freq", type=float, default=440,
                            help="Tone frequency in Hz (default: 440)")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
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
