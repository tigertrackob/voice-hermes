"""
Command-line interface for Voice-Hermes.

Subcommands:
  start       Launch the orchestrator
  stop        Stop a running instance
  status      Check if running
  test-mic    Test microphone
  test-speaker Test speaker
  list-devices List audio devices
"""

import argparse
import sys


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


def main():
    parser = argparse.ArgumentParser(description="Voice-Hermes Agent")
    parser.add_argument("--config", "-c", help="Path to config file")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Launch the orchestrator")
    sub.add_parser("list-devices", help="List audio input/output devices")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "list-devices":
        cmd_list_devices(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
