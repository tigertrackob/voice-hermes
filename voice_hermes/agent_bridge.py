"""
Bridge between Voice-Hermes and Hermes Agent subprocess.

Spawns Hermes as a subprocess, pipes STT output to it,
and streams agent response back for TTS.
"""


class AgentBridge:
    """Manages Hermes Agent subprocess for voice interactions."""

    def __init__(self, command: str = "hermes", profile: str = "default"):
        self.command = command
        self.profile = profile
        self._proc = None

    def start(self):
        """Spawn Hermes as a subprocess."""
        raise NotImplementedError

    def send(self, text: str):
        """Send user query to the agent."""
        raise NotImplementedError

    def receive(self):
        """Yield response text chunks from the agent."""
        raise NotImplementedError

    def stop(self):
        """Gracefully terminate the agent subprocess."""
        raise NotImplementedError
