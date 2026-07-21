"""
Main orchestrator — state machine that ties all components together.

States: IDLE → LISTENING → PROCESSING → QUERY_AGENT → SPEAKING → IDLE
"""

import logging

from voice_hermes.config import Config

logger = logging.getLogger("voice_hermes")


class VoiceHermesOrchestrator:
    """Main state machine orchestrating the voice pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.state = "IDLE"
        logger.info("Orchestrator initialized (state=%s)", self.state)

    async def run(self):
        """Main event loop."""
        logger.info("Starting voice-hermes event loop")
        raise NotImplementedError

    def shutdown(self):
        """Graceful shutdown."""
        self.state = "STOPPED"
        logger.info("Shutdown complete")
