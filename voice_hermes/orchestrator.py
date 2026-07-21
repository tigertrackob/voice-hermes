"""
Main orchestrator — state machine that ties all components together.

States::

    IDLE ──[wake word]──> LISTENING
    LISTENING──[silence]──> PROCESSING
    PROCESSING ──[STT done]──> QUERY_AGENT
    QUERY_AGENT──[response]──> SPEAKING
    SPEAKING ──[TTS done]──> IDLE
    SPEAKING ──[interrupt]──> 🛑 IDLE
    Any state ──[shutdown]──> STOPPED
"""

import asyncio
import logging
import signal
import sys
import time
from enum import Enum, auto

from voice_hermes.audio import AudioCapture
from voice_hermes.config import Config
from voice_hermes.interrupt import InterruptionDetector
from voice_hermes.stt import STTEngine
from voice_hermes.tts import TTSEngine
from voice_hermes.wake_word import WakeWordDetector, WakeWordEvent

logger = logging.getLogger("voice_hermes.orchestrator")


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    QUERY_AGENT = auto()
    SPEAKING = auto()
    STOPPED = auto()


class VoiceHermesOrchestrator:
    """
    Main state machine orchestrating the voice pipeline.

    Usage::

        orch = VoiceHermesOrchestrator(config)
        asyncio.run(orch.run())
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.state = State.IDLE
        self._wake_word: WakeWordDetector | None = None
        self._stt: STTEngine | None = None
        self._tts: TTSEngine | None = None
        self._agent: 'AgentBridge' | None = None  # type: ignore
        self._interrupt: InterruptionDetector | None = None
        self._capture: AudioCapture | None = None

        # Track last interaction for logging
        self._interaction_count = 0

        # asyncio primitives
        self._shutdown_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._interrupt_event = asyncio.Event()
        self._interrupt_message = ""

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on config."""
        level = getattr(logging, self.cfg.general.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            stream=sys.stdout,
        )
        if self.cfg.general.debug:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self):
        """Main event loop — start components and listen for events."""
        logger.info("Voice-Hermes starting...")

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self._handle_shutdown()))

        try:
            await self._init_components()
            await self._event_loop()
        except Exception as e:
            logger.exception("Fatal error in main loop: %s", e)
        finally:
            await self._shutdown()

    async def _init_components(self):
        """Initialize all sub-components."""
        cfg = self.cfg

        # Wake word detector
        self._wake_word = WakeWordDetector(
            wake_words=cfg.wake_word.wake_words,
            model_path=cfg.wake_word.model_path,
            sensitivity=cfg.wake_word.sensitivity,
            frame_length=cfg.wake_word.frame_length,
            input_device=cfg.audio.input_device or None,
        )
        self._wake_word.on_detected = self._on_wake_word

        # STT
        self._stt = STTEngine(
            model_path=cfg.stt.model_path,
            model_size=cfg.stt.model_size,
            language=cfg.stt.language,
            sample_rate=cfg.stt.sample_rate,
        )

        # TTS
        self._tts = TTSEngine(
            model_path=cfg.tts.model_path,
            config_path=cfg.tts.config_path,
            output_device=cfg.audio.output_device or None,
        )

        # Agent bridge
        from voice_hermes.agent_bridge import AgentBridge
        self._agent = AgentBridge(
            command=cfg.agent.hermes_command,
            profile=cfg.agent.hermes_profile,
        )

        # Interruption detector
        self._interrupt = InterruptionDetector(
            stop_words=cfg.interrupt.stop_words,
            enabled=cfg.interrupt.enabled,
            cooldown=cfg.interrupt.cooldown,
            sample_rate=cfg.audio.sample_rate,
            input_device=cfg.audio.input_device or None,
        )
        self._interrupt.on_interrupt = self._on_interruption

        # Audio capture
        self._capture = AudioCapture(
            sample_rate=cfg.audio.sample_rate,
            channels=cfg.audio.channels,
            blocksize=cfg.audio.blocksize,
            device=cfg.audio.input_device or None,
            vad_mode=cfg.stt.vad_mode,
        )

        # Start always-on components
        self._wake_word.start()
        self._interrupt.start()
        self._agent.start()

        logger.info("All components initialized (state=%s)", self.state.name)

    async def _event_loop(self):
        """Wait for events and handle state transitions."""
        logger.info("Entering main event loop (state=%s)", self.state.name)

        while self.state is not State.STOPPED:
            if self.state == State.IDLE:
                await self._handle_idle()
            elif self.state == State.LISTENING:
                await self._handle_listening()
            elif self.state == State.PROCESSING:
                await self._handle_processing()
            elif self.state == State.QUERY_AGENT:
                await self._handle_query_agent()
            elif self.state == State.SPEAKING:
                await self._handle_speaking()
            else:
                await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _handle_idle(self):
        """IDLE: wait for wake word or shutdown."""
        logger.debug("State=IDLE — waiting for wake word")

        # Wait for either wake event or shutdown
        wake_task = asyncio.create_task(self._wait_for_wake())
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())

        done, pending = await asyncio.wait(
            [wake_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if self._shutdown_event.is_set():
            self.state = State.STOPPED
            return

        self._interaction_count += 1
        logger.info("Wake word detected → entering LISTENING "
                    "(interaction #%d)", self._interaction_count)

        # Play a short beep/indication — Piper is too slow for this,
        # so use a short sine beep via sounddevice
        await self._play_attention_sound()
        self.state = State.LISTENING

    async def _handle_listening(self):
        """LISTENING: capture audio until silence."""
        logger.info("State=LISTENING — listening for speech...")

        try:
            self._capture.start()
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                self._capture.record_until_silence,
                self.cfg.stt.silence_timeout * 10,  # max timeout as multiple
                self.cfg.stt.silence_timeout,
                0.2,  # min duration
            )
            self._capture.stop()
        except Exception as e:
            logger.error("Capture error: %s", e)
            self.state = State.IDLE
            return

        if len(audio) < 1600:  # < 100ms of audio
            logger.info("Too short/no audio detected — returning to IDLE")
            self.state = State.IDLE
            return

        self._current_audio = audio
        self.state = State.PROCESSING

    async def _handle_processing(self):
        """PROCESSING: transcribe audio via STT."""
        audio = getattr(self, '_current_audio', None)
        if audio is None:
            self.state = State.IDLE
            return

        logger.info("State=PROCESSING — transcribing (%.1fs)...",
                    len(audio) / self.cfg.stt.sample_rate)

        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None,
                self._stt.transcribe,
                audio,
                self.cfg.stt.sample_rate,
            )
            self._current_audio = None  # Free memory
        except Exception as e:
            logger.error("STT error: %s", e)
            text = ""

        if not text.strip():
            logger.info("No speech recognized — returning to IDLE")
            self.state = State.IDLE
            return

        self._current_text = text
        logger.info("Transcribed: %s", text)
        self.state = State.QUERY_AGENT

    async def _handle_query_agent(self):
        """QUERY_AGENT: send text to Hermes, stream response."""
        text = getattr(self, '_current_text', None)
        if text is None:
            self.state = State.IDLE
            return

        logger.info("State=QUERY_AGENT — sending to Hermes...")
        self._current_text = None  # Free memory

        try:
            # Send query to agent
            await asyncio.get_event_loop().run_in_executor(
                None, self._agent.send, text,
            )

            # Stream response directly to TTS
            self._tts.speak("")  # Ensure TTS thread is alive
            self._interrupt.activate()

            chunks = []
            for chunk in self._agent.receive():
                chunks.append(chunk)
                self._tts.speak(chunk)

            full_response = " ".join(chunks)
            logger.info("Agent response: %.120s", full_response)

            # Store for reference
            self._last_response = full_response

        except Exception as e:
            logger.error("Agent error: %s", e)

        self.state = State.SPEAKING

    async def _handle_speaking(self):
        """SPEAKING: wait for TTS to finish or interrupt."""
        logger.info("State=SPEAKING — waiting for TTS completion")

        # Wait for either TTS to finish or interrupt
        while self._tts.is_busy() and self.state is not State.STOPPED:
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                self._tts.stop()
                self._interrupt.deactivate()
                logger.info("Interrupted by user — returning to IDLE")
                self.state = State.IDLE
                return
            await asyncio.sleep(0.1)

        self._interrupt.deactivate()
        logger.info("TTS finished — returning to IDLE")
        self.state = State.IDLE

    # ------------------------------------------------------------------
    # Event callbacks (called from worker threads)
    # ------------------------------------------------------------------

    def _on_wake_word(self, event: WakeWordEvent):
        """Called by WakeWordDetector thread when wake word is heard."""
        if self.state == State.IDLE:
            self._wake_event.set()

    def _on_interruption(self):
        """Called by InterruptionDetector thread when stop word is heard."""
        if self.state == State.SPEAKING:
            self._interrupt_event.set()

    async def _wait_for_wake(self):
        """Coroutine that waits for the wake event."""
        while not self._shutdown_event.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                return
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _play_attention_sound(self):
        """Play a short beep to indicate the system is listening."""
        try:
            import numpy as np
            import sounddevice as sd

            sample_rate = 16000
            duration = 0.15
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Short 800Hz beep with fade in/out
            fade = np.minimum(
                np.linspace(0, 1, int(sample_rate * 0.02)),
                np.linspace(1, 0, int(sample_rate * 0.02)),
            )
            fade = np.clip(fade, 0, 1)
            pad = np.maximum(0, len(t) - len(fade))
            envelope = np.concatenate([fade[:len(fade)//2], np.ones(pad), fade[len(fade)//2:]])
            beep = (0.3 * np.sin(2 * np.pi * 800 * t) * envelope[:len(t)]).astype(np.float32)
            sd.play(beep, samplerate=sample_rate)
        except Exception as e:
            logger.debug("Could not play attention sound: %s", e)

    async def _handle_shutdown(self):
        """Signal graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()

    async def _shutdown(self):
        """Graceful shutdown of all components."""
        self.state = State.STOPPED
        logger.info("Shutting down...")

        if self._interrupt:
            self._interrupt.stop()
        if self._wake_word:
            self._wake_word.stop()
        if self._tts:
            self._tts.stop()
        if self._agent:
            self._agent.stop()
        if self._capture:
            self._capture.stop()

        logger.info("Shutdown complete")
