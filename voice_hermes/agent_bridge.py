"""
Bridge between Voice-Hermes and Hermes Agent subprocess.

Spawns Hermes CLI as a subprocess with piped stdin/stdout,
sends user queries (STT output) and streams responses back (for TTS).
"""

import logging
import os
import select
import signal
import subprocess
import threading
import time
from queue import Queue, Empty
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class AgentBridge:
    """
    Manages a Hermes Agent subprocess for voice interactions.

    Pipes transcribed user queries to Hermes's stdin and reads
    its response from stdout, yielding text chunks for TTS.

    Typical usage::

        agent = AgentBridge()
        agent.start()
        agent.send("What time is it?")
        for chunk in agent.receive():
            tts.speak(chunk)
        agent.stop()
    """

    def __init__(
        self,
        command: str = "hermes",
        profile: str = "default",
        workdir: Optional[str] = None,
        startup_timeout: float = 10.0,
        response_timeout: float = 120.0,
    ):
        """
        Args:
            command: Hermes CLI command path.
            profile: Hermes profile to use.
            workdir: Working directory for the subprocess.
            startup_timeout: Max seconds to wait for Hermes to be ready.
            response_timeout: Max seconds to wait for a response.
        """
        self.command = command
        self.profile = profile
        self.workdir = workdir or os.getcwd()
        self.startup_timeout = startup_timeout
        self.response_timeout = response_timeout

        self._proc: Optional[subprocess.Popen] = None
        self._output_queue: Queue = Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Spawn Hermes as a subprocess with piped I/O."""
        if self._proc is not None:
            return

        logger.info("Starting Hermes agent (profile=%s, cmd=%s)",
                     self.profile, self.command)

        try:
            self._proc = subprocess.Popen(
                [self.command, "--profile", self.profile],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workdir,
                text=True,
                bufsize=1,  # Line-buffered
                preexec_fn=os.setsid,  # Process group for clean kill
            )
        except FileNotFoundError:
            logger.error(
                "Hermes command not found: '%s'. Install Hermes Agent first.",
                self.command,
            )
            raise

        # Start stdout reader thread
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="agent-stdout-reader",
            daemon=True,
        )
        self._reader_thread.start()

        logger.info("Hermes agent started (PID=%d)", self._proc.pid)

    def send(self, text: str):
        """Send a user query to the agent via stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Agent not started. Call start() first.")

        message = text.strip()
        if not message:
            return

        logger.info("Sending to agent: %.60s", message)
        self._proc.stdin.write(message + "\n")
        self._proc.stdin.flush()

    def receive(self) -> Generator[str, None, None]:
        """
        Yield response text chunks from the agent (blocking).

        Reads from the output queue until the agent signals completion
        or the response timeout is reached.
        """
        if self._proc is None:
            raise RuntimeError("Agent not started. Call start() first.")

        deadline = time.time() + self.response_timeout
        lines_collected = 0

        while time.time() < deadline:
            try:
                line = self._output_queue.get(timeout=0.5)
            except Empty:
                # Check if process is still alive
                if self._proc.poll() is not None:
                    logger.warning("Hermes process exited (code=%s)",
                                   self._proc.returncode)
                    break
                continue

            if line is None:  # Sentinel = end of response
                break

            line = line.strip()
            if line:
                lines_collected += 1
                yield line

        if lines_collected == 0:
            logger.warning("No response received from agent "
                           "(timeout=%.1fs)", self.response_timeout)

    def stop(self):
        """Gracefully terminate the Hermes subprocess."""
        self._stop_event.set()

        if self._proc is not None:
            pid = self._proc.pid
            # Send SIGTERM to the process group
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                self._proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    self._proc.wait(timeout=2)
                except (ProcessLookupError, PermissionError):
                    pass

            self._proc = None
            logger.info("Hermes agent stopped (PID=%d)", pid)

    def is_running(self) -> bool:
        """Check if the agent subprocess is alive."""
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reader_loop(self):
        """Read stdout from Hermes line by line in a background thread."""
        assert self._proc is not None
        assert self._proc.stdout is not None

        try:
            for line in iter(self._proc.stdout.readline, ""):
                if self._stop_event.is_set():
                    break

                line = line.rstrip("\n\r")

                # Detect Hermes prompt markers — these indicate the agent
                # is done responding and waiting for next input.
                if self._is_prompt_line(line):
                    self._output_queue.put(None)  # End sentinel
                    continue

                # Skip Hermes UI/metadata lines (timestamps, statuses)
                if self._should_skip(line):
                    continue

                # Collect stderr for logging
                self._drain_stderr()

                # Queue the actual response content
                self._output_queue.put(line)

        except (OSError, ValueError) as e:
            if not self._stop_event.is_set():
                logger.warning("Agent stdout reader error: %s", e)
        finally:
            self._output_queue.put(None)  # Ensure sentinel

    def _drain_stderr(self):
        """Non-blocking drain of stderr for logging."""
        assert self._proc is not None
        assert self._proc.stderr is not None

        try:
            while True:
                line = self._proc.stderr.readline()
                if not line:
                    break
                line = line.rstrip("\n\r")
                if line:
                    logger.debug("[agent stderr] %s", line)
        except OSError:
            pass

    @staticmethod
    def _is_prompt_line(line: str) -> bool:
        """Detect lines that indicate Hermes is awaiting input."""
        prompts = [
            ">>>",
            "> ",
            "User:",
            "You:",
            "Enter your message:",
        ]
        return any(line.startswith(p) or line == p for p in prompts)

    @staticmethod
    def _should_skip(line: str) -> bool:
        """Filter out non-content lines from Hermes output."""
        skip_prefixes = [
            "[",
            "(",
            "---",
            "INFO",
            "DEBUG",
            "WARNING",
            "ERROR",
            "Loaded",
            "Using",
            "System",
        ]
        return any(line.startswith(p) for p in skip_prefixes) or not line.strip()
