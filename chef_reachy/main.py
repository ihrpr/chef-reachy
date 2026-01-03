"""Chef Reachy - Voice-activated food inventory assistant using Claude Agent SDK."""

import asyncio
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import NamedTuple

import numpy as np
from dotenv import load_dotenv
from reachy_mini import ReachyMini, ReachyMiniApp

from chef_reachy.agent import AgentConfig, ConversationManager
from chef_reachy.audio import (
    AudioConfig,
    MeloTTSPlayer,
    SileroVAD,
    SpeechBuffer,
    VADConfig,
    WhisperConfig,
    WhisperSTT,
)
from chef_reachy.inventory import InventoryManager
from chef_reachy.server import WebSocketManager, setup_routes

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Utterance(NamedTuple):
    """Represents a complete speech utterance."""

    audio: np.ndarray
    timestamp: float


class ChefReachy(ReachyMiniApp):
    """Voice-activated inventory assistant with Claude Agent SDK."""

    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = "default"

    def __init__(self):
        super().__init__()

        logger.info("=" * 60)
        logger.info("INITIALIZING CHEF REACHY")
        logger.info("=" * 60)

        # Initialize Whisper STT
        logger.info("Loading Whisper speech-to-text...")
        whisper_config = WhisperConfig(
            model_size="base", device="cpu", compute_type="int8"
        )
        self.whisper = WhisperSTT(whisper_config)
        self.whisper_ready = self.whisper.initialize()

        if self.whisper_ready:
            logger.info("✓ Whisper STT ready")
        else:
            logger.error("✗ Whisper STT failed to initialize")

        # Initialize Silero VAD
        logger.info("Loading Silero VAD...")
        self.vad_config = VADConfig(
            threshold=0.5,
            min_speech_duration_ms=100,
            silence_duration_ms=600,
            max_speech_duration_s=30.0,
        )
        self.vad = SileroVAD(self.vad_config)
        self.vad_ready = self.vad.initialize()

        if self.vad_ready:
            logger.info("✓ Silero VAD ready")
        else:
            logger.error("✗ Silero VAD failed to initialize")

        # Initialize speech buffer
        self.speech_buffer = SpeechBuffer(self.vad_config)

        # Initialize TTS
        logger.info("Loading Kokoro-82M text-to-speech...")
        audio_config = AudioConfig()
        self.tts_player = MeloTTSPlayer(audio_config)
        self.tts_ready = self.tts_player.initialize()

        if self.tts_ready:
            logger.info("✓ Kokoro-82M TTS ready")
        else:
            logger.error("✗ Kokoro-82M TTS failed to initialize")

        # Initialize inventory manager
        storage_path = os.path.expanduser("~/.chef_reachy/inventory.json")
        self.inventory = InventoryManager(storage_path=storage_path)
        logger.info(
            f"✓ Inventory manager ready ({len(self.inventory.get_all_items())} items)"
        )

        # Initialize Claude Agent config
        try:
            self.agent_config = AgentConfig()
            logger.info("✓ Claude Agent config ready")
        except ValueError as e:
            logger.error(f"✗ Claude Agent config failed: {e}")
            logger.error("Please set ANTHROPIC_API_KEY environment variable")

        # WebSocket manager for event broadcasting
        self.ws_manager = WebSocketManager()

        # Current status for UI
        self.current_status = "idle"

        # Conversation manager (initialized in run())
        self.conversation: ConversationManager | None = None

        # Persistent event loop for agent operations (created in run())
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._agent_thread: threading.Thread | None = None

        # Audio settings
        self.sample_rate = 16000

        # VAD audio buffer (Silero VAD needs 512 samples at 16kHz)
        self._vad_buffer: list[float] = []
        self._vad_chunk_size = 512

        # Track speech status for UI
        self._last_speech_status = False

        # Track if agent is currently processing
        self._agent_processing = False

        # Track last timeout check time for periodic checking
        self._last_timeout_check = 0.0

        # Reachy mini reference (set in run())
        self.reachy_mini: ReachyMini | None = None

        # Background audio capture thread
        self._audio_thread: threading.Thread | None = None
        self._audio_stop_event = threading.Event()

        # Queue for passing utterances from audio thread to main thread
        self._utterance_queue: queue.Queue[Utterance] = queue.Queue()

        # Callback for when speech starts (for TTS interruption)
        self._on_speech_start: list[Callable[[], None]] = []

        # Add custom routes to settings_app
        if self.settings_app is not None:
            setup_routes(self.settings_app, self)
        else:
            logger.warning("No settings_app available, skipping route setup")

        logger.info("=" * 60)
        logger.info("✓ CHEF REACHY READY")
        logger.info("=" * 60)

    @property
    def conversation_active(self) -> bool:
        """Check if conversation is active."""
        return self.conversation is not None and self.conversation.active

    def _set_status(self, status: str):
        """Update current status."""
        self.current_status = status

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Main loop with voice-activated conversation."""
        # Store reachy_mini reference
        self.reachy_mini = reachy_mini

        # Set reachy_mini reference for TTS and start async worker
        if self.tts_ready and self.tts_player:
            self.tts_player.set_reachy_mini(reachy_mini)
            self.tts_player.start_async_worker()

        # Start audio recording and playback
        logger.info("Starting audio recording and playback...")
        reachy_mini.media.start_recording()
        reachy_mini.media.start_playing()
        time.sleep(1)  # Give pipelines time to start

        # Get actual sample rate from the robot
        self.sample_rate = reachy_mini.media.get_input_audio_samplerate()
        logger.info(f"Audio sample rate: {self.sample_rate} Hz")

        # Set up conversation manager
        if self.agent_config is None:
            logger.error("Cannot run without Claude Agent config")
            return

        # Start dedicated asyncio event loop in background thread
        # This keeps ClaudeSDKClient connected across multiple interactions
        self._start_agent_loop()

        self.conversation = ConversationManager(
            config=self.agent_config,
            inventory=self.inventory,
            broadcaster=self.ws_manager.broadcast_sync,
            tts_player=self.tts_player if self.tts_ready else None,
            status_callback=self._set_status,
        )
        self.conversation.setup(reachy_mini)

        # Register TTS interruption callback
        if self.tts_ready and self.tts_player:
            self._on_speech_start.append(self.tts_player.interrupt)

        # Start background audio capture thread
        self._start_audio_thread(reachy_mini)

        # Run the main conversation loop (processes utterances from queue)
        self._main_loop(stop_event)

        # Cleanup
        self._cleanup(reachy_mini)

    def _start_agent_loop(self):
        """Start a dedicated asyncio event loop in a background thread."""
        loop = asyncio.new_event_loop()
        self._agent_loop = loop

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._agent_thread = threading.Thread(target=run_loop, daemon=True)
        self._agent_thread.start()
        logger.info("Started dedicated asyncio event loop for agent operations")

    def _stop_agent_loop(self):
        """Stop the dedicated asyncio event loop."""
        if self._agent_loop is not None:
            self._agent_loop.call_soon_threadsafe(self._agent_loop.stop)
            if self._agent_thread is not None:
                self._agent_thread.join(timeout=2.0)
            self._agent_loop = None
            self._agent_thread = None
            logger.info("Stopped asyncio event loop")

    def _start_audio_thread(self, reachy_mini: ReachyMini):
        """Start background audio capture thread."""
        self._audio_stop_event.clear()

        def audio_worker():
            self._audio_capture_loop(reachy_mini)

        self._audio_thread = threading.Thread(target=audio_worker, daemon=True)
        self._audio_thread.start()
        logger.info("Started background audio capture thread")

    def _stop_audio_thread(self):
        """Stop background audio capture thread."""
        self._audio_stop_event.set()
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None
        logger.info("Stopped audio capture thread")

    def _run_async(self, coro) -> Future:
        """
        Run an async coroutine in the dedicated event loop.

        This keeps the ClaudeSDKClient connection alive across calls.
        Returns a Future that can be waited on.
        """
        if self._agent_loop is None:
            raise RuntimeError("Agent event loop not started")
        return asyncio.run_coroutine_threadsafe(coro, self._agent_loop)

    def _run_async_wait(self, coro, timeout: float | None = 120.0):
        """
        Run an async coroutine and wait for the result.

        Args:
            coro: The coroutine to run
            timeout: Max time to wait (None for no timeout, default 120s for agent ops)
        """
        future = self._run_async(coro)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error(f"Async operation timed out after {timeout}s")
            # Cancel the future to prevent orphaned operations
            future.cancel()
            # Reset conversation state on timeout
            if self.conversation:
                self.conversation.active = False
                self.conversation._client = None
            raise
        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            # Reset conversation state on error
            if self.conversation:
                self.conversation.active = False
            raise

    def _audio_capture_loop(self, reachy_mini: ReachyMini):
        """Background thread: continuously capture audio and detect utterances."""
        logger.info("Starting background audio capture...")

        while not self._audio_stop_event.is_set():
            current_time = time.time()

            try:
                # Get audio sample from Reachy
                audio_sample = reachy_mini.media.get_audio_sample()

                if audio_sample is not None and len(audio_sample) > 0:
                    # Process audio through VAD pipeline
                    utterance_audio = self._process_vad(audio_sample, current_time)

                    if utterance_audio is not None:
                        # Put utterance in queue for main thread to process
                        self._utterance_queue.put(
                            Utterance(utterance_audio, current_time)
                        )

            except Exception as e:
                logger.error(f"Error in audio capture: {e}")

            # Minimal sleep to avoid busy loop
            time.sleep(0.001)

        logger.info("Audio capture loop stopped")

    def _main_loop(self, stop_event: threading.Event):
        """Main thread: process utterances from queue and handle conversation."""
        logger.info("Starting main conversation loop...")
        self.current_status = "idle"

        while not stop_event.is_set():
            current_time = time.time()

            try:
                # Check for new utterances (non-blocking with timeout)
                try:
                    utterance = self._utterance_queue.get(timeout=0.1)
                    self._handle_utterance(utterance.audio, utterance.timestamp)
                except queue.Empty:
                    pass

                # Periodic timeout check (every 5 seconds)
                if current_time - self._last_timeout_check >= 5.0:
                    self._last_timeout_check = current_time
                    self._check_conversation_timeout(current_time)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback

                traceback.print_exc()

        logger.info("Main loop stopped")

    def _process_vad(
        self, audio_sample: np.ndarray | bytes, current_time: float
    ) -> np.ndarray | None:
        """Process audio through VAD pipeline and return complete utterance if detected."""
        # Convert to numpy array if needed
        if isinstance(audio_sample, np.ndarray):
            audio_np = audio_sample
        else:
            audio_np = np.frombuffer(audio_sample, dtype=np.float32)

        # Convert stereo to mono
        if len(audio_np.shape) == 2 and audio_np.shape[1] == 2:
            mono_audio = audio_np.mean(axis=1).astype(np.float32)
        else:
            mono_audio = audio_np.flatten().astype(np.float32)

        # Add to VAD buffer
        self._vad_buffer.extend(mono_audio.tolist())

        # Run VAD when we have enough samples (512 for Silero VAD)
        utterance: np.ndarray | None = None
        while self.vad_ready and len(self._vad_buffer) >= self._vad_chunk_size:
            # Get 512 samples for VAD
            vad_audio = np.array(
                self._vad_buffer[: self._vad_chunk_size], dtype=np.float32
            )
            self._vad_buffer = self._vad_buffer[self._vad_chunk_size :]

            speech_prob = self.vad.process(vad_audio, self.sample_rate)

            # Add audio to speech buffer
            result = self.speech_buffer.add_audio(vad_audio, speech_prob)
            if result is not None:
                utterance = result

            # Broadcast speech status change to UI
            is_speaking = self.speech_buffer.is_speaking
            if is_speaking != self._last_speech_status:
                self._last_speech_status = is_speaking
                self.ws_manager.broadcast_sync(
                    {
                        "type": "speech_status",
                        "is_speaking": is_speaking,
                        "timestamp": current_time,
                    }
                )
                if is_speaking:
                    self.current_status = "speaking"
                    logger.info("Speech detected - buffering...")
                    # Reset conversation timeout when user STARTS speaking
                    # This prevents timeout during long user utterances
                    if self.conversation_active and self.conversation:
                        self.conversation.last_interaction_time = current_time
                    # Call speech start callbacks (e.g., to interrupt TTS)
                    for callback in self._on_speech_start:
                        try:
                            callback()
                        except Exception:
                            pass

        return utterance

    def _handle_utterance(self, utterance: np.ndarray, current_time: float):
        """Handle a complete utterance - transcribe and process."""
        duration = len(utterance) / self.sample_rate
        logger.info(f"Utterance complete: {duration:.2f}s, transcribing...")

        self.current_status = "transcribing"
        self.ws_manager.broadcast_sync(
            {
                "type": "status",
                "status": "transcribing",
                "message": "Transcribing speech...",
                "timestamp": current_time,
            }
        )

        if not (self.whisper_ready and self.whisper):
            return

        text = self.whisper.transcribe(utterance, self.sample_rate)
        logger.info(f"Whisper: '{text}'")

        # Broadcast whisper transcription to UI (for debugging)
        self.ws_manager.broadcast_sync(
            {
                "type": "whisper_transcription",
                "text": text,
                "timestamp": current_time,
            }
        )

        if not text:
            self.vad.reset()
            return

        text_lower = text.lower()

        # Check for stop phrase
        if self.conversation_active and self._is_stop_phrase(text_lower):
            self._end_conversation(current_time)

        # Continue active conversation
        elif self.conversation_active:
            if self.conversation and self.conversation.is_timed_out():
                self._timeout_conversation(current_time)
            else:
                self._continue_conversation(text, current_time)

        # Check for wake word to start new conversation
        elif self.whisper.detect_wake_word(text, self.agent_config.name):
            self._start_conversation(text, current_time)

        else:
            pass  # No active conversation and no wake word

        # Reset VAD state for next utterance
        self.vad.reset()

    def _check_conversation_timeout(self, current_time: float):
        """Periodically check if conversation has timed out (user walked away)."""
        if not self._agent_processing and self.conversation_active:
            if self.conversation and self.conversation.is_timed_out():
                logger.info("Periodic check: conversation timed out (user inactive)")
                self._timeout_conversation(current_time)

    def _clear_audio_buffers(self):
        """Clear audio buffers after agent processing to prevent stale audio/echo."""
        # Clear VAD buffer
        self._vad_buffer.clear()
        # Reset speech buffer state
        self.speech_buffer.reset()
        # Reset VAD state
        if self.vad_ready:
            self.vad.reset()
        # Reset speech status tracking
        self._last_speech_status = False
        # Drain the utterance queue (discard any utterances from TTS echo)
        drained = 0
        while not self._utterance_queue.empty():
            try:
                self._utterance_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break

    def _is_stop_phrase(self, text_lower: str) -> bool:
        """Check if text contains a stop phrase."""
        stop_phrases = [
            "stop",
            f"that's it {self.agent_config.name.lower()}",
            f"goodbye {self.agent_config.name.lower()}",
            f"bye {self.agent_config.name.lower()}",
            f"thank you {self.agent_config.name.lower()}",
            f"thanks {self.agent_config.name.lower()}",
            # Generic ending phrases
            "end conversation",
            "stop listening",
            "never mind",
            "nevermind",
        ]
        return any(phrase in text_lower for phrase in stop_phrases)

    def _start_conversation(self, text: str, current_time: float):
        """Start a new conversation."""
        if self.conversation is None:
            return

        logger.info(
            f"Wake word '{self.agent_config.name}' detected - starting new conversation!"
        )
        self.current_status = "listening"

        self.ws_manager.broadcast_sync(
            {
                "type": "status",
                "status": "listening",
                "message": "Hi! How can I help?",
                "timestamp": current_time,
            }
        )

        self.ws_manager.broadcast_sync(
            {
                "type": "user_speech",
                "text": text,
                "timestamp": current_time,
            }
        )

        # Start session AND process first message in dedicated event loop
        # Using _run_async_wait keeps the ClaudeSDKClient alive across calls
        self._agent_processing = True
        try:
            self._run_async_wait(self.conversation.start_and_process(text))
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
        finally:
            self._agent_processing = False
            # Clear audio buffers to prevent processing stale audio (including TTS echo)
            self._clear_audio_buffers()

    def _continue_conversation(self, text: str, current_time: float):
        """Continue an active conversation."""
        if self.conversation is None:
            return

        logger.info(f"Continuing conversation: '{text}'")

        self.ws_manager.broadcast_sync(
            {
                "type": "user_speech",
                "text": text,
                "timestamp": current_time,
            }
        )

        # Process in dedicated event loop to keep ClaudeSDKClient alive
        self._agent_processing = True
        try:
            self._run_async_wait(self.conversation.process(text))
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
        finally:
            self._agent_processing = False
            # Clear audio buffers to prevent processing stale audio (including TTS echo)
            self._clear_audio_buffers()

    def _end_conversation(self, current_time: float):
        """End the conversation with goodbye."""
        if self.conversation is None:
            return

        logger.info("Stop phrase detected - ending conversation")
        self.current_status = "idle"

        # End in dedicated event loop
        self._run_async_wait(self.conversation.end(), timeout=10.0)

        self.ws_manager.broadcast_sync(
            {
                "type": "status",
                "status": "idle",
                "message": "Conversation ended",
                "timestamp": current_time,
            }
        )

        if self.tts_ready and self.tts_player:
            self.tts_player.speak("Goodbye!")

    def _timeout_conversation(self, current_time: float):
        """End the conversation due to timeout."""
        if self.conversation is None:
            return

        logger.info("Conversation timed out")
        self.current_status = "idle"

        # End in dedicated event loop
        self._run_async_wait(self.conversation.end(), timeout=10.0)

        self.ws_manager.broadcast_sync(
            {
                "type": "status",
                "status": "idle",
                "message": "Conversation timed out",
                "timestamp": current_time,
            }
        )

    def _cleanup(self, reachy_mini: ReachyMini):
        """Clean up resources on shutdown."""
        logger.info("Stopping...")

        # Stop audio capture thread first
        self._stop_audio_thread()

        # Stop TTS worker
        if self.tts_ready and self.tts_player:
            self.tts_player.stop_async_worker()

        # Close conversation if active
        if self.conversation and self.conversation.active:
            try:
                self._run_async_wait(self.conversation.end(), timeout=5.0)
            except Exception as e:
                logger.debug(f"Error closing conversation: {e}")

        # Stop the dedicated asyncio event loop
        self._stop_agent_loop()

        try:
            reachy_mini.media.stop_recording()
            reachy_mini.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping audio: {e}")

        logger.info("Stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = ChefReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
