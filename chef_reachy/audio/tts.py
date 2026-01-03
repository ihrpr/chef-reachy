"""Text-to-speech for Chef Reachy using Kokoro."""

import logging
import queue
import threading
import time
from typing import Any

import numpy as np
import scipy.signal
import torch
from numpy.typing import NDArray

from chef_reachy.audio.config import AudioConfig

logger = logging.getLogger(__name__)

# Minimum audio buffer before starting playback (seconds)
# This is the latency before first audio plays
STREAM_BUFFER_SECONDS = 0.2


class MeloTTSPlayer:
    """Handles text-to-speech using Kokoro and ReachyMini audio output."""

    def __init__(self, audio_config: AudioConfig | None = None):
        """Initialize the TTS player.

        Args:
            audio_config: Configuration for TTS. If None, uses defaults.
        """
        self.audio_config = audio_config or AudioConfig()
        self.model_path = self.audio_config.model_path
        self.voice = self.audio_config.voice
        self.device = self._determine_device()
        self.tts_pipeline = None
        self._initialized = False
        self.reachy_mini = None
        # Interruption support
        self._interrupted = False
        self._is_speaking = False

        # Async TTS queue for non-blocking speech
        self._speech_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._tts_stop_event = threading.Event()
        self._playback_started = False

    def _determine_device(self) -> str:
        """Determine the best available device for model inference.

        Returns:
            Device string: "cuda:0" or "cpu"
        """
        pref = self.audio_config.device_preference

        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        # auto: prefer cuda, else cpu
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """Load Kokoro TTS model onto the selected device.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Force HuggingFace to use local cache only (no network requests)
            # This prevents slow HTTP HEAD requests on every TTS call
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            # Enable MPS (Metal) GPU acceleration on Apple Silicon
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            from kokoro import KPipeline

            logger.info(f"Loading Kokoro TTS: {self.model_path}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Voice: {self.voice}")

            start_time = time.time()

            # Create Kokoro pipeline with 'a' for American English
            self.tts_pipeline = KPipeline(lang_code="a")

            # Preload the voice by doing a dummy synthesis
            # This warms up the model and caches any lazy-loaded components
            for _ in self.tts_pipeline("Hello.", voice=self.voice):
                pass  # Just iterate to trigger loading

            load_time = time.time() - start_time
            logger.info(f"✓ Kokoro TTS loaded in {load_time:.2f}s on {self.device}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            self._initialized = False
            return False

    def set_reachy_mini(self, reachy_mini: Any) -> None:
        """Set ReachyMini instance for audio playback.

        Args:
            reachy_mini: ReachyMini robot instance
        """
        self.reachy_mini = reachy_mini

    def start_async_worker(self) -> None:
        """Start the background TTS worker thread."""
        if self._tts_thread is not None and self._tts_thread.is_alive():
            return

        self._tts_stop_event.clear()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def stop_async_worker(self) -> None:
        """Stop the background TTS worker thread."""
        self._tts_stop_event.set()
        # Send sentinel to unblock queue.get()
        self._speech_queue.put(None)
        if self._tts_thread is not None:
            self._tts_thread.join(timeout=2.0)
            self._tts_thread = None

    def _tts_worker(self) -> None:
        """Background worker that processes TTS queue."""
        idle_count = 0

        while not self._tts_stop_event.is_set():
            try:
                # Check for interrupt before blocking on queue
                if self._interrupted:
                    self._clear_queue()
                    self._stop_playback_session()
                    self._interrupted = False
                    idle_count = 0
                    continue

                # Get text from queue with timeout for interrupt checking
                text = self._speech_queue.get(timeout=0.1)

                if text is None:  # Sentinel value
                    continue

                # Check again after getting item (interrupt may have occurred while waiting)
                if self._interrupted:
                    self._clear_queue()
                    self._stop_playback_session()
                    self._interrupted = False
                    continue

                # Start playback session if not already started
                self._start_playback_session()
                idle_count = 0

                # Synthesize and play (non-blocking - audio continues while we process next)
                self._speak_sync(text)

            except queue.Empty:
                # Stop playback after being idle for a while (no more queued speech)
                idle_count += 1
                if idle_count > 20 and self._playback_started:  # ~2 seconds idle
                    self._stop_playback_session()
                    idle_count = 0
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

        # Clean up on exit
        self._stop_playback_session()

    def _start_playback_session(self) -> None:
        """Start audio playback session if not already started."""
        if not self._playback_started and self.reachy_mini:
            self.reachy_mini.media.start_playing()
            self._playback_started = True

    def _stop_playback_session(self) -> None:
        """Stop audio playback session if started."""
        if self._playback_started and self.reachy_mini:
            # Small delay to let remaining audio play out
            time.sleep(0.1)
            self.reachy_mini.media.stop_playing()
            self._playback_started = False

    def _clear_queue(self) -> None:
        """Clear all pending speech from the queue."""
        while True:
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is currently playing audio."""
        return self._is_speaking

    def interrupt(self) -> None:
        """Interrupt current speech playback and clear queue."""
        self._interrupted = True
        self._clear_queue()
        # Give the worker time to notice the interrupt
        # The worker will reset _interrupted after clearing

    @property
    def queue_size(self) -> int:
        """Get current number of pending speech items."""
        return self._speech_queue.qsize()

    def speak(self, text: str) -> bool:
        """Queue text for non-blocking speech synthesis.

        Text is added to a queue and processed by the background TTS worker.
        Returns immediately without blocking.

        Args:
            text: Text to speak

        Returns:
            True if queued successfully
        """
        if not self._initialized or self.tts_pipeline is None:
            logger.error("TTS not initialized")
            return False

        self._speech_queue.put(text)
        return True

    def speak_sync(self, text: str) -> bool:
        """Synchronous speech - blocks until complete. Use speak() for non-blocking."""
        return self._speak_sync(text)

    def _speak_sync(self, text: str) -> bool:
        """Internal synchronous speech implementation.

        Uses streaming to start playback as soon as enough audio is generated,
        rather than waiting for the entire audio to be generated first.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized or self.tts_pipeline is None:
            logger.error("TTS not initialized")
            return False

        if self.reachy_mini is None:
            logger.error("ReachyMini not set")
            return False

        try:
            start_time = time.time()

            # Kokoro outputs at 24kHz
            source_rate = 24000
            device_rate = self.reachy_mini.media.get_output_audio_samplerate()
            chunk_size = self.audio_config.chunk_size

            # Buffer for accumulating audio before resampling
            # We buffer a minimum amount to avoid resampling artifacts
            min_buffer_samples = int(source_rate * STREAM_BUFFER_SECONDS)

            self._is_speaking = True

            # Buffer for accumulating raw audio from Kokoro
            raw_buffer = []
            raw_buffer_samples = 0

            try:
                # Generate speech using Kokoro pipeline (streaming)
                generator = self.tts_pipeline(text, voice=self.voice)

                for _, _, audio in generator:
                    if self._interrupted:
                        break

                    # Skip if no audio generated for this chunk
                    if audio is None:
                        continue

                    # Convert to numpy if needed (Kokoro returns torch tensors)
                    audio_np: NDArray[np.float32]
                    if isinstance(audio, np.ndarray):
                        audio_np = audio
                    elif hasattr(audio, "numpy"):  # torch.Tensor
                        audio_np = audio.numpy()  # type: ignore[union-attr]
                    else:
                        continue

                    # Add to buffer
                    raw_buffer.append(audio_np)
                    raw_buffer_samples += len(audio_np)

                    # When we have enough buffered, process and play
                    if raw_buffer_samples >= min_buffer_samples:
                        # Concatenate buffer
                        buffered_audio = np.concatenate(raw_buffer)
                        raw_buffer = []
                        raw_buffer_samples = 0

                        # Convert and play
                        converted = self._convert_audio_chunk(buffered_audio, source_rate, device_rate)
                        self._push_audio_chunks(converted, chunk_size)

                # Process any remaining audio in buffer
                if raw_buffer and not self._interrupted:
                    buffered_audio = np.concatenate(raw_buffer)
                    converted = self._convert_audio_chunk(buffered_audio, source_rate, device_rate)
                    self._push_audio_chunks(converted, chunk_size)

                return True

            finally:
                # Don't stop playback - let the worker manage the session
                self._is_speaking = False

        except Exception as e:
            logger.error(f"Speech generation/playback failed: {e}", exc_info=True)
            self._is_speaking = False
            return False

    def _convert_audio_chunk(
        self,
        audio_data: NDArray[np.float32],
        source_rate: int,
        target_rate: int,
    ) -> NDArray[np.float32]:
        """Convert a chunk of audio for streaming playback.

        Args:
            audio_data: Raw audio chunk from TTS
            source_rate: Source sample rate (e.g., 24000)
            target_rate: Target sample rate (e.g., 16000)

        Returns:
            Converted audio chunk
        """
        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1).astype(np.float32)

        # Resample if needed
        if source_rate != target_rate:
            num_samples = int(len(audio_data) * (target_rate / source_rate))
            resampled = scipy.signal.resample(audio_data, num_samples)
            audio_data = np.asarray(resampled, dtype=np.float32)

        return audio_data

    def _push_audio_chunks(self, audio_data: NDArray[np.float32], chunk_size: int) -> None:
        """Push audio data to ReachyMini in chunks.

        Args:
            audio_data: Converted audio data
            chunk_size: Size of each chunk to push
        """
        assert self.reachy_mini is not None, "ReachyMini must be set"
        for i in range(0, len(audio_data), chunk_size):
            if self._interrupted:
                break
            chunk = audio_data[i : i + chunk_size]
            self.reachy_mini.media.push_audio_sample(chunk)

    def _wait_for_playback(self, duration: float) -> None:
        """Wait for audio playback with interruption support.

        Args:
            duration: Expected playback duration in seconds
        """
        buffer_time = 0.3  # Extra time for audio system latency
        total_wait = duration + buffer_time
        check_interval = 0.1

        elapsed = 0.0
        while elapsed < total_wait:
            if self._interrupted:
                break
            time.sleep(min(check_interval, total_wait - elapsed))
            elapsed += check_interval

    def speak_detection(self, label: str) -> bool:
        """Speak about a detected object.

        Args:
            label: Detection label (e.g., "hand holding food")

        Returns:
            True if successful
        """
        phrase = self.audio_config.detection_phrase.format(label=label)
        return self.speak(phrase)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info: dict[str, Any] = {
            "initialized": self._initialized,
            "device": self.device,
            "model_path": self.model_path,
            "voice": self.voice,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        else:
            info["gpu_memory_gb"] = "N/A"

        return info

    def cleanup(self) -> None:
        """Unload model from memory and free resources."""
        try:
            # Stop the async worker first
            self.stop_async_worker()

            if self.tts_pipeline is not None:
                del self.tts_pipeline
                self.tts_pipeline = None

            # Clean up GPU memory if using CUDA
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            self._initialized = False
            logger.info("Kokoro TTS player cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
