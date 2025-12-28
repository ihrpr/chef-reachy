"""Text-to-speech for Chef Reachy using Kokoro."""

import logging
import time
from typing import Any

import numpy as np
import scipy.signal
import torch
from numpy.typing import NDArray

from chef_reachy.audio.config import AudioConfig

logger = logging.getLogger(__name__)


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
            logger.info("Kokoro TTS already initialized")
            return True

        try:
            from kokoro import KPipeline

            logger.info(f"Loading Kokoro TTS: {self.model_path}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Voice: {self.voice}")

            start_time = time.time()

            # Create Kokoro pipeline with 'a' for American English
            self.tts_pipeline = KPipeline(lang_code='a')

            load_time = time.time() - start_time
            logger.info(f"✓ Kokoro TTS loaded in {load_time:.2f}s")
            logger.info(f"Device: {self.device}")

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
        logger.info("ReachyMini reference set for TTS player")

    def speak(self, text: str) -> bool:
        """Generate speech and play through ReachyMini.

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
            logger.info(f"Generating speech: '{text}'")
            start_time = time.time()

            # Generate speech using Kokoro pipeline
            # The pipeline returns a generator yielding (gs, ps, audio) tuples
            generator = self.tts_pipeline(text, voice=self.voice)

            # Collect all audio chunks
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)

            # Concatenate all chunks
            if not audio_chunks:
                logger.error("No audio generated")
                return False

            audio_data = np.concatenate(audio_chunks)

            gen_time = time.time() - start_time
            logger.info(f"Speech generated in {gen_time:.2f}s")

            # Kokoro outputs at 24kHz, we need to convert to ReachyMini's format
            source_rate = 24000

            # Convert audio format for ReachyMini
            audio_data = self._convert_audio_format(audio_data, source_rate)

            # Play audio through ReachyMini
            self._play_audio(audio_data)

            return True

        except Exception as e:
            logger.error(f"Speech generation/playback failed: {e}", exc_info=True)
            return False

    def _convert_audio_format(
        self, audio_data: NDArray[np.float32], source_rate: int
    ) -> NDArray[np.float32]:
        """Convert TTS output to ReachyMini format.

        Args:
            audio_data: Raw audio from TTS pipeline
            source_rate: Source sample rate

        Returns:
            Converted audio (float32, mono, target sample rate)
        """
        assert self.reachy_mini is not None, "ReachyMini must be set"

        # Step 1: Ensure float32
        if audio_data.dtype != np.float32:
            logger.debug(f"Converting {audio_data.dtype} to float32")
            audio_data = audio_data.astype(np.float32)

        # Step 2: Convert stereo to mono if needed
        if audio_data.ndim > 1:
            logger.debug(f"Converting {audio_data.shape[1]} channels to mono")
            audio_data = np.mean(audio_data, axis=1).astype(np.float32)

        # Step 3: Resample if needed
        device_rate = self.reachy_mini.media.get_output_audio_samplerate()
        if source_rate != device_rate:
            logger.info(f"Resampling from {source_rate}Hz to {device_rate}Hz")
            num_samples = int(len(audio_data) * (device_rate / source_rate))
            resampled = scipy.signal.resample(audio_data, num_samples)
            audio_data = resampled.astype(np.float32)  # Resample may change dtype

        return audio_data

    def _play_audio(self, audio_data: NDArray[np.float32]) -> None:
        """Play audio through ReachyMini speaker using chunked playback.

        Args:
            audio_data: Audio samples (float32, mono, correct sample rate)
        """
        sample_rate = self.reachy_mini.media.get_output_audio_samplerate()
        chunk_size = self.audio_config.chunk_size

        logger.info(
            f"Playing audio: {len(audio_data)} samples @ {sample_rate}Hz"
        )

        try:
            self.reachy_mini.media.start_playing()

            # Push all chunks
            num_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                self.reachy_mini.media.push_audio_sample(chunk)
                chunk_idx = i // chunk_size + 1
                logger.debug(f"Pushed chunk {chunk_idx}/{num_chunks}")

            # Wait for playback to complete
            duration = len(audio_data) / sample_rate
            buffer_time = 0.5  # Extra time for audio system latency
            logger.info(f"Waiting {duration:.2f}s for playback...")
            time.sleep(duration + buffer_time)

        finally:
            self.reachy_mini.media.stop_playing()
            logger.info("Playback complete")

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
            info["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory // (1024**3)
            )
        else:
            info["gpu_memory_gb"] = "N/A"

        return info

    def cleanup(self) -> None:
        """Unload model from memory and free resources."""
        try:
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
