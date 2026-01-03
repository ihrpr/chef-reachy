"""Voice Activity Detection using Silero VAD."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for Silero VAD."""

    threshold: float = 0.5  # Speech probability threshold (0.0-1.0)
    min_speech_duration_ms: int = 100  # Minimum speech duration to trigger
    silence_duration_ms: int = 600  # Silence duration to end speech
    max_speech_duration_s: float = 30.0  # Maximum utterance length
    sample_rate: int = 16000  # Required by Silero VAD


class SileroVAD:
    """Wrapper for Silero VAD model."""

    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()
        self.model: Any = None
        self._initialized = False

    def initialize(self) -> bool:
        """Load Silero VAD model."""
        try:
            logger.info("Loading Silero VAD model from torch hub...")

            # Load model from torch hub (returns tuple of model, utils)
            result = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model = result[0] if isinstance(result, tuple) else result

            # Set model to evaluation mode
            self.model.eval()

            self._initialized = True
            logger.info("Silero VAD model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Process audio chunk and return speech probability.

        Args:
            audio: Audio samples as float32 numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Speech probability (0.0-1.0)
        """
        if not self._initialized or self.model is None:
            return 0.0

        try:
            # Ensure audio is float32 and 1D
            audio = np.asarray(audio, dtype=np.float32).flatten()

            # Resample if needed (Silero VAD requires 16kHz)
            if sample_rate != 16000:
                from scipy.signal import resample

                new_length = int(len(audio) * 16000 / sample_rate)
                resampled = resample(audio, new_length)
                audio = np.asarray(resampled, dtype=np.float32)

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio)

            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, 16000).item()

            return float(speech_prob)

        except Exception as e:
            logger.debug(f"VAD processing error: {e}")
            return 0.0

    def reset(self) -> None:
        """Reset VAD internal state."""
        if self.model is not None:
            try:
                self.model.reset_states()
            except Exception:
                pass

    @property
    def is_initialized(self) -> bool:
        """Check if VAD is initialized."""
        return self._initialized


class SpeechState(Enum):
    """State machine states for speech detection."""

    IDLE = "idle"
    SPEAKING = "speaking"
    SILENCE_AFTER_SPEECH = "silence_after_speech"


class SpeechBuffer:
    """
    Buffer that collects audio during speech and returns complete utterances.

    State machine:
        IDLE → (speech detected for min_duration) → SPEAKING
        SPEAKING → (silence for silence_duration) → SILENCE_AFTER_SPEECH → return audio → IDLE
        SPEAKING → (speech continues) → SPEAKING
    """

    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()

        # State
        self.state = SpeechState.IDLE
        self.audio_buffer: list[float] = []

        # Counters (in samples)
        self.speech_samples = 0
        self.silence_samples = 0

        # Thresholds (convert ms to samples)
        self.min_speech_samples = int(
            self.config.min_speech_duration_ms * self.config.sample_rate / 1000
        )
        self.silence_threshold_samples = int(
            self.config.silence_duration_ms * self.config.sample_rate / 1000
        )
        self.max_samples = int(
            self.config.max_speech_duration_s * self.config.sample_rate
        )

    def add_audio(
        self, audio: np.ndarray, speech_prob: float
    ) -> np.ndarray | None:
        """
        Add audio chunk and check if utterance is complete.

        Args:
            audio: Audio samples as numpy array
            speech_prob: Speech probability from VAD (0.0-1.0)

        Returns:
            Complete utterance as numpy array if speech ended, None otherwise
        """
        is_speech = speech_prob >= self.config.threshold
        audio_flat = np.asarray(audio, dtype=np.float32).flatten()
        num_samples = len(audio_flat)

        if self.state == SpeechState.IDLE:
            if is_speech:
                self.speech_samples += num_samples
                self.audio_buffer.extend(audio_flat)

                if self.speech_samples >= self.min_speech_samples:
                    # Enough speech detected, transition to SPEAKING
                    self.state = SpeechState.SPEAKING
            else:
                # Reset speech counter if silence in IDLE
                self.speech_samples = 0
                self.audio_buffer.clear()

        elif self.state == SpeechState.SPEAKING:
            # Always buffer audio while speaking
            self.audio_buffer.extend(audio_flat)

            if is_speech:
                # Reset silence counter, continue speaking
                self.silence_samples = 0
            else:
                # Count silence
                self.silence_samples += num_samples

                if self.silence_samples >= self.silence_threshold_samples:
                    # Enough silence, end utterance
                    return self._finalize_utterance()

            # Check max duration
            if len(self.audio_buffer) >= self.max_samples:
                return self._finalize_utterance()

        return None

    def _finalize_utterance(self) -> np.ndarray:
        """Finalize and return the buffered utterance."""
        # Get audio (trim trailing silence)
        audio = np.array(self.audio_buffer, dtype=np.float32)

        # Reset state
        self.reset()

        return audio

    def reset(self) -> None:
        """Reset buffer state."""
        self.state = SpeechState.IDLE
        self.audio_buffer.clear()
        self.speech_samples = 0
        self.silence_samples = 0

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speaking state."""
        return self.state == SpeechState.SPEAKING

    @property
    def buffered_duration(self) -> float:
        """Get current buffered audio duration in seconds."""
        return len(self.audio_buffer) / self.config.sample_rate
