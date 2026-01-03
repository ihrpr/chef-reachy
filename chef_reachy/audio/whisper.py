"""Local Whisper speech-to-text using faster-whisper."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    """Configuration for Whisper STT."""

    model_size: str = "base"  # tiny, base, small, medium, large
    device: str = "cpu"  # cpu or cuda
    compute_type: str = "int8"  # int8, float16, float32
    language: str = "en"


class WhisperSTT:
    """Speech-to-text using local Whisper model."""

    def __init__(self, config: WhisperConfig | None = None):
        self.config = config or WhisperConfig()
        self.model = None
        self._initialized = False

    def initialize(self) -> bool:
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper {self.config.model_size} model...")
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._initialized = True
            logger.info("Whisper model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32)
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text
        """
        if not self._initialized or self.model is None:
            logger.error("Whisper model not initialized")
            return ""

        try:
            # faster-whisper expects float32 audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio to [-1, 1] range if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()

            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                vad_parameters={"threshold": 0.5},
            )

            # Combine all segments
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def detect_wake_word(self, text: str, wake_word: str = "claude") -> bool:
        """
        Check if wake word is present in transcribed text.

        Args:
            text: Transcribed text
            wake_word: Wake word to detect (default: "claude")

        Returns:
            True if wake word detected
        """
        import re

        text_lower = text.lower().strip()
        wake_word_lower = wake_word.lower()

        # Check if wake word appears at start
        if text_lower.startswith(wake_word_lower):
            return True

        # Remove punctuation and check for wake word as a word
        text_clean = re.sub(r"[^\w\s]", "", text_lower)
        words = text_clean.split()
        return wake_word_lower in words

    def is_ready(self) -> bool:
        """Check if model is ready."""
        return self._initialized and self.model is not None
