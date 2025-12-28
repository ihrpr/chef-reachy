"""Configuration for audio processing."""

import os
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Configuration for text-to-speech."""

    model_path: str = "hexgrad/Kokoro-82M"
    voice: str = "af_heart"  # Default voice for Kokoro
    cache_dir: str = os.path.expanduser("~/.cache/huggingface")
    device_preference: str = "auto"  # "auto", "cuda", "cpu"
    target_sample_rate: int = 16000  # ReachyMini requires 16kHz
    chunk_size: int = 1024  # Audio chunk size for push_audio_sample

    detection_phrase: str = "I found {label}"

    def __post_init__(self):
        """Load configuration from environment variables if available."""
        self.model_path = os.getenv("TTS_MODEL", self.model_path)
        self.cache_dir = os.getenv("HF_HOME", self.cache_dir)
        self.device_preference = os.getenv("TTS_DEVICE", self.device_preference)

        if sample_rate := os.getenv("TTS_SAMPLE_RATE"):
            self.target_sample_rate = int(sample_rate)
