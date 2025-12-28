"""Configuration for vision processing."""

import os
from dataclasses import dataclass


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    model_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    cache_dir: str = os.path.expanduser("~/.cache/huggingface")
    max_new_tokens: int = 64
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "cpu"  # "auto", "cuda", "mps", "cpu" - CPU is safer for 8GB RAM systems
    default_prompt: str = "Identify the food items in this image. List what you see."

    def __post_init__(self):
        """Load configuration from environment variables if available."""
        self.model_path = os.getenv("LOCAL_VISION_MODEL", self.model_path)
        self.cache_dir = os.getenv("HF_HOME", self.cache_dir)
        self.device_preference = os.getenv("VISION_DEVICE", self.device_preference)

        # Convert string values to appropriate types
        if max_tokens := os.getenv("VISION_MAX_TOKENS"):
            self.max_new_tokens = int(max_tokens)
        if jpeg_quality := os.getenv("VISION_JPEG_QUALITY"):
            self.jpeg_quality = int(jpeg_quality)
