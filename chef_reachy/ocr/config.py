"""OCR configuration for EasyOCR."""

from dataclasses import dataclass


@dataclass
class OCRConfig:
    """Configuration for EasyOCR text detection."""

    # Languages to detect (English by default, can add more)
    languages: list[str] | None = None

    # Device preference: 'mps' for Apple Silicon, 'cuda' for NVIDIA GPU, 'cpu' for CPU
    device_preference: str = "mps"

    # Use GPU if available
    use_gpu: bool = True

    # Recognition confidence threshold (0.0-1.0)
    confidence_threshold: float = 0.3

    # Text detection parameters
    contrast_threshold: float = 0.4
    adjust_contrast: float = 0.5

    # Cache directory for model downloads
    cache_dir: str = "~/.cache/easyocr"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]
