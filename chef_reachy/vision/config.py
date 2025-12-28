"""Configuration for vision processing."""

import os
from dataclasses import dataclass, field


@dataclass
class VisionConfig:
    """Configuration for OWL-ViT object detection."""

    model_path: str = "google/owlvit-base-patch32"
    cache_dir: str = os.path.expanduser("~/.cache/huggingface")
    detection_threshold: float = 0.15
    jpeg_quality: int = 85
    device_preference: str = "auto"  # "auto", "cuda", "cpu"

    # Labels for detecting hand holding food
    food_labels: list[str] = field(
        default_factory=lambda: [
            "hand holding food",
            "hand with food",
            "person hand holding object",
            "hand holding object",
            "human hand with food",
            "food packaging",
            "cookies",
            "hand",  # fallback
        ]
    )

    # Camera tracking settings
    enable_tracking: bool = True
    tracking_kp: float = 1.0  # Proportional gain for camera tracking
    tracking_update_rate: float = (
        2.5  # Update rate in seconds (~0.4 Hz, matches OWL-ViT inference time)
    )
    max_rotation_deg: float = 30.0  # Maximum rotation angle in degrees

    def __post_init__(self):
        """Load configuration from environment variables if available."""
        self.model_path = os.getenv("VISION_MODEL", self.model_path)
        self.cache_dir = os.getenv("HF_HOME", self.cache_dir)
        self.device_preference = os.getenv("VISION_DEVICE", self.device_preference)

        # Convert string values to appropriate types
        if threshold := os.getenv("VISION_DETECTION_THRESHOLD"):
            self.detection_threshold = float(threshold)
        if jpeg_quality := os.getenv("VISION_JPEG_QUALITY"):
            self.jpeg_quality = int(jpeg_quality)

        # Parse food labels from environment
        if food_labels_str := os.getenv("FOOD_LABELS"):
            self.food_labels = [label.strip() for label in food_labels_str.split(",")]

        # Parse tracking settings from environment
        if enable_tracking := os.getenv("ENABLE_TRACKING"):
            self.enable_tracking = enable_tracking.lower() in ("true", "1", "yes")
        if tracking_kp := os.getenv("TRACKING_KP"):
            self.tracking_kp = float(tracking_kp)
        if tracking_update_rate := os.getenv("TRACKING_UPDATE_RATE"):
            self.tracking_update_rate = float(tracking_update_rate)
        if max_rotation_deg := os.getenv("MAX_ROTATION_DEG"):
            self.max_rotation_deg = float(max_rotation_deg)
