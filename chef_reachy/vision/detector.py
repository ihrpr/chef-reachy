"""OWL-ViT object detection for food items."""

import logging
import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import pipeline

from chef_reachy.vision.config import VisionConfig

logger = logging.getLogger(__name__)


class OwlVitDetector:
    """Handles OWL-ViT zero-shot object detection."""

    def __init__(self, vision_config: VisionConfig | None = None):
        """Initialize the detector.

        Args:
            vision_config: Configuration for detection. If None, uses defaults.
        """
        self.vision_config = vision_config or VisionConfig()
        self.model_path = self.vision_config.model_path
        self.device = self._determine_device()
        self.detector = None
        self._initialized = False

    def _determine_device(self) -> str:
        """Determine the best available device for model inference.

        Returns:
            Device string: "cuda:0" or "cpu"
        """
        pref = self.vision_config.device_preference

        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        # auto: prefer cuda, else cpu (MPS not recommended for OWL-ViT)
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """Load OWL-ViT detector onto the selected device.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.info("OWL-ViT detector already initialized")
            return True

        try:
            logger.info(f"Loading OWL-ViT detector: {self.model_path}")
            logger.info(f"Device: {self.device}")

            start_time = time.time()

            # Create zero-shot object detection pipeline
            self.detector = pipeline(
                "zero-shot-object-detection",
                model=self.model_path,
                device=self.device if self.device.startswith("cuda") else -1,
            )

            load_time = time.time() - start_time
            logger.info(f"✓ OWL-ViT detector loaded in {load_time:.2f}s")
            logger.info(f"Device: {self.device}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OWL-ViT detector: {e}")
            self._initialized = False
            return False

    def detect(
        self,
        cv2_image: NDArray[np.uint8],
        candidate_labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Detect objects in image using zero-shot detection.

        Args:
            cv2_image: Image in OpenCV format (BGR)
            candidate_labels: List of object labels to detect. If None, uses config defaults.
            threshold: Confidence threshold (0-1). If None, uses config default.

        Returns:
            List of detections, each with:
                - label: str
                - score: float (0-1)
                - box: dict with xmin, ymin, xmax, ymax (int)
        """
        if not self._initialized or self.detector is None:
            logger.error("Detector not initialized")
            return []

        if candidate_labels is None:
            candidate_labels = self.vision_config.food_labels

        if threshold is None:
            threshold = self.vision_config.detection_threshold

        try:
            # Convert BGR to RGB
            image_rgb = cv2_image[:, :, ::-1]

            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Run detection
            logger.info(f"Running detection with {len(candidate_labels)} labels...")
            logger.info(f"Labels: {', '.join(candidate_labels)}")

            start_time = time.time()

            predictions = self.detector(
                pil_image,
                candidate_labels=candidate_labels,
            )

            inference_time = time.time() - start_time
            logger.info(f"Detection completed in {inference_time:.2f}s")

            # Filter by threshold
            filtered = [p for p in predictions if p["score"] >= threshold]

            logger.info(
                f"Found {len(filtered)} detections above threshold {threshold:.2f}"
            )
            for det in filtered:
                logger.info(
                    f"  - {det['label']}: {det['score']:.2f} at "
                    f"({det['box']['xmin']}, {det['box']['ymin']}) - "
                    f"({det['box']['xmax']}, {det['box']['ymax']})"
                )

            return filtered

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info: dict[str, Any] = {
            "initialized": self._initialized,
            "device": self.device,
            "model_path": self.model_path,
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
            if self.detector is not None:
                del self.detector
                self.detector = None

            # Clean up GPU memory if using CUDA
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            self._initialized = False
            logger.info("OWL-ViT detector cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
