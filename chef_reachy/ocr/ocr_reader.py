"""OCR text detection using EasyOCR."""

import logging
import os
from typing import Any

import cv2
import numpy as np

from chef_reachy.ocr.config import OCRConfig

logger = logging.getLogger(__name__)


class OCRReader:
    """OCR reader using EasyOCR for text detection."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.reader = None
        self._model_loaded = False

    def initialize(self) -> bool:
        """Initialize EasyOCR reader."""
        try:
            import easyocr

            # Set cache directory
            cache_dir = os.path.expanduser(self.config.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["EASYOCR_MODULE_PATH"] = cache_dir

            # Determine device
            device = None
            if self.config.use_gpu:
                try:
                    import torch

                    if self.config.device_preference == "mps" and torch.backends.mps.is_available():
                        device = "mps"
                        logger.info("Using Apple MPS (Metal Performance Shaders) for OCR")
                    elif self.config.device_preference == "cuda" and torch.cuda.is_available():
                        device = "cuda"
                        logger.info("Using CUDA GPU for OCR")
                    else:
                        device = "cpu"
                        logger.info("GPU not available, using CPU for OCR")
                except ImportError:
                    device = "cpu"
                    logger.info("PyTorch not available, using CPU for OCR")
            else:
                device = "cpu"
                logger.info("Using CPU for OCR (GPU disabled)")

            gpu = device != "cpu"

            logger.info(f"Initializing EasyOCR with languages: {self.config.languages}")
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=gpu,
                model_storage_directory=cache_dir,
            )

            self._model_loaded = True
            logger.info("EasyOCR reader initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            return False

    def read_text(
        self,
        image: np.ndarray,
        detail: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Read text from image using EasyOCR.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            detail: Level of detail (0=simple list, 1=detailed with bounding boxes)

        Returns:
            List of detected text with metadata
        """
        if not self._model_loaded or self.reader is None:
            logger.error("OCR reader not initialized")
            return []

        try:
            # EasyOCR expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Run OCR
            results = self.reader.readtext(
                image_rgb,
                detail=detail,
                contrast_ths=self.config.contrast_threshold,
                adjust_contrast=self.config.adjust_contrast,
            )

            # Parse results based on detail level
            if detail == 0:
                # Simple list of text strings
                return [{"text": text} for text in results]
            else:
                # Detailed results with bounding boxes and confidence
                parsed_results = []
                for bbox, text, confidence in results:
                    if float(confidence) >= self.config.confidence_threshold:
                        parsed_results.append(
                            {
                                "bbox": bbox,  # List of 4 [x,y] coordinates
                                "text": text,
                                "confidence": confidence,
                            }
                        )
                return parsed_results

        except Exception as e:
            logger.error(f"OCR text detection failed: {e}")
            return []

    def get_full_text(self, image: np.ndarray) -> str:
        """
        Extract all text from image as a single string.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            Concatenated text from all detections
        """
        results = self.read_text(image, detail=1)
        texts = [r["text"] for r in results]
        return " ".join(texts)

    def is_ready(self) -> bool:
        """Check if OCR reader is ready."""
        return self._model_loaded and self.reader is not None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "languages": self.config.languages,
            "device": "gpu" if self.config.use_gpu else "cpu",
        }
