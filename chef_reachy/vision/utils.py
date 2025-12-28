"""Utility functions for object detection visualization."""

import base64
import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Color palette for different object classes (BGR format for OpenCV)
DEFAULT_COLORS = {
    "tomato": (0, 0, 255),  # Red
    "onion": (0, 165, 255),  # Orange
    "carrot": (0, 140, 255),  # Dark Orange
    "potato": (139, 69, 19),  # Brown
    "knife": (128, 128, 128),  # Gray
    "cutting board": (92, 51, 23),  # Dark Brown
    "bowl": (255, 191, 0),  # Deep Sky Blue
    "plate": (238, 130, 238),  # Violet
    "cup": (255, 255, 0),  # Cyan
    "apple": (0, 50, 255),  # Dark Red
    "banana": (0, 255, 255),  # Yellow
    "bread": (19, 69, 139),  # Saddle Brown
    "cheese": (0, 215, 255),  # Gold
    "lettuce": (0, 255, 0),  # Green
    "pepper": (0, 128, 0),  # Dark Green
    "garlic": (245, 245, 245),  # White Smoke
}

# Default color for unknown objects
DEFAULT_COLOR = (255, 0, 255)  # Magenta


def draw_bboxes(
    image: NDArray[np.uint8],
    predictions: list[dict[str, Any]],
    colors: dict[str, tuple[int, int, int]] | None = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> NDArray[np.uint8]:
    """Draw bounding boxes on image with labels and scores.

    Args:
        image: Image in OpenCV format (BGR)
        predictions: List of detections from OWL-ViT, each with:
            - label: str
            - score: float (0-1)
            - box: dict with xmin, ymin, xmax, ymax
        colors: Optional custom color mapping {label: (B, G, R)}
        thickness: Line thickness for bounding boxes
        font_scale: Font scale for labels

    Returns:
        Annotated image (copy, original is not modified)
    """
    # Create a copy to avoid modifying original
    annotated = image.copy()

    if colors is None:
        colors = DEFAULT_COLORS

    for pred in predictions:
        label = pred["label"]
        score = pred["score"]
        box = pred["box"]

        # Get bounding box coordinates
        xmin = int(box["xmin"])
        ymin = int(box["ymin"])
        xmax = int(box["xmax"])
        ymax = int(box["ymax"])

        # Get color for this label
        color = colors.get(label, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, thickness)

        # Prepare label text
        label_text = f"{label}: {score:.2f}"

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=1
        )

        # Draw filled rectangle as background for text
        cv2.rectangle(
            annotated,
            (xmin, ymin - text_height - baseline - 5),
            (xmin + text_width + 5, ymin),
            color,
            -1,  # Filled
        )

        # Draw label text on top of background
        cv2.putText(
            annotated,
            label_text,
            (xmin + 2, ymin - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def encode_image_base64(
    image: NDArray[np.uint8], quality: int = 85
) -> str:
    """Convert CV2 image to base64 JPEG string.

    Args:
        image: Image in OpenCV format (BGR)
        quality: JPEG quality (1-100)

    Returns:
        Base64-encoded JPEG string
    """
    try:
        success, jpeg_buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        if not success:
            logger.error("Failed to encode image to JPEG")
            return ""

        image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")
        return image_base64

    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""
