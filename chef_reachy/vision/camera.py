"""Camera capture for ReachyMini."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera frame capture from ReachyMini."""

    def __init__(self, reachy_mini: ReachyMini):
        """Initialize camera capture.

        Args:
            reachy_mini: ReachyMini instance for camera access
        """
        self.reachy_mini = reachy_mini
        self._last_capture_time: float | None = None

    def capture_frame(self) -> NDArray[np.uint8] | None:
        """Capture a frame from the camera.

        Returns:
            Image in OpenCV format (BGR) or None if capture failed
        """
        try:
            frame = self.reachy_mini.media.get_frame()

            if frame is not None:
                import time

                self._last_capture_time = time.time()
                return frame.copy()

            logger.warning("Camera returned None frame")
            return None

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None

    def get_camera_status(self) -> dict[str, Any]:
        """Get camera status information.

        Returns:
            Dictionary with camera status
        """
        return {
            "available": self.reachy_mini.media is not None,
            "last_capture_time": self._last_capture_time,
        }
