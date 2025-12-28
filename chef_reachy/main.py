import logging
import threading
import time
from datetime import datetime

import numpy as np
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

from chef_reachy.vision import CameraCapture, VisionConfig, VisionProcessor

logger = logging.getLogger(__name__)


class ChefReachy(ReachyMiniApp):
    # Optional: URL to a custom configuration page for the app
    # eg. "http://localhost:8042"
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Optional: specify a media backend ("gstreamer", "default", etc.)
    request_media_backend: str | None = None

    def __init__(self):
        super().__init__()

        # Set HF_HOME BEFORE any HuggingFace imports or operations
        import os
        cache_dir = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir

        # Initialize vision model BEFORE server starts
        logger.info("=" * 60)
        logger.info("INITIALIZING VISION MODEL")
        logger.info("=" * 60)
        logger.info("This will download ~5GB on first run and may take 10-30 seconds...")

        self.vision_processor: VisionProcessor | None = None
        self.vision_ready = False
        self.vision_error: str | None = None

        try:
            vision_config = VisionConfig()
            logger.info(f"Model: {vision_config.model_path}")
            logger.info(f"Device: {vision_config.device_preference}")
            logger.info(f"Cache: {vision_config.cache_dir}")

            self.vision_processor = VisionProcessor(vision_config)

            if self.vision_processor.initialize():
                model_info = self.vision_processor.get_model_info()
                device = model_info.get("device", "unknown")
                logger.info("=" * 60)
                logger.info(f"✓ VISION MODEL READY ON {device.upper()}!")
                logger.info("=" * 60)
                self.vision_ready = True
            else:
                self.vision_error = "Failed to initialize vision model"
                logger.error("=" * 60)
                logger.error("✗ VISION INITIALIZATION FAILED")
                logger.error("=" * 60)
        except Exception as e:
            self.vision_error = str(e)
            logger.error("=" * 60)
            logger.error(f"✗ VISION ERROR: {e}")
            logger.error("=" * 60)
            logger.error("App will continue without vision capabilities")

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # Vision state
        camera_capture: CameraCapture | None = None
        last_image_description = ""
        last_image_base64 = ""
        is_processing = False

        # Initialize camera capture
        logger.info("Initializing camera capture...")
        try:
            camera_capture = CameraCapture(reachy_mini)
            logger.info("Camera capture initialized")
        except Exception as e:
            logger.error(f"Failed to initialize camera capture: {e}")

        # You can ignore this part if you don't want to add settings to your app.
        # If you set custom_app_url to None, you have to remove this part as well.
        # === vvv ===
        assert self.settings_app is not None, "settings_app must be available when custom_app_url is set"

        class ProcessRequest(BaseModel):
            prompt: str | None = None

        @self.settings_app.post("/vision/capture_and_process")
        def capture_and_process(request: ProcessRequest):
            """Capture frame and process with vision model."""
            nonlocal is_processing, last_image_description, last_image_base64

            if is_processing:
                return {
                    "status": "busy",
                    "message": "Processing already in progress",
                }

            if not self.vision_ready or self.vision_processor is None:
                return {
                    "status": "error",
                    "message": self.vision_error or "Vision model not initialized",
                }

            if camera_capture is None:
                return {
                    "status": "error",
                    "message": "Camera not available",
                }

            is_processing = True

            try:
                # Capture frame
                logger.info("Capturing frame...")
                frame = camera_capture.capture_frame()

                if frame is None:
                    is_processing = False
                    return {
                        "status": "error",
                        "message": "Failed to capture frame from camera",
                    }

                # Convert frame to base64 for UI display
                import base64
                import cv2

                success, jpeg_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    last_image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")
                else:
                    last_image_base64 = ""

                # Process with vision model
                logger.info("Processing image with vision model...")
                prompt = request.prompt if request.prompt else None
                description = self.vision_processor.process_image(frame, prompt)

                last_image_description = description
                timestamp = datetime.now().isoformat()

                logger.info(f"Vision result: {description}")

                is_processing = False
                return {
                    "status": "success",
                    "description": description,
                    "image": last_image_base64,
                    "timestamp": timestamp,
                }

            except Exception as e:
                logger.error(f"Vision processing error: {e}")
                is_processing = False
                return {
                    "status": "error",
                    "message": str(e),
                }

        # === ^^^ ===

        # Main control loop - keep robot idle
        logger.info("Chef Reachy ready! Use the web interface to capture and identify food.")

        while not stop_event.is_set():
            time.sleep(0.1)


if __name__ == "__main__":
    app = ChefReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
