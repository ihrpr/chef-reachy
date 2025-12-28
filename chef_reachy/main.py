import asyncio
import logging
import threading
import time
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from reachy_mini import ReachyMini, ReachyMiniApp

from chef_reachy.vision import (
    CameraCapture,
    OwlVitDetector,
    VisionConfig,
    draw_bboxes,
    encode_image_base64,
)

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

        # Initialize OWL-ViT detector BEFORE server starts
        logger.info("=" * 60)
        logger.info("INITIALIZING OWL-ViT DETECTOR")
        logger.info("=" * 60)
        logger.info("This will download ~600MB on first run and may take 10-30 seconds...")

        self.detector: OwlVitDetector | None = None
        self.detector_ready = False
        self.detector_error: str | None = None

        try:
            vision_config = VisionConfig()
            logger.info(f"Model: {vision_config.model_path}")
            logger.info(f"Device: {vision_config.device_preference}")
            logger.info(f"Cache: {vision_config.cache_dir}")
            logger.info(f"Detection threshold: {vision_config.detection_threshold}")
            logger.info(f"Food labels: {', '.join(vision_config.food_labels)}")

            self.detector = OwlVitDetector(vision_config)

            if self.detector.initialize():
                model_info = self.detector.get_model_info()
                device = model_info.get("device", "unknown")
                logger.info("=" * 60)
                logger.info(f"✓ OWL-ViT DETECTOR READY ON {device.upper()}!")
                logger.info("=" * 60)
                self.detector_ready = True
            else:
                self.detector_error = "Failed to initialize OWL-ViT detector"
                logger.error("=" * 60)
                logger.error("✗ DETECTOR INITIALIZATION FAILED")
                logger.error("=" * 60)
        except Exception as e:
            self.detector_error = str(e)
            logger.error("=" * 60)
            logger.error(f"✗ DETECTOR ERROR: {e}")
            logger.error("=" * 60)
            logger.error("App will continue without detection capabilities")

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # Vision state
        camera_capture: CameraCapture | None = None

        # WebSocket connections for real-time streaming
        active_websockets: list[WebSocket] = []
        event_loop: asyncio.AbstractEventLoop | None = None

        def broadcast_to_websockets(message: dict):
            """Send message to all connected WebSocket clients (thread-safe)."""
            if not active_websockets or event_loop is None:
                return

            async def send_to_all():
                for ws in active_websockets[:]:
                    try:
                        await ws.send_json(message)
                    except Exception as e:
                        logger.warning(f"Failed to send to WebSocket: {e}")
                        if ws in active_websockets:
                            active_websockets.remove(ws)

            try:
                asyncio.run_coroutine_threadsafe(send_to_all(), event_loop)
            except Exception as e:
                logger.error(f"Failed to schedule broadcast: {e}")

        # Camera tracking state
        tracking_enabled = self.detector.vision_config.enable_tracking if self.detector else True
        last_detection_update = time.time()

        # Initialize camera capture
        logger.info("Initializing camera capture...")
        try:
            camera_capture = CameraCapture(reachy_mini)
            logger.info("Camera capture initialized")
        except Exception as e:
            logger.error(f"Failed to initialize camera capture: {e}")

        assert self.settings_app is not None, "settings_app must be available when custom_app_url is set"

        @self.settings_app.websocket("/vision/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket endpoint for real-time detection streaming."""
            nonlocal event_loop

            await websocket.accept()
            active_websockets.append(websocket)

            if event_loop is None:
                event_loop = asyncio.get_running_loop()
                logger.info("Event loop captured for WebSocket broadcasting")

            logger.info(f"WebSocket client connected. Active connections: {len(active_websockets)}")

            if self.detector_ready:
                await websocket.send_json({
                    "status": "connected",
                    "message": "Live stream connected - detection running",
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Detector not ready: {self.detector_error}",
                    "timestamp": datetime.now().isoformat(),
                })

            try:
                while True:
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in active_websockets:
                    active_websockets.remove(websocket)
                logger.info(f"WebSocket connection closed. Active connections: {len(active_websockets)}")

        if tracking_enabled:
            logger.info("Chef Reachy ready! Continuous detection and tracking enabled.")
        else:
            logger.info("Chef Reachy ready! Continuous detection enabled.")

        while not stop_event.is_set():
            # Continuous detection loop - runs regardless of whether food is detected
            if self.detector_ready and self.detector is not None and camera_capture:
                current_time = time.time()
                if current_time - last_detection_update >= self.detector.vision_config.tracking_update_rate:
                    try:
                        # Capture current frame
                        frame = camera_capture.capture_frame()
                        if frame is None:
                            logger.warning("Failed to capture frame from camera")
                            broadcast_to_websockets({
                                "status": "camera_error",
                                "message": "Camera frame capture failed",
                                "detections": [],
                                "timestamp": datetime.now().isoformat(),
                            })
                            last_detection_update = current_time
                            time.sleep(0.01)
                            continue

                        # Run detection to find hand with food
                        detections = self.detector.detect(
                            frame,
                            candidate_labels=self.detector.vision_config.food_labels,
                            threshold=self.detector.vision_config.detection_threshold,
                        )

                        if detections:
                            best_detection = max(detections, key=lambda d: d["score"])
                            box = best_detection["box"]

                            target_x = (box["xmin"] + box["xmax"]) / 2
                            target_y = (box["ymin"] + box["ymax"]) / 2

                            logger.info(
                                f"Food detected: {best_detection['label']} ({best_detection['score']:.2f}) "
                                f"at ({int(target_x)}, {int(target_y)})"
                            )

                            if tracking_enabled:
                                reachy_mini.look_at_image(
                                    int(target_x),
                                    int(target_y),
                                    duration=0.3
                                )

                            annotated_frame = draw_bboxes(frame, detections)
                            annotated_base64 = encode_image_base64(annotated_frame)

                            broadcast_to_websockets({
                                "status": "detected",
                                "detections": detections,
                                "annotated_image": annotated_base64,
                                "timestamp": datetime.now().isoformat(),
                            })

                        else:
                            logger.info("No food detected")

                            image_base64 = encode_image_base64(frame)

                            broadcast_to_websockets({
                                "status": "no_detection",
                                "detections": [],
                                "annotated_image": image_base64,
                                "timestamp": datetime.now().isoformat(),
                            })

                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                        broadcast_to_websockets({
                            "status": "error",
                            "message": str(e),
                            "detections": [],
                            "timestamp": datetime.now().isoformat(),
                        })

                    last_detection_update = current_time

            time.sleep(0.01)  # 100Hz loop rate


if __name__ == "__main__":
    app = ChefReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
