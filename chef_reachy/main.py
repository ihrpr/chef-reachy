import asyncio
import logging
import threading
import time
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from reachy_mini import ReachyMini, ReachyMiniApp

from chef_reachy.audio import (
    AudioConfig,
    MeloTTSPlayer,
)
from chef_reachy.inventory import (
    FoodItem,
    InventoryManager,
)
from chef_reachy.llm import (
    LLMConfig,
    OllamaClient,
)
from chef_reachy.ocr import (
    OCRConfig,
    OCRReader,
)
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

        # Initialize Kokoro TTS player BEFORE server starts
        logger.info("=" * 60)
        logger.info("INITIALIZING KOKORO-82M TEXT-TO-SPEECH")
        logger.info("=" * 60)
        logger.info("This will download model on first run (82MB)...")

        self.tts_player: MeloTTSPlayer | None = None
        self.tts_ready = False
        self.tts_error: str | None = None

        try:
            audio_config = AudioConfig()
            logger.info(f"Model: {audio_config.model_path}")
            logger.info(f"Device: {audio_config.device_preference}")
            logger.info(f"Sample rate: {audio_config.target_sample_rate}Hz")

            self.tts_player = MeloTTSPlayer(audio_config)

            if self.tts_player.initialize():
                model_info = self.tts_player.get_model_info()
                device = model_info.get("device", "unknown")
                logger.info("=" * 60)
                logger.info(f"✓ KOKORO-82M TTS READY ON {device.upper()}!")
                logger.info("=" * 60)
                self.tts_ready = True
            else:
                self.tts_error = "Failed to initialize Kokoro-82M TTS"
                logger.error("=" * 60)
                logger.error("✗ TTS INITIALIZATION FAILED")
                logger.error("=" * 60)
        except Exception as e:
            self.tts_error = str(e)
            logger.error("=" * 60)
            logger.error(f"✗ TTS ERROR: {e}")
            logger.error("=" * 60)
            logger.error("App will continue without TTS capabilities")

        # Initialize OCR reader BEFORE server starts
        logger.info("=" * 60)
        logger.info("INITIALIZING EASYOCR TEXT DETECTION")
        logger.info("=" * 60)
        logger.info("This will download model on first run (~80MB)...")

        self.ocr_reader: OCRReader | None = None
        self.ocr_ready = False
        self.ocr_error: str | None = None

        try:
            ocr_config = OCRConfig()
            logger.info(f"Languages: {ocr_config.languages}")
            logger.info(f"Device: {ocr_config.device_preference}")
            logger.info(f"Confidence threshold: {ocr_config.confidence_threshold}")

            self.ocr_reader = OCRReader(ocr_config)

            if self.ocr_reader.initialize():
                model_info = self.ocr_reader.get_model_info()
                logger.info("=" * 60)
                logger.info("✓ EASYOCR READER READY!")
                logger.info("=" * 60)
                self.ocr_ready = True
            else:
                self.ocr_error = "Failed to initialize EasyOCR reader"
                logger.error("=" * 60)
                logger.error("✗ OCR INITIALIZATION FAILED")
                logger.error("=" * 60)
        except Exception as e:
            self.ocr_error = str(e)
            logger.error("=" * 60)
            logger.error(f"✗ OCR ERROR: {e}")
            logger.error("=" * 60)
            logger.error("App will continue without OCR capabilities")

        # Initialize LLM client BEFORE server starts
        logger.info("=" * 60)
        logger.info("INITIALIZING OLLAMA LLM CLIENT")
        logger.info("=" * 60)
        logger.info("Connecting to Ollama server...")

        self.llm_client: OllamaClient | None = None
        self.llm_ready = False
        self.llm_error: str | None = None

        try:
            llm_config = LLMConfig()
            logger.info(f"Model: {llm_config.model_name}")
            logger.info(f"Host: {llm_config.host}")

            self.llm_client = OllamaClient(llm_config)

            if self.llm_client.initialize():
                model_info = self.llm_client.get_model_info()
                logger.info("=" * 60)
                logger.info("✓ OLLAMA LLM CLIENT READY!")
                logger.info("=" * 60)
                self.llm_ready = True
            else:
                self.llm_error = "Failed to initialize Ollama client"
                logger.error("=" * 60)
                logger.error("✗ LLM INITIALIZATION FAILED")
                logger.error("=" * 60)
        except Exception as e:
            self.llm_error = str(e)
            logger.error("=" * 60)
            logger.error(f"✗ LLM ERROR: {e}")
            logger.error("=" * 60)
            logger.error("App will continue without LLM capabilities")

        # Initialize inventory manager (in-memory only)
        logger.info("=" * 60)
        logger.info("INITIALIZING INVENTORY MANAGER")
        logger.info("=" * 60)

        self.inventory = InventoryManager(storage_path=None)
        logger.info("✓ INVENTORY MANAGER READY (in-memory)")
        logger.info("=" * 60)

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

        # Set ReachyMini reference for TTS player
        if self.tts_ready and self.tts_player:
            self.tts_player.set_reachy_mini(reachy_mini)
            logger.info("TTS player connected to ReachyMini audio")

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
                    except TimeoutError:
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

                            # Process OCR and LLM if enabled
                            ocr_result = None
                            product_info = None
                            if (
                                self.ocr_ready
                                and self.ocr_reader is not None
                                and self.llm_ready
                                and self.llm_client is not None
                            ):
                                try:
                                    # Crop detected region with more padding to capture text
                                    import cv2

                                    height, width = frame.shape[:2]
                                    padding = 50  # Increased padding to capture more text
                                    x1 = max(0, int(box["xmin"]) - padding)
                                    y1 = max(0, int(box["ymin"]) - padding)
                                    x2 = min(width, int(box["xmax"]) + padding)
                                    y2 = min(height, int(box["ymax"]) + padding)

                                    cropped = frame[y1:y2, x1:x2]

                                    # Save cropped image for debugging
                                    debug_path = "/tmp/chef_reachy_ocr_debug.jpg"
                                    cv2.imwrite(debug_path, cropped)
                                    logger.info(f"Saved cropped region to {debug_path} (size: {cropped.shape})")

                                    # Run OCR on cropped region
                                    logger.info("Running OCR on detected food item...")
                                    ocr_results = self.ocr_reader.read_text(cropped, detail=1)
                                    logger.info(f"OCR found {len(ocr_results)} text regions (before filtering)")

                                    # Log all detections for debugging
                                    for result in ocr_results:
                                        logger.info(f"  Text: '{result['text']}' (confidence: {result['confidence']:.2f})")

                                    ocr_text = self.ocr_reader.get_full_text(cropped)

                                    if ocr_text.strip():
                                        logger.info(f"OCR text: {ocr_text}")

                                        # Use LLM to extract product info
                                        logger.info("Extracting product info with LLM...")
                                        product_info = self.llm_client.extract_product_info(ocr_text)

                                        if product_info.get("product_name"):
                                            # Add to inventory
                                            item = FoodItem(
                                                product_name=product_info["product_name"],
                                                expiration_date=product_info.get("expiration_date"),
                                                ocr_text=ocr_text,
                                                confidence=best_detection["score"],
                                            )
                                            self.inventory.add_item(item)

                                            logger.info(
                                                f"Added to inventory: {item.product_name} "
                                                f"(expires: {item.expiration_date})"
                                            )

                                            # Speak about the new item
                                            if self.tts_ready and self.tts_player:
                                                try:
                                                    msg = f"Added {item.product_name} to inventory"
                                                    if item.expiration_date:
                                                        msg += f" expires on {item.expiration_date}"
                                                    self.tts_player.speak_detection(msg)
                                                except Exception as e:
                                                    logger.error(f"TTS playback error: {e}")
                                        else:
                                            logger.info("Could not extract product name from OCR text")
                                    else:
                                        logger.info("No text detected in the food item")

                                except Exception as e:
                                    logger.error(f"OCR/LLM processing error: {e}")

                            # Get current inventory as serializable dict list
                            inventory_items = [
                                {
                                    "id": item.id,
                                    "product_name": item.product_name,
                                    "expiration_date": item.expiration_date,
                                    "detected_at": item.detected_at.isoformat(),
                                    "confidence": item.confidence,
                                }
                                for item in self.inventory.get_all_items()
                            ]

                            broadcast_to_websockets({
                                "status": "detected",
                                "detections": detections,
                                "annotated_image": annotated_base64,
                                "timestamp": datetime.now().isoformat(),
                                "ocr_result": ocr_result,
                                "product_info": product_info,
                                "inventory": inventory_items,
                            })

                            # Speak about the detection (if OCR/LLM didn't speak)
                            if self.tts_ready and self.tts_player and not product_info:
                                try:
                                    self.tts_player.speak_detection(best_detection["label"])
                                except Exception as e:
                                    logger.error(f"TTS playback error: {e}")

                        else:
                            logger.info("No food detected")

                            image_base64 = encode_image_base64(frame)

                            # Get current inventory as serializable dict list
                            inventory_items = [
                                {
                                    "id": item.id,
                                    "product_name": item.product_name,
                                    "expiration_date": item.expiration_date,
                                    "detected_at": item.detected_at.isoformat(),
                                    "confidence": item.confidence,
                                }
                                for item in self.inventory.get_all_items()
                            ]

                            broadcast_to_websockets({
                                "status": "no_detection",
                                "detections": [],
                                "annotated_image": image_base64,
                                "timestamp": datetime.now().isoformat(),
                                "inventory": inventory_items,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    app = ChefReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
