"""FastAPI routes for Chef Reachy."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from chef_reachy.main import ChefReachy

logger = logging.getLogger(__name__)


def setup_routes(app: FastAPI, chef: "ChefReachy"):
    """Add custom API and WebSocket routes to the FastAPI app."""

    @app.on_event("startup")
    async def startup():
        """Set up event loop for WebSocket manager."""
        chef.ws_manager.event_loop = asyncio.get_event_loop()

    @app.get("/api/status")
    async def get_status():
        """Get current status."""
        return {
            "whisper_ready": chef.whisper_ready,
            "tts_ready": chef.tts_ready,
            "agent_ready": chef.agent_config is not None,
            "conversation_active": chef.conversation_active,
            "current_status": chef.current_status,
            "inventory_items": len(chef.inventory.get_all_items()),
        }

    @app.get("/api/inventory")
    async def get_inventory():
        """Get inventory data."""
        return chef.inventory.to_dict()

    @app.get("/api/video/frame")
    async def get_video_frame():
        """Get current camera frame as JPEG."""
        try:
            if chef.reachy_mini is not None:
                frame = chef.reachy_mini.media.get_frame()
                if frame is not None:
                    from chef_reachy.agent.tools import encode_image_base64

                    img_b64 = encode_image_base64(frame)
                    return {"image": img_b64, "timestamp": time.time()}
            return {"error": "No frame available"}
        except Exception as e:
            return {"error": str(e)}

    @app.websocket("/vision/stream")
    async def vision_stream(websocket: WebSocket):
        """WebSocket endpoint for event and video streaming."""
        await chef.ws_manager.connect(websocket)

        # Send initial status
        await websocket.send_json(
            {
                "type": "connected",
                "status": chef.current_status,
                "message": "Connected to Chef Reachy",
                "timestamp": time.time(),
            }
        )

        # Send current inventory
        items = chef.inventory.get_all_items()
        await websocket.send_json(
            {
                "type": "inventory_update",
                "items": [
                    {
                        "id": item.id,
                        "product_name": item.product_name,
                        "expiration_date": item.expiration_date,
                        "detected_at": item.detected_at.isoformat(),
                    }
                    for item in items
                ],
                "timestamp": time.time(),
            }
        )

        # Stream video frames
        async def send_video_frames():
            while True:
                try:
                    if chef.reachy_mini is not None:
                        frame = chef.reachy_mini.media.get_frame()
                        if frame is not None:
                            from chef_reachy.agent.tools import encode_image_base64

                            img_b64 = encode_image_base64(frame)
                            await websocket.send_json(
                                {
                                    "type": "video_frame",
                                    "image": img_b64,
                                    "timestamp": time.time(),
                                }
                            )
                    await asyncio.sleep(0.1)  # 10 FPS
                except Exception:
                    break

        # Start video streaming task
        video_task = asyncio.create_task(send_video_frames())

        try:
            while True:
                # Keep connection alive, handle any client messages
                await websocket.receive_text()
        except WebSocketDisconnect:
            video_task.cancel()
            chef.ws_manager.disconnect(websocket)
