"""WebSocket connection manager for Chef Reachy."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class WebSocketManager:
    """Manage WebSocket connections and broadcast events."""

    connections: list[WebSocket] = field(default_factory=list)
    event_loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, event: dict[str, Any]):
        """Broadcast event to all connected clients."""
        if not self.connections:
            return

        disconnected = []
        for ws in self.connections:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    def broadcast_sync(self, event: dict[str, Any]):
        """Broadcast from sync context (runs in event loop)."""
        if not self.connections or self.event_loop is None:
            return

        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(event), self.event_loop)
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")
