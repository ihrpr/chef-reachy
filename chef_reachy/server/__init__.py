"""Server module for Chef Reachy."""

from chef_reachy.server.routes import setup_routes
from chef_reachy.server.websocket import WebSocketManager

__all__ = ["WebSocketManager", "setup_routes"]
