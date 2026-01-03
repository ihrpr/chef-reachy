"""Conversation manager for Claude Agent SDK."""

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import StreamEvent
from reachy_mini import ReachyMini

from chef_reachy.agent.config import AgentConfig
from chef_reachy.agent.tools import create_chef_tools
from chef_reachy.audio import MeloTTSPlayer
from chef_reachy.inventory import InventoryManager

logger = logging.getLogger(__name__)

# Regex pattern for splitting text into speakable chunks
# Matches sentence-ending punctuation, commas followed by space, or colons
CHUNK_PATTERN = re.compile(r"(?<=[.!?,:])\s+")

# Minimum characters before sending to TTS (avoids tiny fragments)
# Lower = faster response, but may sound choppy with very short phrases
MIN_CHUNK_SIZE = 5

# Type alias for broadcaster and status callback
Broadcaster = Callable[[dict[str, Any]], None]
StatusCallback = Callable[[str], None]


class ConversationManager:
    """Manages Claude Agent conversations with persistent session."""

    def __init__(
        self,
        config: AgentConfig,
        inventory: InventoryManager,
        broadcaster: Broadcaster,
        tts_player: MeloTTSPlayer | None = None,
        status_callback: StatusCallback | None = None,
    ):
        """
        Initialize conversation manager.

        Args:
            config: Agent configuration
            inventory: Inventory manager for tools
            broadcaster: Function to broadcast WebSocket events
            tts_player: Optional TTS player for speaking responses
            status_callback: Optional callback to update status
        """
        self.config = config
        self.inventory = inventory
        self.broadcaster = broadcaster
        self.tts_player = tts_player
        self.status_callback = status_callback

        # Conversation state
        self.active = False
        self.last_interaction_time = 0.0
        self.timeout = 30.0  # End conversation after 30s of silence

        # Claude Agent client (persistent across turns)
        self._client: ClaudeSDKClient | None = None
        self._options: ClaudeAgentOptions | None = None

        # Reachy Mini reference (set when starting)
        self._reachy_mini: ReachyMini | None = None

    def setup(self, reachy_mini: ReachyMini):
        """
        Set up agent options with tools.

        Args:
            reachy_mini: Reachy Mini instance for camera/media access
        """
        self._reachy_mini = reachy_mini

        # Create tools with broadcaster for event streaming
        tools = create_chef_tools(
            reachy_mini,
            self.inventory,
            self.config.api_key,
            broadcaster=self.broadcaster,
        )
        mcp_server = create_sdk_mcp_server(
            name="chef-reachy", version="0.3.0", tools=tools
        )

        self._options = ClaudeAgentOptions(
            mcp_servers={"chef": mcp_server},
            allowed_tools=[
                "mcp__chef__scan_food_item",
                "mcp__chef__get_inventory",
                "mcp__chef__remove_item",
                "mcp__chef__clear_inventory",
            ],
            system_prompt=self.config.system_prompt,
            model=self.config.model,
            # Enable streaming for real-time text deltas
            include_partial_messages=True,
        )

        logger.info("Conversation manager setup complete")

    def _set_status(self, status: str):
        """Update status via callback if available."""
        if self.status_callback:
            self.status_callback(status)

    async def start(self):
        """Start a new conversation session."""
        # Close existing client if any
        await self.end()

        if self._options is None:
            raise RuntimeError("ConversationManager not set up. Call setup() first.")

        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()

        self.active = True
        self.last_interaction_time = time.time()

        logger.info("Started new conversation session")

    async def end(self):
        """End the current conversation session."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.debug(f"Error closing agent client: {e}")
            self._client = None

        self.active = False
        logger.info("Ended conversation session")

    async def start_and_process(self, user_text: str):
        """
        Start a new conversation and process the first message.

        This combines start() and process() in a single async context
        to avoid event loop issues with asyncio.run().

        Args:
            user_text: User's first message text
        """
        try:
            await self.start()
            await self._send_and_receive(user_text)
        except Exception as e:
            self._handle_error(e)

    async def process(self, user_text: str):
        """
        Process user input and generate response.

        Args:
            user_text: User's message text
        """
        try:
            # Ensure we have an active client
            if self._client is None:
                await self.start()

            await self._send_and_receive(user_text)

        except Exception as e:
            self._handle_error(e)

    async def _send_and_receive(self, user_text: str):
        """
        Send message to Claude and receive streaming response.

        Uses Claude's streaming API to receive text deltas in real-time,
        streaming them to UI immediately and queueing TTS asynchronously.

        Args:
            user_text: User's message text
        """
        # Update status
        self._set_status("processing")
        self.broadcaster(
            {
                "type": "status",
                "status": "processing",
                "message": "Processing your request...",
                "timestamp": time.time(),
            }
        )

        # Send user query to persistent client
        client = self._client
        if client is None:
            raise RuntimeError("Client not initialized")

        await client.query(user_text)

        # Buffer for accumulating streamed text for TTS chunking
        tts_buffer = ""
        full_response = ""

        # Receive streaming response
        async for message in client.receive_response():
            # Check for interruption
            if self.tts_player and self.tts_player._interrupted:
                break

            # Handle streaming text deltas
            if isinstance(message, StreamEvent):
                event = message.event
                event_type = event.get("type", "")

                # Handle content_block_delta events with text deltas
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_delta = delta.get("text", "")
                        if text_delta:
                            tts_buffer += text_delta
                            full_response += text_delta

                            # Broadcast delta to websockets IMMEDIATELY for real-time UI
                            self.broadcaster(
                                {
                                    "type": "agent_response_delta",
                                    "delta": text_delta,
                                    "timestamp": time.time(),
                                }
                            )

                            # Queue complete chunks for TTS (non-blocking)
                            if self.tts_player:
                                tts_buffer = self._queue_speakable_chunks(tts_buffer)

            # Handle complete assistant messages (final)
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Broadcast complete response
                        self.broadcaster(
                            {
                                "type": "agent_response",
                                "text": block.text,
                                "timestamp": time.time(),
                            }
                        )

                    elif isinstance(block, ToolUseBlock):
                        # Queue any buffered text before tool execution
                        if tts_buffer.strip() and self.tts_player:
                            self.tts_player.speak(tts_buffer.strip())
                            tts_buffer = ""

                        # Broadcast tool execution start
                        self._set_status("executing_tool")
                        self.broadcaster(
                            {
                                "type": "tool_start",
                                "tool_name": block.name,
                                "timestamp": time.time(),
                            }
                        )

        # Queue any remaining text in the buffer (non-blocking)
        if tts_buffer.strip() and self.tts_player:
            self.tts_player.speak(tts_buffer.strip())

        # Update status back to listening
        self._set_status("listening")
        self.last_interaction_time = time.time()
        self.broadcaster(
            {
                "type": "status",
                "status": "listening",
                "message": "Listening...",
                "timestamp": time.time(),
            }
        )

    def _queue_speakable_chunks(self, text_buffer: str) -> str:
        """
        Extract speakable chunks from buffer and queue for TTS (non-blocking).

        Args:
            text_buffer: Accumulated text that may contain complete chunks

        Returns:
            Remaining text that doesn't form a complete chunk yet
        """
        if not self.tts_player:
            return text_buffer

        # Find chunk boundaries (sentences, clauses at commas/colons)
        chunks = CHUNK_PATTERN.split(text_buffer)

        if len(chunks) <= 1:
            return text_buffer

        # Queue all complete chunks (all but the last fragment)
        for chunk in chunks[:-1]:
            chunk = chunk.strip()
            # Only queue chunks that are substantial enough
            if chunk and len(chunk) >= MIN_CHUNK_SIZE:
                self.tts_player.speak(chunk)

                # Check if interrupted - stop processing more chunks
                if self.tts_player._interrupted:
                    return ""

        # Return the incomplete last part
        return chunks[-1]

    def _handle_error(self, e: Exception):
        """Handle conversation error."""
        logger.error(f"Error in conversation: {e}")
        import traceback

        traceback.print_exc()

        self._set_status("idle")
        self.broadcaster(
            {
                "type": "error",
                "message": str(e),
                "timestamp": time.time(),
            }
        )
        # Reset client on error - will recreate on next message
        self._client = None
        self.active = False

        if self.tts_player:
            self.tts_player.speak("Sorry, I encountered an error. Please try again.")

    def is_timed_out(self) -> bool:
        """Check if conversation has timed out."""
        if not self.active:
            return False
        return time.time() - self.last_interaction_time > self.timeout
