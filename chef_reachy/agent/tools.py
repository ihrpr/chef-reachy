"""Claude Agent SDK tools for Chef Reachy."""

import asyncio
import base64
import io
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from anthropic import Anthropic
from PIL import Image
from reachy_mini import ReachyMini

from chef_reachy.inventory.manager import InventoryManager
from chef_reachy.inventory.models import FoodItem

logger = logging.getLogger(__name__)

# Type alias for broadcaster function
Broadcaster = Callable[[dict[str, Any]], None]


def encode_image_base64(frame: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    # Convert BGR to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = frame

    # Convert to PIL Image
    pil_img = Image.fromarray(rgb_frame)

    # Encode to JPEG bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    # Base64 encode
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def analyze_images_with_claude(
    images: list[str], api_key: str, model: str = "claude-sonnet-4-5-20250929"
) -> dict[str, Any]:
    """
    Analyze food packaging images with Claude Vision API.

    Args:
        images: List of base64-encoded images
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        Dictionary with product_name and expiration_date
    """
    try:
        from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam

        client = Anthropic(api_key=api_key)

        # Build content with all images
        content: list[ImageBlockParam | TextBlockParam] = []
        for img_b64 in images:
            content.append(
                ImageBlockParam(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                )
            )

        # Add prompt
        content.append(
            TextBlockParam(
                type="text",
                text=(
                    "Analyze these food packaging images from different angles. "
                    "Extract the product name and expiration date from visible text. "
                    "Return ONLY a JSON object with this exact format: "
                    '{"product_name": "Product Name", "expiration_date": "YYYY-MM-DD"} '
                    "If expiration date is not found, use null. "
                    "Do not include any other text or explanation."
                ),
            )
        )

        # Call Claude Vision API
        message_param: MessageParam = {"role": "user", "content": content}
        response = client.messages.create(
            model=model, max_tokens=512, messages=[message_param]
        )

        # Parse response - handle different content block types
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = block.text.strip()
                break

        # Extract JSON from response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            return {
                "product_name": data.get("product_name", "Unknown"),
                "expiration_date": data.get("expiration_date"),
            }

        logger.error(f"No JSON found in response: {response_text}")
        return {"product_name": "Unknown", "expiration_date": None}

    except Exception as e:
        logger.error(f"Error analyzing images with Claude: {e}")
        return {"product_name": "Unknown", "expiration_date": None}


def create_chef_tools(
    reachy_mini: ReachyMini,
    inventory: InventoryManager,
    api_key: str,
    broadcaster: Broadcaster | None = None,
):
    """
    Create MCP server tools for Chef Reachy.

    Args:
        reachy_mini: Reachy Mini instance
        inventory: Inventory manager
        api_key: Anthropic API key
        broadcaster: Optional callback to broadcast events to WebSocket clients

    Returns:
        List of tool functions decorated with @tool
    """
    from claude_agent_sdk import tool
    from chef_reachy.agent.movements import RobotMovements

    # Initialize movement controller
    movements = RobotMovements(reachy_mini)

    def broadcast(event: dict[str, Any]):
        """Broadcast event if broadcaster is available."""
        if broadcaster:
            broadcaster(event)

    def trigger_movement(movement_type: str) -> None:
        """
        Trigger a robot movement in fire-and-forget mode.

        Args:
            movement_type: One of "nod", "no_shake", "greet", "thinking"
        """
        movements.trigger_movement_async(movement_type)

    @tool(
        "scan_food_item",
        "Capture and analyze food packaging from multiple angles to add to inventory",
        {"num_angles": int},
    )
    async def scan_food_item(args: dict[str, Any]) -> dict[str, Any]:
        """
        Scan a food item by capturing images from multiple angles.

        This tool:
        1. Captures 3 images at 3-second intervals
        2. Sends them to Claude Vision API for analysis
        3. Extracts product name and expiration date
        4. Adds the item to inventory
        """
        num_angles = args.get("num_angles", 3)

        try:
            # Robot looks curious and ready to scan
            trigger_movement("thinking")

            # Broadcast tool start
            broadcast(
                {
                    "type": "tool_progress",
                    "tool_name": "scan_food_item",
                    "message": f"Starting food scan with {num_angles} angles...",
                    "timestamp": time.time(),
                }
            )

            # Capture images
            images = []
            for i in range(num_angles):
                try:
                    frame = reachy_mini.media.get_frame()
                    if frame is not None:
                        img_b64 = encode_image_base64(frame)
                        images.append(img_b64)

                        # Broadcast captured image
                        broadcast(
                            {
                                "type": "tool_image",
                                "tool_name": "scan_food_item",
                                "image": img_b64,
                                "image_index": i + 1,
                                "total_images": num_angles,
                                "message": f"Captured image {i + 1}/{num_angles}",
                                "timestamp": time.time(),
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to capture frame {i + 1}: {e}")

                # Wait between captures (except for last one)
                if i < num_angles - 1:
                    await asyncio.sleep(3)

            if not images:
                broadcast(
                    {
                        "type": "tool_error",
                        "tool_name": "scan_food_item",
                        "message": "Failed to capture any images",
                        "timestamp": time.time(),
                    }
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Failed to capture any images from camera.",
                        }
                    ],
                    "is_error": True,
                }

            # Broadcast analyzing status
            broadcast(
                {
                    "type": "tool_progress",
                    "tool_name": "scan_food_item",
                    "message": f"Analyzing {len(images)} images with Claude Vision...",
                    "timestamp": time.time(),
                }
            )

            # Analyze with Claude Vision - robot thinking
            trigger_movement("thinking")
            result = await analyze_images_with_claude(images, api_key)

            if result["product_name"] == "Unknown":
                # Robot shows concern/confusion
                trigger_movement("no_shake")
                broadcast(
                    {
                        "type": "tool_error",
                        "tool_name": "scan_food_item",
                        "message": "Could not identify product from images",
                        "timestamp": time.time(),
                    }
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Could not identify product from images. Please ensure packaging is clearly visible.",
                        }
                    ],
                    "is_error": True,
                }

            # Create food item
            item = FoodItem(
                product_name=result["product_name"],
                expiration_date=result["expiration_date"],
                ocr_text=f"Analyzed from {len(images)} angles with Claude Vision",
                confidence=0.95,
            )

            # Add to inventory
            inventory.add_item(item)

            # Robot nods in acknowledgment of successful scan
            trigger_movement("nod")

            # Broadcast success with result
            broadcast(
                {
                    "type": "tool_result",
                    "tool_name": "scan_food_item",
                    "status": "success",
                    "product_name": item.product_name,
                    "expiration_date": item.expiration_date,
                    "message": f"Added {item.product_name} to inventory",
                    "timestamp": time.time(),
                }
            )

            # Broadcast inventory update
            all_items = inventory.get_all_items()
            broadcast(
                {
                    "type": "inventory_update",
                    "items": [
                        {
                            "id": it.id,
                            "product_name": it.product_name,
                            "expiration_date": it.expiration_date,
                            "detected_at": it.detected_at.isoformat(),
                        }
                        for it in all_items
                    ],
                    "timestamp": time.time(),
                }
            )

            # Return MCP-formatted response
            result_text = f"Successfully added '{item.product_name}' to inventory."
            if item.expiration_date:
                result_text += f" Expiration date: {item.expiration_date}."
            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as e:
            logger.error(f"Error scanning food item: {e}")
            broadcast(
                {
                    "type": "tool_error",
                    "tool_name": "scan_food_item",
                    "message": str(e),
                    "timestamp": time.time(),
                }
            )
            return {
                "content": [{"type": "text", "text": f"Error scanning food item: {e}"}],
                "is_error": True,
            }

    @tool("get_inventory", "Retrieve all items in the food inventory", {})
    async def get_inventory(args: dict[str, Any]) -> dict[str, Any]:
        """Get all items in inventory."""
        try:
            # Robot shows it's thinking/checking
            trigger_movement("thinking")

            items = inventory.get_all_items()

            # Broadcast inventory query
            broadcast(
                {
                    "type": "tool_result",
                    "tool_name": "get_inventory",
                    "status": "success",
                    "message": f"Retrieved {len(items)} items from inventory",
                    "timestamp": time.time(),
                }
            )

            # Build human-readable response for Claude
            if not items:
                result_text = "The inventory is empty. No items have been added yet."
            else:
                item_lines = []
                has_expired = False
                for item in items:
                    line = f"- {item.product_name}"
                    if item.expiration_date:
                        if item.is_expired():
                            line += f" (EXPIRED: {item.expiration_date})"
                            has_expired = True
                        else:
                            line += f" (expires: {item.expiration_date})"
                    item_lines.append(line)
                result_text = f"Inventory contains {len(items)} item(s):\n" + "\n".join(
                    item_lines
                )

                # Show concern if there are expired items
                if has_expired:
                    trigger_movement("no_shake")

            return {"content": [{"type": "text", "text": result_text}]}
        except Exception as e:
            logger.error(f"Error getting inventory: {e}")
            return {
                "content": [{"type": "text", "text": f"Error getting inventory: {e}"}],
                "is_error": True,
            }

    @tool(
        "remove_item",
        "Remove an item from inventory by product name",
        {"product_name": str},
    )
    async def remove_item(args: dict[str, Any]) -> dict[str, Any]:
        """Remove item from inventory by product name."""
        try:
            product_name = args.get("product_name", "")
            if not product_name:
                return {
                    "content": [
                        {"type": "text", "text": "Error: Product name is required"}
                    ],
                    "is_error": True,
                }

            # Find matching items
            matching_items = inventory.get_items_by_name(product_name)

            if not matching_items:
                broadcast(
                    {
                        "type": "tool_error",
                        "tool_name": "remove_item",
                        "message": f"No items found matching '{product_name}'",
                        "timestamp": time.time(),
                    }
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"No items found matching '{product_name}'",
                        }
                    ],
                    "is_error": True,
                }

            # Remove the first matching item
            removed = inventory.remove_item(matching_items[0].id)

            if removed:
                # Robot nods in confirmation
                trigger_movement("nod")

                # Broadcast removal
                broadcast(
                    {
                        "type": "tool_result",
                        "tool_name": "remove_item",
                        "status": "success",
                        "product_name": matching_items[0].product_name,
                        "message": f"Removed {matching_items[0].product_name} from inventory",
                        "timestamp": time.time(),
                    }
                )

                # Broadcast inventory update
                all_items = inventory.get_all_items()
                broadcast(
                    {
                        "type": "inventory_update",
                        "items": [
                            {
                                "id": it.id,
                                "product_name": it.product_name,
                                "expiration_date": it.expiration_date,
                                "detected_at": it.detected_at.isoformat(),
                            }
                            for it in all_items
                        ],
                        "timestamp": time.time(),
                    }
                )

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Successfully removed '{matching_items[0].product_name}' from inventory.",
                        }
                    ]
                }
            else:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Failed to remove item from inventory.",
                        }
                    ],
                    "is_error": True,
                }

        except Exception as e:
            logger.error(f"Error removing item: {e}")
            return {
                "content": [{"type": "text", "text": f"Error removing item: {e}"}],
                "is_error": True,
            }

    @tool("clear_inventory", "Clear all items from inventory", {})
    async def clear_inventory(args: dict[str, Any]) -> dict[str, Any]:
        """Clear all items from inventory."""
        try:
            inventory.clear()

            # Robot nods to confirm clearing
            trigger_movement("nod")

            # Broadcast clear
            broadcast(
                {
                    "type": "tool_result",
                    "tool_name": "clear_inventory",
                    "status": "success",
                    "message": "Inventory cleared",
                    "timestamp": time.time(),
                }
            )

            # Broadcast empty inventory
            broadcast(
                {
                    "type": "inventory_update",
                    "items": [],
                    "timestamp": time.time(),
                }
            )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Inventory has been cleared. All items have been removed.",
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error clearing inventory: {e}")
            return {
                "content": [{"type": "text", "text": f"Error clearing inventory: {e}"}],
                "is_error": True,
            }

    return [scan_food_item, get_inventory, remove_item, clear_inventory]
