"""Inventory manager for tracking food items."""

import json
import logging
from pathlib import Path

from chef_reachy.inventory.models import FoodItem

logger = logging.getLogger(__name__)


class InventoryManager:
    """Manages the list of detected food items."""

    def __init__(self, storage_path: str | None = None):
        self.items: list[FoodItem] = []
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.load_from_file()

    def add_item(self, item: FoodItem) -> None:
        """Add a new item to the inventory."""
        self.items.append(item)
        logger.info(f"Added item to inventory: {item.product_name} (expires: {item.expiration_date})")

        if self.storage_path:
            self.save_to_file()

    def remove_item(self, item_id: str) -> bool:
        """Remove an item from inventory by ID."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                removed = self.items.pop(i)
                logger.info(f"Removed item from inventory: {removed.product_name}")

                if self.storage_path:
                    self.save_to_file()
                return True
        return False

    def get_all_items(self) -> list[FoodItem]:
        """Get all items in the inventory."""
        return self.items

    def get_expired_items(self) -> list[FoodItem]:
        """Get all expired items."""
        return [item for item in self.items if item.is_expired()]

    def get_items_by_name(self, product_name: str) -> list[FoodItem]:
        """Get all items matching a product name."""
        return [item for item in self.items if product_name.lower() in item.product_name.lower()]

    def clear(self) -> None:
        """Clear all items from inventory."""
        self.items.clear()
        logger.info("Cleared all items from inventory")

        if self.storage_path:
            self.save_to_file()

    def save_to_file(self) -> None:
        """Save inventory to JSON file."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {"items": [item.to_dict() for item in self.items]}

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved inventory to {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to save inventory: {e}")

    def load_from_file(self) -> None:
        """Load inventory from JSON file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for item_data in data.get("items", []):
                from datetime import datetime

                item = FoodItem(
                    id=item_data.get("id"),
                    product_name=item_data.get("product_name", "Unknown"),
                    expiration_date=item_data.get("expiration_date"),
                    detected_at=datetime.fromisoformat(item_data.get("detected_at", datetime.now().isoformat())),
                    ocr_text=item_data.get("ocr_text"),
                    confidence=item_data.get("confidence", 1.0),
                )
                self.items.append(item)

            logger.info(f"Loaded {len(self.items)} items from {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to load inventory: {e}")

    def to_dict(self) -> dict:
        """Convert inventory to dictionary."""
        return {
            "total_items": len(self.items),
            "expired_items": len(self.get_expired_items()),
            "items": [item.to_dict() for item in self.items],
        }
