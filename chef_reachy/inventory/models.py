"""Data models for inventory management."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FoodItem:
    """Represents a food item in the inventory."""

    product_name: str
    expiration_date: str | None = None
    detected_at: datetime = field(default_factory=datetime.now)
    ocr_text: str | None = None
    confidence: float = 1.0
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            # Generate a simple ID based on timestamp
            self.id = f"item_{int(self.detected_at.timestamp())}"

    def is_expired(self) -> bool:
        """Check if the item is expired."""
        if not self.expiration_date:
            return False

        try:
            exp_date = datetime.strptime(self.expiration_date, "%Y-%m-%d")
            return datetime.now() > exp_date
        except ValueError:
            # If date format is invalid, assume not expired
            return False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "product_name": self.product_name,
            "expiration_date": self.expiration_date,
            "detected_at": self.detected_at.isoformat(),
            "ocr_text": self.ocr_text,
            "confidence": self.confidence,
            "is_expired": self.is_expired(),
        }
