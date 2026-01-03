"""Configuration for Claude Agent."""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for Claude Agent SDK."""

    name = "Chef"
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024
    temperature: float = 0.7

    def __post_init__(self):
        """Load API key from environment if not provided."""
        self.system_prompt: str = (
            f"Your name is {self.name}. You are a helpful kitchen assistant managing food inventory for a Reachy Mini robot. "
            "IMPORTANT: Keep responses very concise since they will be spoken aloud."
            "When users ask you to scan items, use the scan_food_item tool. "
        )
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or ""
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY must be set in environment or provided in config"
                )
