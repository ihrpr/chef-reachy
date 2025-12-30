"""LLM module for Ollama integration."""

from chef_reachy.llm.client import OllamaClient
from chef_reachy.llm.config import LLMConfig

__all__ = ["LLMConfig", "OllamaClient"]
