"""LLM configuration for Ollama."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for Ollama LLM."""

    # Model to use (gemma:2b is very light and fast)
    model_name: str = "gemma:2b"

    # Ollama server URL
    host: str = "http://localhost:11434"

    # Temperature for generation (0.0-1.0, lower = more deterministic)
    temperature: float = 0.1

    # Maximum tokens to generate
    max_tokens: int = 500

    # System prompt for extraction tasks
    system_prompt: str = (
        "You are a helpful assistant that extracts structured information from OCR text. "
        "Extract product name and expiration date from the given text. "
        "Respond in JSON format with 'product_name' and 'expiration_date' fields. "
        "If information is not found, use null for that field. "
        "For expiration date, convert to YYYY-MM-DD format if possible."
    )
