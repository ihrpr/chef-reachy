"""LLM client using Ollama."""

import json
import logging
from typing import Any

from chef_reachy.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._initialized = False
        self._ollama = None

    def initialize(self) -> bool:
        """Initialize Ollama client and check if model is available."""
        try:
            import ollama

            self._ollama = ollama

            # Check if Ollama server is running
            try:
                client = ollama.Client(host=self.config.host)
                models_response = client.list()
                logger.info(f"Connected to Ollama server at {self.config.host}")

                # Check if the model is available
                # The response has a 'models' attribute which is a list of model objects
                model_names = [m.model for m in models_response.models]
                if self.config.model_name not in model_names:
                    logger.warning(
                        f"Model '{self.config.model_name}' not found. Available models: {model_names}"
                    )
                    logger.warning(f"Run 'ollama pull {self.config.model_name}' to download the model")
                    return False

                logger.info(f"Model '{self.config.model_name}' is available")
                self._initialized = True
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Ollama server: {e}")
                logger.error("Make sure Ollama is running with 'ollama serve'")
                return False

        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
            return False

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (uses config default if None)

        Returns:
            Generated text
        """
        if not self._initialized or self._ollama is None:
            logger.error("Ollama client not initialized")
            return ""

        try:
            client = self._ollama.Client(host=self.config.host)

            response = client.generate(
                model=self.config.model_name,
                prompt=prompt,
                system=system_prompt or self.config.system_prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )

            return response.get("response", "")

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def extract_product_info(self, ocr_text: str) -> dict[str, Any]:
        """
        Extract product name and expiration date from OCR text.

        Args:
            ocr_text: Text extracted from OCR

        Returns:
            Dictionary with 'product_name' and 'expiration_date' fields
        """
        if not self._initialized:
            logger.error("Ollama client not initialized")
            return {"product_name": None, "expiration_date": None, "error": "LLM not initialized"}

        prompt = f"""
Extract the product name and expiration date from the following text:

{ocr_text}

Respond ONLY with valid JSON in this format:
{{"product_name": "...", "expiration_date": "YYYY-MM-DD"}}

If you cannot find the information, use null for that field.
"""

        try:
            response = self.generate(prompt)
            logger.info(f"LLM raw response: {response}")

            # Try to parse JSON from response
            # Sometimes LLMs add extra text, so try to extract JSON
            response = response.strip()

            # Find the first complete JSON object
            start_idx = response.find("{")
            if start_idx == -1:
                logger.error("No JSON found in LLM response")
                return {
                    "product_name": None,
                    "expiration_date": None,
                    "error": "No JSON in response",
                    "raw_response": response,
                }

            # Find matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                logger.info(f"Extracted JSON: {json_str}")
                result = json.loads(json_str)

                return {
                    "product_name": result.get("product_name"),
                    "expiration_date": result.get("expiration_date"),
                    "raw_response": response,
                }
            else:
                logger.error("Could not find matching closing brace")
                return {
                    "product_name": None,
                    "expiration_date": None,
                    "error": "Invalid JSON structure",
                    "raw_response": response,
                }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            return {
                "product_name": None,
                "expiration_date": None,
                "error": f"JSON parse error: {e}",
            }
        except Exception as e:
            logger.error(f"Product info extraction failed: {e}")
            return {
                "product_name": None,
                "expiration_date": None,
                "error": str(e),
            }

    def is_ready(self) -> bool:
        """Check if LLM client is ready."""
        return self._initialized

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "status": "ready",
            "model": self.config.model_name,
            "host": self.config.host,
        }
