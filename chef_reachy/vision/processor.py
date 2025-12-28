"""Vision processor using SmolVLM2 model."""

import base64
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from transformers import AutoModelForImageTextToText, AutoProcessor

from chef_reachy.vision.config import VisionConfig

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Handles SmolVLM2 model loading and inference."""

    def __init__(self, vision_config: VisionConfig | None = None):
        """Initialize the vision processor.

        Args:
            vision_config: Configuration for vision processing. If None, uses defaults.
        """
        self.vision_config = vision_config or VisionConfig()
        self.model_path = self.vision_config.model_path
        self.device = self._determine_device()
        self.processor = None
        self.model = None
        self._initialized = False

    def _determine_device(self) -> str:
        """Determine the best available device for model inference.

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        pref = self.vision_config.device_preference

        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if pref == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"

        # auto: prefer mps on Apple Silicon, then cuda, else cpu
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """Load model and processor onto the selected device.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.info("Vision processor already initialized")
            return True

        try:
            # Use cache directory from environment or config
            cache_dir = os.path.expanduser(self.vision_config.cache_dir)
            logger.info(f"Cache directory: {cache_dir}")

            # Check if model is already cached
            model_cache_path = os.path.join(
                cache_dir,
                f"models--{self.model_path.replace('/', '--')}"
            )
            is_cached = os.path.exists(model_cache_path)

            if is_cached:
                logger.info(f"✓ Model found in cache at {model_cache_path}")
                logger.info("Loading from cache (should be fast)...")
            else:
                logger.info(f"✗ Model NOT in cache")
                logger.info(f"Will download to {model_cache_path}")
                logger.info("This will take 10-30 seconds and download ~5GB...")

            start_time = time.time()
            logger.info(f"Loading processor for {self.model_path}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )  # type: ignore
            processor_time = time.time() - start_time
            logger.info(f"Processor loaded in {processor_time:.2f}s")

            # Select dtype depending on device
            if self.device == "cuda":
                dtype = torch.bfloat16
            elif self.device == "mps":
                dtype = torch.float32  # best for MPS
            else:
                dtype = torch.float32

            model_kwargs: dict[str, Any] = {"dtype": dtype}

            # flash_attention_2 is CUDA-only; skip on MPS/CPU
            if self.device == "cuda":
                model_kwargs["_attn_implementation"] = "flash_attention_2"

            # Load model weights
            logger.info(f"Loading model weights on {self.device}...")
            logger.info("This may take 10-30 seconds and use ~8GB RAM...")
            logger.info("Your laptop may appear to hang - this is normal!")
            model_start_time = time.time()

            # Use low_cpu_mem_usage to reduce memory spikes during loading
            self.model = AutoModelForImageTextToText.from_pretrained(  # type: ignore
                self.model_path,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                **model_kwargs
            ).to(self.device)

            model_time = time.time() - model_start_time
            logger.info(f"Model loaded in {model_time:.2f}s")

            if self.model is not None:
                self.model.eval()

            self._initialized = True
            total_time = time.time() - start_time
            logger.info(f"✓ Vision initialization complete in {total_time:.2f}s total")
            logger.info(f"Device: {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            self._initialized = False
            return False

    def process_image(
        self,
        cv2_image: NDArray[np.uint8],
        prompt: str | None = None,
    ) -> str:
        """Process CV2 image and return description with retry logic.

        Args:
            cv2_image: Image in OpenCV format (BGR)
            prompt: Custom prompt for the model. If None, uses default.

        Returns:
            Description string from the model
        """
        if not self._initialized or self.processor is None or self.model is None:
            return "Vision model not initialized"

        if prompt is None:
            prompt = self.vision_config.default_prompt

        for attempt in range(self.vision_config.max_retries):
            try:
                # Convert to JPEG bytes
                success, jpeg_buffer = cv2.imencode(
                    ".jpg",
                    cv2_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.vision_config.jpeg_quality],
                )
                if not success:
                    return "Failed to encode image"

                # Convert to base64
                image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                # Move tensors to device WITHOUT forcing dtype (keeps input_ids as torch.long)
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.vision_config.max_new_tokens,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                generated_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                # Extract just the response part
                full_text = generated_texts[0]
                response = self._extract_response(full_text)

                # Clean up GPU memory if using CUDA/MPS
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

                return response.replace(chr(10), " ").strip()

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on attempt {attempt + 1}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if attempt < self.vision_config.max_retries - 1:
                    time.sleep(self.vision_config.retry_delay * (attempt + 1))
                else:
                    return "GPU out of memory - vision processing failed"

            except Exception as e:
                logger.error(f"Vision processing failed (attempt {attempt + 1}): {e}")
                if attempt < self.vision_config.max_retries - 1:
                    time.sleep(self.vision_config.retry_delay)
                else:
                    return f"Vision processing error after {self.vision_config.max_retries} attempts"

        return "Vision processing failed"

    def _extract_response(self, full_text: str) -> str:
        """Extract the assistant's response from the full generated text.

        Args:
            full_text: Full text generated by the model

        Returns:
            Extracted response text
        """
        # Handle different response formats
        markers = ["assistant\n", "Assistant:", "Response:", "\n\n"]

        for marker in markers:
            if marker in full_text:
                response = full_text.split(marker)[-1].strip()
                if response:  # Ensure we got a meaningful response
                    return response

        # Fallback: return the full text cleaned up
        return full_text.strip()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info: dict[str, Any] = {
            "initialized": self._initialized,
            "device": self.device,
            "model_path": self.model_path,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }

        if torch.cuda.is_available():
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        else:
            info["gpu_memory_gb"] = "N/A"

        return info

    def cleanup(self) -> None:
        """Unload model from memory and free resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            # Clean up GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

            self._initialized = False
            logger.info("Vision model cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
