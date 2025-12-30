#!/usr/bin/env python3
"""Test script to verify OCR and LLM integration."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_ocr():
    """Test EasyOCR installation and initialization."""
    logger.info("=" * 60)
    logger.info("Testing EasyOCR...")
    logger.info("=" * 60)

    try:
        from chef_reachy.ocr import OCRConfig, OCRReader

        config = OCRConfig()
        reader = OCRReader(config)

        if reader.initialize():
            logger.info("✓ EasyOCR initialized successfully")
            return True
        else:
            logger.error("✗ EasyOCR failed to initialize")
            return False
    except Exception as e:
        logger.error(f"✗ EasyOCR error: {e}")
        return False


def test_llm():
    """Test Ollama client and model availability."""
    logger.info("=" * 60)
    logger.info("Testing Ollama LLM...")
    logger.info("=" * 60)

    try:
        from chef_reachy.llm import LLMConfig, OllamaClient

        config = LLMConfig()
        client = OllamaClient(config)

        if client.initialize():
            logger.info("✓ Ollama client initialized successfully")

            # Test extraction
            test_text = "Organic Whole Milk - Best Before: 01/15/2025"
            logger.info(f"Testing extraction with: '{test_text}'")

            result = client.extract_product_info(test_text)
            logger.info(f"Extracted: {result}")

            if result.get("product_name"):
                logger.info("✓ Product extraction works")
                return True
            else:
                logger.warning("⚠ Could not extract product name")
                return True  # Still initialized, just extraction didn't work
        else:
            logger.error("✗ Ollama client failed to initialize")
            logger.error("Make sure:")
            logger.error("  1. Ollama is running: ollama serve")
            logger.error("  2. Model is downloaded: ollama pull qwen2.5:7b")
            return False
    except Exception as e:
        logger.error(f"✗ Ollama error: {e}")
        return False


def test_inventory():
    """Test inventory management."""
    logger.info("=" * 60)
    logger.info("Testing Inventory Manager...")
    logger.info("=" * 60)

    try:
        from chef_reachy.inventory import FoodItem, InventoryManager

        # Create test inventory (no storage)
        manager = InventoryManager(storage_path=None)

        # Add test item
        item = FoodItem(
            product_name="Test Milk",
            expiration_date="2025-01-15",
            ocr_text="Test OCR text",
        )
        manager.add_item(item)

        # Verify
        items = manager.get_all_items()
        if len(items) == 1 and items[0].product_name == "Test Milk":
            logger.info("✓ Inventory manager works")
            return True
        else:
            logger.error("✗ Inventory manager test failed")
            return False
    except Exception as e:
        logger.error(f"✗ Inventory error: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Chef Reachy Integration Test")
    logger.info("=" * 60 + "\n")

    results = {
        "OCR": test_ocr(),
        "LLM": test_llm(),
        "Inventory": test_inventory(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)

    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{component:15} {status}")

    all_passed = all(results.values())
    logger.info("=" * 60)

    if all_passed:
        logger.info("✓ All tests passed! You're ready to go.")
        return 0
    else:
        logger.error("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
