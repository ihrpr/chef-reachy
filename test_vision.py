#!/usr/bin/env python3
"""Test script to debug SmolVLM2 model loading."""

import sys
import traceback

print("=" * 60)
print("SmolVLM2 Model Loading Test")
print("=" * 60)

# Test 1: Check transformers version
print("\n1. Checking transformers version...")
try:
    import transformers
    print(f"   ✓ transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"   ✗ Failed to import transformers: {e}")
    sys.exit(1)

# Test 2: Check if AutoProcessor exists
print("\n2. Checking AutoProcessor...")
try:
    from transformers import AutoProcessor
    print("   ✓ AutoProcessor imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import AutoProcessor: {e}")
    sys.exit(1)

# Test 3: Check if AutoModelForImageTextToText exists
print("\n3. Checking AutoModelForImageTextToText...")
try:
    from transformers import AutoModelForImageTextToText
    print("   ✓ AutoModelForImageTextToText imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import AutoModelForImageTextToText: {e}")
    sys.exit(1)

# Test 4: Try to load the processor
print("\n4. Attempting to load SmolVLM2 processor...")
try:
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        cache_dir="./cache/huggingface"
    )
    print(f"   ✓ Processor loaded successfully: {type(processor)}")
except Exception as e:
    print(f"   ✗ Failed to load processor:")
    print(f"   Error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try to load the model (this will be slow on first run)
print("\n5. Attempting to load SmolVLM2 model...")
print("   (This may take 10-30 seconds...)")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        cache_dir="./cache/huggingface",
        device_map="auto",
        torch_dtype="auto"
    )
    print(f"   ✓ Model loaded successfully: {type(model)}")
    print(f"   Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"   ✗ Failed to load model:")
    print(f"   Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
