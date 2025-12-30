# Chef Reachy - OCR & LLM Integration Setup Guide

This guide will help you set up OCR text detection and LLM-based product information extraction for your Chef Reachy app.

## What's New

The app now includes:
1. **EasyOCR** - Reads text from food packaging
2. **Ollama with Qwen2.5-7B** - Extracts product name and expiration date from OCR text
3. **Inventory Management** - Tracks detected food items with expiration dates

## Installation Steps

### 1. Install Dependencies

First, update your dependencies:

```bash
cd /Users/inna/code/reachy_apps/chef_reachy
pip install -e .
```

This will install:
- `easyocr>=1.7.0` (~80MB model download on first run)
- `ollama>=0.4.0` (Python client for Ollama)

### 2. Install and Setup Ollama

#### Install Ollama

```bash
brew install ollama
```

#### Start Ollama Server

```bash
ollama serve
```

Keep this running in a separate terminal.

#### Download Qwen2.5-7B Model

In another terminal:

```bash
ollama pull qwen2.5:7b
```

This will download ~4.7GB. The model is optimized for your M2 Mac and will run at 15-20 tokens/second.

#### Verify Installation

```bash
ollama list
```

You should see `qwen2.5:7b` in the list.

## How It Works

### Workflow

1. **Food Detection** - OWL-ViT detects hand with food in the camera view
2. **OCR Processing** - When food is detected, EasyOCR reads text from the packaging
3. **LLM Extraction** - Qwen2.5 extracts product name and expiration date from OCR text
4. **Inventory Update** - Item is added to the in-memory inventory list
5. **TTS Feedback** - Reachy announces the new item

### In-Memory Storage

Items are stored in memory during the app session. You can access them via:

```python
# In your code
items = self.inventory.get_all_items()
for item in items:
    print(f"{item.product_name} - expires: {item.expiration_date}")
```

## Configuration

### OCR Configuration

Edit `chef_reachy/ocr/config.py`:

```python
@dataclass
class OCRConfig:
    languages: list[str] = ["en"]  # Add more languages: ["en", "fr", "es"]
    device_preference: str = "mps"  # "mps" for Apple Silicon, "cuda" for NVIDIA
    confidence_threshold: float = 0.5  # Min confidence for text detection
```

### LLM Configuration

Edit `chef_reachy/llm/config.py`:

```python
@dataclass
class LLMConfig:
    model_name: str = "qwen2.5:7b"  # Or "phi3.5:mini" for faster/lighter
    temperature: float = 0.1  # Lower = more deterministic
    max_tokens: int = 500
```

## Testing

### 1. Start the App

```bash
python -m chef_reachy.main
```

You should see initialization logs for:
- OWL-ViT Detector
- Kokoro TTS
- EasyOCR Reader
- Ollama LLM Client
- Inventory Manager

### 2. Test Detection

Hold a food package with visible text in front of the camera. The app will:
1. Detect the hand with food item (every 2.5 seconds)
2. Run OCR on the packaging when food is detected
3. Extract product info with LLM
4. Add to in-memory inventory list
5. Announce via TTS

Watch the logs to see:
- OCR text detected
- Product info extracted
- Item added to inventory

## Troubleshooting

### Ollama Not Connected

If you see "Failed to connect to Ollama server":
1. Make sure `ollama serve` is running
2. Check the host in `llm/config.py` (default: `http://localhost:11434`)

### Model Not Found

If you see "Model 'qwen2.5:7b' not found":
```bash
ollama pull qwen2.5:7b
```

### OCR Not Working

If EasyOCR fails to initialize:
1. Check that PyTorch is installed correctly
2. Verify MPS is available on your M2 Mac
3. Check logs for specific errors

### No Text Detected

If OCR doesn't find text:
1. Ensure the packaging has clear, readable text
2. Hold the package steady and close enough to the camera
3. Adjust `confidence_threshold` in `OCRConfig` (try 0.3 for more sensitivity)

## Performance Tips

### Using Phi-3.5-mini Instead

If Qwen2.5-7B is too slow or uses too much memory, switch to Phi-3.5-mini:

```bash
ollama pull phi3.5:mini
```

Update `chef_reachy/llm/config.py`:
```python
model_name: str = "phi3.5:mini"
```

This is about half the size and faster, with slightly lower accuracy.

### Optimizing OCR

For faster OCR at the cost of accuracy:
- Set `confidence_threshold: float = 0.6` (higher = fewer false positives)
- Reduce `contrast_threshold` if text is low contrast

## Module Structure

```
chef_reachy/
├── ocr/              # EasyOCR integration
│   ├── config.py     # OCR configuration
│   ├── ocr_reader.py # OCR implementation
│   └── __init__.py
├── llm/              # Ollama/Qwen2.5 integration
│   ├── config.py     # LLM configuration
│   ├── client.py     # Ollama client
│   └── __init__.py
├── inventory/        # Inventory management
│   ├── models.py     # FoodItem data model
│   ├── manager.py    # InventoryManager
│   └── __init__.py
└── main.py           # Main app with integrated workflow
```

## Next Steps

1. Create a web frontend to display the inventory
2. Add notifications for expiring items
3. Implement barcode scanning as an alternative to OCR
4. Add support for custom product categories
5. Integrate with recipe suggestions based on inventory

Enjoy your smart Chef Reachy! 🤖🍕
