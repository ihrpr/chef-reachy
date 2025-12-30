---
title: Chef Reachy
emoji: 👨‍🍳
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Food detection using OWL-ViT object detection on Reachy Mini
tags:
 - reachy_mini
 - reachy_mini_python_app
 - computer_vision
 - object_detection
 - food_detection
---

# Chef Reachy

A Reachy Mini application that uses OWL-ViT (Open-World Localization with Vision Transformer) for zero-shot object detection to detect hands holding food and automatically track them with the robot's cameras.

## Features

- **Continuous food detection** - automatically detects hands holding food every 2.5 seconds
- **Zero-shot detection** using OWL-ViT (google/owlvit-base-patch32) - no training required
- **Automatic camera tracking** - cameras follow the detected hand with food using `look_at_image()` API
- **OCR text detection** - reads product packaging text using EasyOCR
- **LLM-powered extraction** - extracts product name and expiration date using Gemma 2B
- **Inventory tracking** - maintains in-memory list of detected food items with expiration dates
- **Real-time WebSocket streaming** - live camera feed with detection updates
- **Bounding box visualization** - shows where hands and food are located in the frame
- **Fully automatic operation** - no button clicks needed, just open the web interface
- **Smooth camera movements** - robot looks at detected hand position in image
- Device-aware model loading (CUDA or CPU)
- Lightweight model (~600MB vs 4.5GB for SmolVLM2)
- Fast inference (2-4 seconds on CPU)
- **Text-to-speech announcements** - Robot speaks friendly phrases when food is detected using Kokoro-82M (82MB model)

## Architecture

Chef Reachy now includes three intelligent modules:

1. **OCR Module** (`chef_reachy/ocr/`) - EasyOCR integration for reading text from packaging
2. **LLM Module** (`chef_reachy/llm/`) - Ollama with Gemma 2B for extracting product information
3. **Inventory Module** (`chef_reachy/inventory/`) - In-memory tracking of food items with expiration dates

## Workflow

```
Hand with food detected (OWL-ViT)
    ↓
Crop detected region
    ↓
Run OCR (EasyOCR)
    ↓
Extract product info (Gemma 2B LLM)
    ↓
Add to inventory list
    ↓
Announce via TTS
```

## How It Works

1. **Automatic Initialization**: OWL-ViT model loads on startup (before server starts)
2. **Continuous Detection Loop**: Main loop captures frames and runs detection every 2.5 seconds (matches inference time)
3. **Zero-Shot Detection**: Uses text queries like "hand holding food" to detect hands without any training
4. **Bounding Box Localization**: Returns precise bounding boxes showing where the hand is in the image
5. **Camera Tracking**: When food is detected, calculates center of bounding box and uses `reachy_mini.look_at_image(x, y)` to move cameras
6. **OCR Processing**: Crops detected region and runs EasyOCR to read text from packaging
7. **LLM Extraction**: Uses Gemma 2B via Ollama to extract product name and expiration date from OCR text
8. **Inventory Update**: Adds item to in-memory inventory with metadata (product name, expiration date, confidence)
9. **WebSocket Streaming**: Broadcasts detection results (both "detected" and "no_detection" status) in real-time to web interface
10. **Text-to-Speech**: When food detected, generates speech using Kokoro-82M and announces the item added to inventory
11. **Live Visualization**: Web interface displays camera feed with bounding boxes and detection status automatically

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
uv sync
# or
pip install -e .
```

3. Install and setup Ollama for LLM-powered extraction:
```bash
# Install Ollama
brew install ollama

# Start Ollama server (in a separate terminal)
ollama serve

# Download Gemma 2B model
ollama pull gemma:2b
```

The Gemma 2B model is lightweight (~1.6GB) and runs efficiently on M2 Macs.

## Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

Configuration options:
- `HF_HOME`: Directory for model cache (default: `~/.cache/huggingface`)
- `HF_TOKEN`: Optional Hugging Face token for gated models
- `VISION_MODEL`: Vision model to use (default: `google/owlvit-base-patch32`)
- `VISION_DEVICE`: Device preference (`auto`, `cuda`, or `cpu`)
- `VISION_DETECTION_THRESHOLD`: Confidence threshold (default: 0.15)
- `VISION_JPEG_QUALITY`: JPEG quality for encoding (1-100, default: 85)
- `FOOD_LABELS`: Comma-separated list of detection labels (defaults defined in `chef_reachy/vision/config.py`)
- `ENABLE_TRACKING`: Enable automatic camera tracking (default: true)
- `TRACKING_KP`: Proportional gain for tracking (default: 1.0, higher = faster response)
- `TRACKING_UPDATE_RATE`: Update rate in seconds (default: 2.5s = ~0.4Hz, matches OWL-ViT inference time)
- `MAX_ROTATION_DEG`: Maximum camera rotation angle (default: 30.0 degrees)
- `TTS_MODEL`: Text-to-speech model (default: `hexgrad/Kokoro-82M`)
- `TTS_VOICE`: Voice to use (default: `af_heart`)
- `TTS_DEVICE`: Device for TTS (`auto`, `cuda`, or `cpu`)
- `TTS_SAMPLE_RATE`: Audio sample rate (default: 16000Hz)

## Usage

1. Run the application:
```bash
uv run ./chef_reachy/main.py
```

2. The OWL-ViT detector and Kokoro-82M TTS will automatically initialize **before the server starts**
   - Watch the console for initialization progress logs
   - First run may take 20-40 seconds to download models (~680MB total)
   - Subsequent runs load from cache (much faster)

3. Open the web interface at `http://0.0.0.0:8042`
   - The web interface automatically connects via WebSocket and starts streaming
   - **No button clicks needed** - detection runs continuously in the background

4. Hold food in your hand (or have Reachy hold food in its gripper)

5. The system will automatically:
   - **Continuously detect** hands holding food using OWL-ViT (every 2.5 seconds)
   - Show **live camera feed** with bounding boxes when food is detected
   - **Automatically track the hand** - robot cameras follow the detected hand position using `look_at_image()`
   - **Speak friendly phrases** like "I found hand holding food" using text-to-speech
   - Display **real-time detection status** with confidence scores and timestamps
   - Log "No food detected" when no hand with food is visible

6. Move the hand around - the cameras will follow it automatically with live updates!

7. Check the console logs to see detection results in real-time

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU processing
- **Recommended**: 16GB RAM, NVIDIA GPU (CUDA)
- **Note**: OWL-ViT is much lighter than SmolVLM2 and runs well on CPU

## Storage Requirements

- ~600MB for OWL-ViT model cache
- ~82MB for Kokoro-82M TTS model
- ~80MB for EasyOCR model cache
- ~1.6GB for Gemma 2B LLM model
- ~100MB temporary space for processing
- **Total**: ~2.5GB for all models

## Performance

**Model Loading:**
- First load: 10-30 seconds (downloading + loading)
- Subsequent loads: 2-5 seconds (from cache)

**Inference Time:**
- NVIDIA GPU (CUDA): 1-2 seconds per image
- CPU: 2-4 seconds per image

**Memory Usage:**
- Model: ~1.5GB RAM
- Peak during inference: ~2-3GB RAM

## API Endpoints

### WebSocket API

- `WS /vision/stream` - Real-time continuous detection streaming
  - **Automatically streams detection results** every 2.5 seconds (matches inference time)
  - **Message format when food detected**:
    ```json
    {
      "status": "detected",
      "detections": [
        {"label": "hand holding food", "score": 0.87, "box": {"xmin": 150, "ymin": 100, "xmax": 350, "ymax": 300}}
      ],
      "annotated_image": "base64_encoded_jpeg_with_bounding_boxes",
      "timestamp": "2025-12-28T10:30:15.123Z"
    }
    ```
  - **Message format when no food detected**:
    ```json
    {
      "status": "no_detection",
      "detections": [],
      "annotated_image": "base64_encoded_jpeg_without_bounding_boxes",
      "timestamp": "2025-12-28T10:30:18.456Z"
    }
    ```
  - **Connection**: Client connects on page load and receives continuous live updates

## Why OWL-ViT over SmolVLM2?

| Feature | OWL-ViT | SmolVLM2 |
|---------|---------|----------|
| **Task** | Object detection | Image captioning |
| **Output** | Bounding boxes + labels | Text description |
| **Model size** | ~600MB | ~4.5GB |
| **Inference time (CPU)** | 2-4 seconds | 5-10 seconds |
| **Memory usage** | ~2-3GB | ~8GB |
| **Use case fit** | Perfect for locating food | General purpose |

## License

See LICENSE file for details.
