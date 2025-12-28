---
title: Chef Reachy
emoji: 👨‍🍳
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Food identification using SmolVLM2 vision model on Reachy Mini
tags:
 - reachy_mini
 - reachy_mini_python_app
 - computer_vision
 - food_recognition
---

# Chef Reachy

A Reachy Mini application that uses the SmolVLM2 vision model to identify food items from the robot's camera.

## Features

- Real-time food identification using SmolVLM2-2.2B-Instruct model
- Web-based control interface
- On-demand image capture and processing
- Custom prompt support for flexible queries
- Device-aware model loading (CUDA, MPS, or CPU)
- Memory management with model load/unload controls

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

Configuration options:
- `HF_HOME`: Directory for model cache (default: `./cache/huggingface`)
- `HF_TOKEN`: Optional Hugging Face token for gated models
- `LOCAL_VISION_MODEL`: Vision model to use (default: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`)
- `VISION_DEVICE`: Device preference (`auto`, `cuda`, `mps`, or `cpu`)
- `VISION_MAX_TOKENS`: Max tokens for vision response (default: 64)
- `VISION_JPEG_QUALITY`: JPEG quality for encoding (1-100, default: 85)

## Usage

1. Run the application (through the Reachy Mini app system)
   - The vision model will automatically download and initialize **before the server starts**
   - Watch the console for initialization progress logs
   - First run may take 10-30 seconds to download the ~5GB model

2. Open the web interface at `http://0.0.0.0:8042`

3. The status will show "Model ready on [device]" when initialization is complete

4. Position food items in the camera view

5. Click "Capture & Identify Food" to analyze the image

6. (Optional) Enter custom prompts for specific queries

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU processing
- **Recommended**: 16GB RAM, Apple Silicon M1/M2 or NVIDIA GPU
- **Optimal**: 32GB RAM, Apple Silicon M3/M4 or NVIDIA RTX GPU

## Storage Requirements

- ~5GB for model cache
- ~100MB temporary space for processing

## Performance

**Model Loading:**
- First load: 10-30 seconds (downloading + loading)
- Subsequent loads: 3-10 seconds (from cache)

**Inference Time:**
- Apple Silicon (MPS): 2-4 seconds
- NVIDIA GPU (CUDA): 1-2 seconds
- CPU: 5-10 seconds

## API Endpoints

- `POST /vision/capture_and_process` - Capture and process image
- `GET /vision/status` - Get vision system status

Note: The vision model initializes automatically **before** the app server starts. The model stays loaded for the lifetime of the application.

## Design Documentation

See [VISION_DESIGN.md](VISION_DESIGN.md) for detailed architecture and design decisions.

## License

See LICENSE file for details.