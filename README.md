---
title: Chef Reachy
emoji: 👨‍🍳
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Voice-activated food inventory assistant with Claude Agent SDK
tags:
 - reachy_mini
 - reachy_mini_python_app
 - voice_assistant
 - claude_agent
 - food_inventory
---

# Chef Reachy

A voice-activated food inventory assistant for Reachy Mini using **Claude Agent SDK**, **local Whisper**, and **Claude Vision API**.

## Features

- **Voice activation** - Say "Claude" to start a conversation
- **Natural conversation** - Multi-turn dialogue with stateful context
- **Multi-angle capture** - Captures 3 different angles of food packaging
- **Claude Vision analysis** - Extracts product name and expiration date from images
- **Inventory management** - Tracks food items with expiration dates
- **Local speech-to-text** - Whisper running on-device (no cloud STT costs)
- **Text-to-speech** - Kokoro-82M for natural voice responses
- **WebSocket streaming** - Real-time updates to web interface
- **Persistent storage** - Saves inventory to ~/.chef_reachy/inventory.json

## Architecture

```
Audio (Reachy mic) → Local Whisper → Wake word "Claude" → Claude Agent SDK
                                                              ↓
                                        [Tools: scan_food, get_inventory, remove_item]
                                                              ↓
                                        Camera captures → Claude Vision processes
                                                              ↓
                                           Inventory DB ← Response → Kokoro-82M TTS → Audio out
```

### Components

1. **Whisper STT** (`chef_reachy/audio/whisper.py`) - Local speech-to-text using faster-whisper
2. **Claude Agent** (`chef_reachy/agent/`) - Claude Agent SDK with custom tools for inventory management
3. **Inventory** (`chef_reachy/inventory/`) - Persistent food item tracking
4. **Kokoro TTS** (`chef_reachy/audio/tts.py`) - Text-to-speech for voice responses

## Installation

### Prerequisites

- Python 3.12+
- Reachy Mini robot
- Anthropic API key (for Claude)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chef_reachy.git
cd chef_reachy
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

4. Run the app:
```bash
uv run reachy-mini-apps run
```

## Usage

### Starting a conversation

1. Say **"Claude"** to activate the assistant
2. Ask questions or give commands naturally:
   - "What's in my inventory?"
   - "Add this item" (then show the food packaging to the camera)
   - "Remove the milk from inventory"
   - "Clear the inventory"

### Example conversations

```
You: "Claude, what do I have in my fridge?"
Reachy: "You currently have 3 items: Organic Milk expiring on February 15th,
         Greek Yogurt expiring on February 20th, and Cheddar Cheese expiring
         on March 1st."

You: "Add this item"
Reachy: "I'll scan this item for you. Please hold it steady while I take
         pictures from different angles."
         [Captures 3 images]
         "Added Organic Eggs expiring on February 25th to your inventory."
```

## Configuration

### Whisper Model

Edit `chef_reachy/main.py` to change Whisper model size:

```python
whisper_config = WhisperConfig(
    model_size="base",  # Options: tiny, base, small, medium, large
    device="cpu",
    compute_type="int8"  # Options: int8, float16, float32
)
```

Smaller models are faster but less accurate. Recommended:
- **tiny** - Fastest, good for simple speech (~1GB RAM)
- **base** - Balanced speed/accuracy (~1.5GB RAM) **[Default]**
- **small** - Better accuracy (~2GB RAM)

### Claude Agent

Edit `chef_reachy/agent/config.py`:

```python
@dataclass
class AgentConfig:
    model: str = "claude-3-5-sonnet-20241022"  # Claude model
    max_tokens: int = 1024
    temperature: float = 0.7
```

## Tools Available to Claude

The assistant has these tools:

1. **scan_food_item** - Capture and analyze food packaging
   - Takes 3 photos at different angles (3 seconds apart)
   - Sends images to Claude Vision API
   - Extracts product name and expiration date
   - Adds item to inventory

2. **get_inventory** - Retrieve all items
   - Returns product names, expiration dates, and expired status

3. **remove_item** - Remove item by name
   - Removes first matching item from inventory

4. **clear_inventory** - Clear all items
   - Empties the entire inventory

## Performance

- **Latency**: ~3-5 seconds per interaction
  - Whisper transcription: ~1-2s
  - Claude API: ~2-3s
  - TTS: ~500ms

- **Memory**: ~2-3GB total
  - Whisper base: ~1.5GB
  - Kokoro TTS: ~100MB
  - Application: ~500MB

- **Cost**: ~$5-10/month for typical use
  - Claude API: ~$0.003 per request (text)
  - Claude Vision: ~$0.005 per image analysis
  - No STT costs (local Whisper)
  - No TTS costs (local Kokoro)

## Development

### Running tests

```bash
# Install dev dependencies
uv sync --group dev

# Run type checking
uv run pyright

# Run linting
uv run ruff check
uv run ruff format
```

### Project structure

```
chef_reachy/
├── agent/              # Claude Agent SDK integration
│   ├── config.py       # Agent configuration
│   └── tools.py        # Custom tools (scan, inventory, etc.)
├── audio/              # Speech processing
│   ├── whisper.py      # Whisper STT
│   └── tts.py          # Kokoro TTS
├── inventory/          # Inventory management
│   ├── models.py       # FoodItem model
│   └── manager.py      # InventoryManager
├── static/             # Web UI assets
└── main.py             # Main application
```

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

Create a `.env` file with your API key:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Whisper model download fails

The first run downloads the Whisper model (~300MB for base). Ensure you have:
- Internet connection
- Sufficient disk space (~1GB)
- Write access to `~/.cache/huggingface/`

### Audio not being captured

Check that:
- Reachy's microphone is working
- `media_backend="default"` is set in main.py
- No other app is using the microphone

## License

This project uses third-party models with their own licenses:

- **Whisper** - MIT License (OpenAI)
- **Kokoro-82M** - Apache 2.0 License
- **Claude API** - Anthropic Terms of Service
- **faster-whisper** - MIT License

## Credits

Built with:
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy-mini)
