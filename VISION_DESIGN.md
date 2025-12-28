# Chef Reachy Vision Processing Design

## Overview

This document describes the design for integrating SmolVLM2 vision model into the Chef Reachy application to enable image recognition and food identification capabilities.

## Current Project Structure

```
chef_reachy/
├── chef_reachy/
│   ├── __init__.py
│   ├── main.py              # ReachyMiniApp with settings server
│   └── static/
│       ├── index.html       # Frontend UI
│       ├── main.js          # Frontend controls
│       └── style.css
├── pyproject.toml
└── README.md
```

### Current Capabilities
- Custom settings app server (FastAPI) at http://0.0.0.0:8042
- Control endpoints for antennas and sound playback
- Simple HTML/JS frontend for user controls

## Reference Implementation

The `reachy_mini_conversation_app` provides a robust implementation of SmolVLM2 integration:

**Key Components:**
- `VisionProcessor`: Handles model loading and inference
- `VisionManager`: Manages periodic vision processing with threading
- `CameraWorker`: Thread-safe camera frame capture and buffering
- `VisionConfig`: Configuration dataclass for all vision settings

**Key Features:**
- Device-aware model loading (CUDA, MPS, CPU)
- Retry logic for robustness
- Thread-safe frame access
- Base64 image encoding for model input
- Automatic cache management
- Memory cleanup after inference

## Proposed Architecture for Chef Reachy

### 1. Module Structure

```
chef_reachy/
├── chef_reachy/
│   ├── __init__.py
│   ├── main.py                    # Updated ChefReachy app
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── config.py              # VisionConfig dataclass
│   │   ├── processor.py           # VisionProcessor class
│   │   └── camera.py              # CameraCapture class
│   └── static/
│       ├── index.html             # Updated UI with camera controls
│       ├── main.js                # Updated with vision controls
│       └── style.css
├── pyproject.toml                 # Updated dependencies
├── .env.example                   # Environment config template
└── VISION_DESIGN.md              # This document
```

### 2. Component Design

#### 2.1 VisionConfig (`chef_reachy/vision/config.py`)

```python
from dataclasses import dataclass

@dataclass
class VisionConfig:
    """Configuration for vision processing."""
    model_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    cache_dir: str = "./cache/huggingface"
    max_new_tokens: int = 64
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "auto"  # "auto", "cuda", "mps", "cpu"
    default_prompt: str = "Identify the food items in this image. List what you see."
```

**Rationale:**
- Centralized configuration makes it easy to adjust settings
- Sensible defaults based on reference implementation
- Food-specific default prompt for chef use case

#### 2.2 VisionProcessor (`chef_reachy/vision/processor.py`)

Based on the reference implementation but simplified for single-shot image processing:

```python
class VisionProcessor:
    """Handles SmolVLM2 model loading and inference."""

    def __init__(self, vision_config: VisionConfig | None = None)
    def initialize(self) -> bool
    def process_image(self, cv2_image: NDArray[np.uint8], prompt: str | None = None) -> str
    def get_model_info(self) -> Dict[str, Any]
    def cleanup(self) -> None
```

**Key Features:**
- Device auto-detection (prefer MPS on Apple Silicon, CUDA on NVIDIA, fallback to CPU)
- Lazy initialization (model loaded on first use)
- Retry logic with exponential backoff
- Memory cleanup after processing
- Thread-safe operation

**Differences from Reference:**
- No periodic processing (on-demand only)
- No threading (single-shot operation)
- Simpler interface focused on manual capture

#### 2.3 CameraCapture (`chef_reachy/vision/camera.py`)

```python
class CameraCapture:
    """Handles camera frame capture from ReachyMini."""

    def __init__(self, reachy_mini: ReachyMini)
    def capture_frame(self) -> NDArray[np.uint8] | None
    def get_camera_status(self) -> Dict[str, Any]
```

**Rationale:**
- Simplified wrapper around reachy_mini.media.get_frame()
- No threading needed (capture on demand)
- Status info for debugging

#### 2.4 Updated ChefReachy App (`chef_reachy/main.py`)

**State Management:**
```python
class ChefReachy(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"

    # Vision components
    vision_processor: VisionProcessor | None = None
    camera_capture: CameraCapture | None = None

    # Processing state
    last_image_description: str = ""
    is_processing: bool = False
```

**New API Endpoints:**

1. **POST /vision/capture_and_process**
   - Capture current frame and process with SmolVLM2
   - Request body: `{"prompt": "optional custom prompt"}`
   - Returns: `{"description": "...", "timestamp": "..."}`

2. **GET /vision/status**
   - Get current vision system status
   - Returns: `{"ready": bool, "error": str | null, "processing": bool, "last_result": "...", "model_info": {...}}`

**Note:** The vision model initializes in `__init__()` **BEFORE** the server starts. All initialization logs are sent to console output. The model stays loaded for the application lifetime.

**Integration Pattern:**
```python
class ChefReachy(ReachyMiniApp):
    def __init__(self):
        super().__init__()

        # Initialize vision model BEFORE server starts
        logger.info("INITIALIZING VISION MODEL")
        vision_config = VisionConfig()
        self.vision_processor = VisionProcessor(vision_config)
        self.vision_ready = self.vision_processor.initialize()

        if self.vision_ready:
            logger.info("✓ VISION MODEL READY!")
        else:
            logger.error("✗ VISION INITIALIZATION FAILED")

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # Initialize camera (fast)
        camera_capture = CameraCapture(reachy_mini)

    # Register endpoints
    @self.settings_app.post("/vision/capture_and_process")
    async def capture_and_process(request: ProcessRequest):
        # Implementation
        pass

    # ... existing control loop
```

### 3. Frontend Design

#### 3.1 UI Updates (`chef_reachy/static/index.html`)

Add new section for vision controls:

```html
<div id="vision-controls">
    <h2>Food Vision</h2>

    <!-- Status indicator -->
    <div id="vision-status" class="status-indicator">
        Model: Not initialized
    </div>

    <!-- Control buttons -->
    <button id="init-vision-btn">Initialize Vision Model</button>
    <button id="capture-btn" disabled>Capture & Identify Food</button>
    <button id="cleanup-vision-btn" disabled>Unload Model</button>

    <!-- Custom prompt input -->
    <div id="prompt-section">
        <label>Custom Prompt (optional):</label>
        <input type="text" id="custom-prompt"
               placeholder="Identify the food items...">
    </div>

    <!-- Results display -->
    <div id="vision-results">
        <h3>Latest Identification:</h3>
        <p id="result-text">No results yet</p>
        <small id="result-timestamp"></small>
    </div>

    <!-- Loading indicator -->
    <div id="processing-indicator" style="display: none;">
        Processing image...
    </div>
</div>
```

#### 3.2 JavaScript Updates (`chef_reachy/static/main.js`)

Add vision control functions:

```javascript
let visionInitialized = false;
let isProcessing = false;

async function initializeVision() {
    try {
        const resp = await fetch("/vision/initialize", { method: "POST" });
        const data = await resp.json();
        visionInitialized = true;
        updateVisionUI();
        console.log("Vision initialized:", data);
    } catch (e) {
        console.error("Failed to initialize vision:", e);
    }
}

async function captureAndProcess() {
    if (isProcessing) return;

    isProcessing = true;
    updateVisionUI();

    try {
        const prompt = document.getElementById("custom-prompt").value;
        const body = prompt ? JSON.stringify({ prompt }) : JSON.stringify({});

        const resp = await fetch("/vision/capture_and_process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: body
        });

        const data = await resp.json();
        displayResults(data);
    } catch (e) {
        console.error("Vision processing failed:", e);
    } finally {
        isProcessing = false;
        updateVisionUI();
    }
}

function updateVisionUI() {
    document.getElementById("capture-btn").disabled = !visionInitialized || isProcessing;
    document.getElementById("cleanup-vision-btn").disabled = !visionInitialized;
    document.getElementById("processing-indicator").style.display =
        isProcessing ? "block" : "none";
}
```

### 4. Dependencies

Update `pyproject.toml`:

```toml
[project]
dependencies = [
    "reachy-mini[gstreamer]>=1.2.4",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "opencv-python>=4.8.0",
    "huggingface-hub>=0.20.0",
    "numpy>=1.24.0",
    "python-dotenv>=1.0.0",
]
```

### 5. Configuration

Create `.env.example`:

```bash
# Hugging Face Configuration
HF_HOME=./cache/huggingface
HF_TOKEN=  # Optional, only needed for gated models
LOCAL_VISION_MODEL=HuggingFaceTB/SmolVLM2-2.2B-Instruct

# Vision Processing Configuration
VISION_DEVICE=auto  # auto, cuda, mps, cpu
VISION_MAX_TOKENS=64
VISION_JPEG_QUALITY=85
```

### 6. Implementation Flow

#### 6.1 First-Time Setup

1. User opens web interface
2. Clicks "Initialize Vision Model"
3. Frontend calls `/vision/initialize`
4. Backend:
   - Downloads model to cache (if not present)
   - Loads model onto appropriate device
   - Returns status
5. Frontend enables "Capture & Identify Food" button

#### 6.2 Food Identification

1. User positions food item in camera view
2. Clicks "Capture & Identify Food"
3. Frontend:
   - Shows processing indicator
   - Calls `/vision/capture_and_process` with optional custom prompt
4. Backend:
   - Captures current camera frame
   - Processes with SmolVLM2
   - Returns food identification
5. Frontend displays results

#### 6.3 Memory Management

1. User clicks "Unload Model" when done
2. Frontend calls `/vision/cleanup`
3. Backend:
   - Releases model from GPU/memory
   - Clears cache
4. Frontend resets to uninitialized state

### 7. Error Handling

**Model Loading Failures:**
- Display clear error message
- Suggest checking HF_TOKEN if model is gated
- Fall back to CPU if GPU unavailable

**Capture Failures:**
- Retry logic (up to 3 attempts)
- Display error if camera not accessible
- Suggest checking robot connection

**Processing Failures:**
- Show user-friendly error message
- Log detailed error for debugging
- Don't crash the app

### 8. Performance Considerations

**Model Loading:**
- First load takes 10-30 seconds (downloading + loading)
- Subsequent loads are faster (cached)
- MPS (Apple Silicon): ~5-10 seconds
- CUDA: ~3-5 seconds
- CPU: ~10-20 seconds

**Inference Time:**
- MPS: ~2-4 seconds per image
- CUDA: ~1-2 seconds per image
- CPU: ~5-10 seconds per image

**Memory Usage:**
- Model size: ~4.5GB
- Peak memory during inference: ~6GB
- Recommend 8GB+ RAM

### 9. Testing Plan

**Unit Tests:**
- VisionConfig initialization
- VisionProcessor device selection
- CameraCapture frame retrieval
- Error handling in processor

**Integration Tests:**
- End-to-end capture and process flow
- API endpoint responses
- Frontend-backend communication

**Manual Tests:**
- Test with various food items
- Test with different prompts
- Test model unload/reload cycle
- Test on different devices (Mac, Linux with CUDA, CPU-only)

### 10. Future Enhancements

**Phase 2:**
- Stream camera view to frontend
- Show live camera feed with overlay
- Display bounding boxes for detected items

**Phase 3:**
- Store identification history
- Recipe suggestions based on identified ingredients
- Multi-language support for descriptions

**Phase 4:**
- Fine-tune model on food-specific dataset
- Add nutritional information lookup
- Integration with recipe databases

## Migration from Reference Implementation

### What to Keep:
1. VisionProcessor architecture (proven robust)
2. Device selection logic (works across platforms)
3. Retry mechanism (handles transient failures)
4. Memory cleanup patterns (prevents OOM)
5. Base64 encoding approach (compatible with transformers)

### What to Simplify:
1. No VisionManager (no periodic processing)
2. No threading in vision processing (on-demand only)
3. Simpler camera interface (no face tracking)
4. Remove interpolation logic (not needed)
5. Single-shot processing instead of continuous

### What to Add:
1. Frontend UI for manual control
2. API endpoints for vision operations
3. Status tracking in main app
4. Custom prompt support
5. Lazy model initialization

## Security Considerations

1. **Input Validation:**
   - Sanitize custom prompts
   - Validate image dimensions
   - Limit prompt length

2. **Resource Limits:**
   - Rate limit API calls
   - Timeout long-running requests
   - Prevent concurrent processing

3. **Data Privacy:**
   - Don't store captured images by default
   - Clear image data after processing
   - No external API calls (fully local)

## Deployment Notes

**Hardware Requirements:**
- Minimum: 8GB RAM, CPU-only processing
- Recommended: 16GB RAM, Apple Silicon M1/M2 or NVIDIA GPU
- Optimal: 32GB RAM, Apple Silicon M3/M4 or NVIDIA RTX GPU

**Storage:**
- ~5GB for model cache
- Temporary space for processing (~100MB)

**Network:**
- Initial setup requires internet (model download)
- Offline operation after first download

## Conclusion

This design provides a clean, maintainable integration of SmolVLM2 vision processing into Chef Reachy while:
- Learning from the proven patterns in reachy_mini_conversation_app
- Simplifying to match the use case (on-demand vs continuous)
- Maintaining separation of concerns
- Providing good user experience
- Being resource-conscious

The implementation can be done incrementally:
1. Backend components (vision module)
2. API endpoints
3. Frontend controls
4. Testing and refinement
