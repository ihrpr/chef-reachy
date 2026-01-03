// Chef Reachy - Frontend JavaScript

let ws = null;
let isConnected = false;

// State for streaming response
let currentStreamingEvent = null;

// DOM Elements
const connectionStatus = document.getElementById("connection-status");
const statusIndicator = document.getElementById("status-indicator");
const statusText = document.getElementById("status-text");
const transcriptionText = document.getElementById("transcription-text");
const videoFrame = document.getElementById("video-frame");
const videoPlaceholder = document.getElementById("video-placeholder");
const eventFeed = document.getElementById("event-feed");
const inventoryGrid = document.getElementById("inventory-grid");
const inventoryCount = document.getElementById("inventory-count");

// Event icons
const EVENT_ICONS = {
    user_speech: "🎤",
    agent_response: "🤖",
    tool_start: "⚙️",
    tool_progress: "⏳",
    tool_image: "📷",
    tool_result: "✓",
    tool_error: "✗",
    status: "ℹ️",
    error: "⚠️",
    connected: "🔗",
    inventory_update: "📦",
    speech_status: "🎙️"
};

// Format timestamp
function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
}

// Add event to feed
function addEvent(type, content, imageUrl = null, timestamp = null) {
    // Remove empty placeholder if present
    const placeholder = eventFeed.querySelector(".feed-empty");
    if (placeholder) {
        placeholder.remove();
    }

    const eventItem = document.createElement("div");
    eventItem.className = `event-item ${type.replace("_", "-")}`;

    const icon = EVENT_ICONS[type] || "•";
    const time = timestamp ? formatTime(timestamp) : new Date().toLocaleTimeString();

    let html = `
        <div class="event-header">
            <span class="event-icon">${icon}</span>
            <span class="event-type">${type.replace("_", " ")}</span>
            <span class="event-time">${time}</span>
        </div>
        <div class="event-content">${content}</div>
    `;

    if (imageUrl) {
        html += `<img class="event-image" src="${imageUrl}" alt="Captured image">`;
    }

    eventItem.innerHTML = html;
    eventFeed.appendChild(eventItem);

    // Scroll to bottom
    eventFeed.scrollTop = eventFeed.scrollHeight;

    return eventItem;
}

// Start or get streaming event for agent response
function getOrCreateStreamingEvent(timestamp) {
    if (!currentStreamingEvent) {
        // Remove empty placeholder if present
        const placeholder = eventFeed.querySelector(".feed-empty");
        if (placeholder) {
            placeholder.remove();
        }

        const eventItem = document.createElement("div");
        eventItem.className = "event-item agent-response streaming";

        const icon = EVENT_ICONS.agent_response || "🤖";
        const time = timestamp ? formatTime(timestamp) : new Date().toLocaleTimeString();

        eventItem.innerHTML = `
            <div class="event-header">
                <span class="event-icon">${icon}</span>
                <span class="event-type">agent response</span>
                <span class="event-time">${time}</span>
            </div>
            <div class="event-content"></div>
        `;

        eventFeed.appendChild(eventItem);
        currentStreamingEvent = eventItem;
    }
    return currentStreamingEvent;
}

// Append text to streaming event
function appendToStreamingEvent(delta, timestamp) {
    const eventItem = getOrCreateStreamingEvent(timestamp);
    const contentDiv = eventItem.querySelector(".event-content");
    contentDiv.textContent += delta;

    // Scroll to bottom
    eventFeed.scrollTop = eventFeed.scrollHeight;
}

// Finalize streaming event
function finalizeStreamingEvent() {
    if (currentStreamingEvent) {
        currentStreamingEvent.classList.remove("streaming");
        currentStreamingEvent = null;
    }
}

// Update status
function updateStatus(status, message) {
    statusIndicator.className = `status-indicator ${status}`;
    statusText.textContent = message;
}

// Update inventory display
function updateInventory(items) {
    inventoryCount.textContent = items.length;

    if (!items || items.length === 0) {
        inventoryGrid.innerHTML = '<div class="inventory-empty">No items in inventory</div>';
        return;
    }

    const cardsHtml = items.map(item => {
        let expiryClass = "";
        let expiryText = "No expiration date";

        if (item.expiration_date) {
            const expDate = new Date(item.expiration_date);
            const now = new Date();
            const daysUntil = Math.ceil((expDate - now) / (1000 * 60 * 60 * 24));

            if (daysUntil < 0) {
                expiryClass = "expired";
                expiryText = `Expired ${Math.abs(daysUntil)} days ago`;
            } else if (daysUntil <= 3) {
                expiryClass = "soon";
                expiryText = `Expires in ${daysUntil} days`;
            } else {
                expiryText = `Expires: ${item.expiration_date}`;
            }
        }

        return `
            <div class="inventory-card">
                <div class="inventory-card-name">${item.product_name}</div>
                <div class="inventory-card-expiry ${expiryClass}">${expiryText}</div>
            </div>
        `;
    }).join("");

    inventoryGrid.innerHTML = cardsHtml;
}

// Update video frame
function updateVideoFrame(imageBase64) {
    if (imageBase64) {
        videoFrame.src = `data:image/jpeg;base64,${imageBase64}`;
        videoFrame.style.display = "block";
        videoPlaceholder.style.display = "none";
    }
}

// Handle WebSocket message
function handleMessage(data) {
    switch (data.type) {
        case "connected":
            updateStatus(data.status || "idle", data.message || "Connected");
            addEvent("connected", data.message || "Connected to Chef Reachy", null, data.timestamp);
            break;

        case "video_frame":
            updateVideoFrame(data.image);
            break;

        case "status":
            updateStatus(data.status, data.message);
            // Finalize streaming when going back to listening/idle
            if (data.status === "listening" || data.status === "idle") {
                finalizeStreamingEvent();
                addEvent("status", data.message, null, data.timestamp);
            }
            break;

        case "speech_status":
            // Update speaking indicator
            if (data.is_speaking) {
                statusIndicator.classList.add("speaking");
                updateStatus("speaking", "Speaking...");
            } else {
                statusIndicator.classList.remove("speaking");
            }
            break;

        case "user_speech":
            // Finalize any previous streaming response before showing user speech
            finalizeStreamingEvent();
            addEvent("user_speech", `"${data.text}"`, null, data.timestamp);
            break;

        case "agent_response_delta":
            // Stream text deltas in real-time
            appendToStreamingEvent(data.delta, data.timestamp);
            break;

        case "agent_response":
            // Final complete response - finalize the streaming event
            finalizeStreamingEvent();
            // Don't add duplicate - streaming already showed the text
            break;

        case "tool_start":
            // Finalize streaming text before tool execution
            finalizeStreamingEvent();
            updateStatus("executing", `Executing: ${data.tool_name}`);
            addEvent("tool_start", `Starting tool: ${data.tool_name}`, null, data.timestamp);
            break;

        case "tool_progress":
            addEvent("tool_progress", data.message, null, data.timestamp);
            break;

        case "tool_image":
            // Update main video with the captured image
            updateVideoFrame(data.image);
            // Add to event feed with thumbnail
            addEvent("tool_image", data.message, `data:image/jpeg;base64,${data.image}`, data.timestamp);
            break;

        case "tool_result":
            if (data.status === "success") {
                addEvent("tool_result", data.message, null, data.timestamp);
            }
            updateStatus("listening", "Listening...");
            break;

        case "tool_error":
            addEvent("tool_error", data.message, null, data.timestamp);
            updateStatus("error", data.message);
            break;

        case "error":
            addEvent("error", data.message, null, data.timestamp);
            updateStatus("error", data.message);
            break;

        case "inventory_update":
            updateInventory(data.items);
            break;

        case "whisper_transcription":
            // Show the raw Whisper transcription in the left panel
            console.log("Whisper transcription received:", data.text);
            transcriptionText.textContent = data.text ? `"${data.text}"` : "(empty)";
            break;
    }
}

// Connect WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/vision/stream`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        isConnected = true;
        connectionStatus.textContent = "Connected";
        connectionStatus.className = "connection-status connected";
        connectionStatus.style.display = "block";
        updateStatus("idle", "Say 'Claude' to start a conversation");

        // Hide connection status after 2 seconds
        setTimeout(() => {
            connectionStatus.style.display = "none";
        }, 2000);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch (e) {
            console.error("Failed to parse message:", e);
        }
    };

    ws.onerror = () => {
        connectionStatus.textContent = "Connection error";
        connectionStatus.className = "connection-status";
        connectionStatus.style.display = "block";
        updateStatus("error", "Connection error");
    };

    ws.onclose = () => {
        isConnected = false;
        connectionStatus.textContent = "Disconnected - Reconnecting...";
        connectionStatus.className = "connection-status";
        connectionStatus.style.display = "block";
        updateStatus("idle", "Disconnected - Reconnecting...");

        // Reconnect after 3 seconds
        setTimeout(() => {
            if (!isConnected) {
                connectWebSocket();
            }
        }, 3000);
    };
}

// Initialize
connectWebSocket();

// Fetch initial inventory
fetch("/api/inventory")
    .then(response => response.json())
    .then(data => {
        if (data.items) {
            updateInventory(data.items);
        }
    })
    .catch(e => console.error("Failed to fetch inventory:", e));
