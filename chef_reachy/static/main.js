let ws = null;
let isStreaming = false;

function updateInventoryDisplay(inventory) {
    const inventoryList = document.getElementById("inventory-list");
    const inventoryCount = document.getElementById("inventory-count");

    if (!inventory || inventory.length === 0) {
        inventoryList.innerHTML = '<li class="inventory-empty">No items detected yet</li>';
        inventoryCount.textContent = "0";
        return;
    }

    // Update count
    inventoryCount.textContent = inventory.length;

    // Render inventory items (newest first)
    const itemsHtml = inventory
        .slice()
        .reverse()
        .map(item => {
            const detectedDate = new Date(item.detected_at);
            const dateStr = detectedDate.toLocaleString();

            return `
                <li class="inventory-item">
                    <div class="inventory-item-content">
                        <div class="inventory-item-name">${item.product_name}</div>
                        <div class="inventory-item-details">
                            ${item.expiration_date ? `Expires: ${item.expiration_date}` : 'No expiration date'}
                        </div>
                        <div class="inventory-item-details">
                            Detected: ${dateStr}
                        </div>
                    </div>
                </li>
            `;
        })
        .join("");

    inventoryList.innerHTML = itemsHtml;
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/vision/stream`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        isStreaming = true;
        document.getElementById("vision-status").textContent = "Live stream active - continuous food detection running";
        document.getElementById("vision-status").className = "status-indicator success";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const imageContainer = document.getElementById("image-container");
        const annotatedImage = document.getElementById("annotated-image");
        const resultText = document.getElementById("result-text");
        const resultTimestamp = document.getElementById("result-timestamp");
        const statusEl = document.getElementById("vision-status");

        // Handle different message types
        if (data.status === "connected") {
            // Initial connection message
            statusEl.textContent = data.message || "Live stream active - continuous food detection running";
            statusEl.className = "status-indicator success";
            resultText.textContent = "Waiting for first detection...";
            resultText.style.color = "#757575";
            resultText.style.fontWeight = "normal";
        } else if (data.status === "error" || data.status === "camera_error") {
            // Error status
            statusEl.textContent = `Error: ${data.message || "Detection error"}`;
            statusEl.className = "status-indicator error";
            resultText.textContent = data.message || "Detection error occurred";
            resultText.style.color = "#d32f2f"; // red
            resultText.style.fontWeight = "normal";
        } else {
            // Detection results (detected or no_detection)

            // Update image (always show the latest frame)
            if (data.annotated_image) {
                annotatedImage.src = `data:image/jpeg;base64,${data.annotated_image}`;
                imageContainer.style.display = "block";
            }

            // Update detection results based on status
            if (data.status === "detected" && data.detections && data.detections.length > 0) {
                // Food detected!
                const detectionList = data.detections
                    .map(d => `${d.label} (${(d.score * 100).toFixed(1)}%)`)
                    .join(", ");
                resultText.textContent = `✓ Detected: ${detectionList}`;
                resultText.style.color = "#2e7d32"; // green
                resultText.style.fontWeight = "bold";
            } else if (data.status === "no_detection") {
                // No food detected
                resultText.textContent = "No food detected";
                resultText.style.color = "#757575"; // grey
                resultText.style.fontWeight = "normal";
            }

            // Update timestamp
            if (data.timestamp) {
                resultTimestamp.textContent = `Last update: ${new Date(data.timestamp).toLocaleTimeString()}`;
            }

            // Update inventory display if present
            if (data.inventory) {
                updateInventoryDisplay(data.inventory);
            }
        }
    };

    ws.onerror = () => {
        document.getElementById("vision-status").textContent = "Streaming error - reconnecting...";
        document.getElementById("vision-status").className = "status-indicator error";
    };

    ws.onclose = () => {
        isStreaming = false;
        document.getElementById("vision-status").textContent = "Streaming stopped - reconnecting...";
        document.getElementById("vision-status").className = "status-indicator";

        setTimeout(() => {
            if (!isStreaming) {
                connectWebSocket();
            }
        }, 3000);
    };
}

connectWebSocket();

document.getElementById("vision-status").textContent = "Connecting to live stream...";
document.getElementById("vision-status").className = "status-indicator";
