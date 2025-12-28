console.log("Chef Reachy UI v6 loaded");

// Vision state
let isProcessing = false;

async function captureAndProcess() {
    if (isProcessing) return;

    const statusEl = document.getElementById("vision-status");
    const resultText = document.getElementById("result-text");
    const resultTimestamp = document.getElementById("result-timestamp");
    const processingIndicator = document.getElementById("processing-indicator");
    const imageContainer = document.getElementById("image-container");
    const capturedImage = document.getElementById("captured-image");
    const captureBtn = document.getElementById("capture-btn");

    isProcessing = true;
    captureBtn.disabled = true;

    try {
        // Show processing state
        processingIndicator.classList.add("active");
        statusEl.textContent = "Processing image...";
        statusEl.className = "status-indicator processing";

        const prompt = document.getElementById("custom-prompt").value.trim();
        const body = prompt ? JSON.stringify({ prompt }) : JSON.stringify({});

        const resp = await fetch("/vision/capture_and_process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: body,
        });

        const data = await resp.json();

        if (data.status === "success") {
            // Display the captured image
            if (data.image) {
                capturedImage.src = `data:image/jpeg;base64,${data.image}`;
                imageContainer.style.display = "block";
            }

            // Display the description
            resultText.textContent = data.description;
            resultTimestamp.textContent = `Captured at: ${new Date(data.timestamp).toLocaleString()}`;
            statusEl.textContent = "Processing complete!";
            statusEl.className = "status-indicator success";
            console.log("Vision result:", data.description);

            // Reset status after a few seconds
            setTimeout(() => {
                statusEl.textContent = "Ready";
                statusEl.className = "status-indicator success";
            }, 3000);
        } else if (data.status === "busy") {
            statusEl.textContent = "Already processing, please wait...";
            statusEl.className = "status-indicator processing";
        } else {
            resultText.textContent = `Error: ${data.message || "Unknown error"}`;
            resultTimestamp.textContent = "";
            statusEl.textContent = "Processing failed";
            statusEl.className = "status-indicator error";
        }
    } catch (e) {
        console.error("Vision processing failed:", e);
        resultText.textContent = `Error: ${e.message}`;
        resultTimestamp.textContent = "";
        statusEl.textContent = "Connection error";
        statusEl.className = "status-indicator error";
    } finally {
        isProcessing = false;
        processingIndicator.classList.remove("active");
        captureBtn.disabled = false;
    }
}

// Event listener for capture button
document.getElementById("capture-btn").addEventListener("click", () => {
    captureAndProcess();
});

// Allow Enter key in prompt input to trigger capture
document.getElementById("custom-prompt").addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !isProcessing) {
        captureAndProcess();
    }
});

// Set initial status
document.getElementById("vision-status").textContent = "Ready";
document.getElementById("vision-status").className = "status-indicator success";
