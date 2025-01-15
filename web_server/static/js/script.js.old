document.addEventListener("DOMContentLoaded", () => {
    // References
    const chatLog = document.getElementById("chat-log");
    const chatInput = document.getElementById("chat-input");
    const sendButton = document.getElementById("chat-send-btn");
    const fileInput = document.getElementById("file-input");
    const uploadButton = document.getElementById("chat-upload-btn");
    const visualizationButton = document.getElementById("generate-visualization");
    const visualizationOutput = document.getElementById("visualization-output");
    const galleryContainer = document.getElementById("gallery-container");

    // Add Message to Chat Log
    function addMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.textContent = `${sender}: ${message}`;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    // Send Message to MAIA
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        addMessage("You", message);
        chatInput.value = "";

        fetch("/ask_maia", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: message }),
        })
            .then((response) => response.json())
            .then((data) => {
                addMessage("MAIA", data.response || "I didn't understand that.");
            })
            .catch(() => {
                addMessage("MAIA", "Error: Unable to process your message.");
            });
    }

    // Upload Files
    function uploadFiles() {
        const files = Array.from(fileInput.files);
        if (!files.length) {
            alert("No files selected.");
            return;
        }

        files.forEach((file) => {
            const formData = new FormData();
            formData.append("file", file);

            fetch("/upload", { method: "POST", body: formData })
                .then((response) => response.json())
                .then((data) => {
                    addMessage("MAIA", `File "${file.name}" processed: ${data.message}`);
                })
                .catch(() => {
                    addMessage("MAIA", `Error processing file "${file.name}".`);
                });
        });
    }

    // Generate Thought Visualization
    function generateVisualization() {
        visualizationOutput.textContent = "Generating...";
        fetch("/v1/visualize_thoughts")
            .then((response) => response.json())
            .then((data) => {
                visualizationOutput.textContent = JSON.stringify(data.visualization, null, 2);
            })
            .catch(() => {
                visualizationOutput.textContent = "Error: Unable to generate visualization.";
            });
    }

    // Load Gallery Images
    function loadGallery() {
        fetch("/get_gallery_images")
            .then((response) => response.json())
            .then((data) => {
                galleryContainer.innerHTML = "";
                data.images.forEach((filename) => {
                    const img = document.createElement("img");
                    img.src = `static/images/${filename}`;
                    img.alt = filename;
                    galleryContainer.appendChild(img);
                });
            })
            .catch(() => {
                galleryContainer.textContent = "Error loading gallery.";
            });
    }

    // Event Listeners
    sendButton.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendMessage();
    });
    uploadButton.addEventListener("click", uploadFiles);
    visualizationButton.addEventListener("click", generateVisualization);

    // Load Gallery on Page Load
    loadGallery();
});
