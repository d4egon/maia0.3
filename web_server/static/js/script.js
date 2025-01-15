document.addEventListener("DOMContentLoaded", () => {
    // References
    const chatLog = document.getElementById("chat-log");
    const chatInput = document.getElementById("chat-input");
    const sendButton = document.getElementById("chat-send-btn");
    const fileInput = document.getElementById("file-input");
    const uploadButton = document.getElementById("chat-upload-btn");
    let mediaRecorder;
    let audioChunks = [];
    const recordButton = document.getElementById("record-button");
    const recordingStatus = document.getElementById("recording-status");
    //const visualizationButton = document.getElementById("generate-visualization");
    //const visualizationOutput = document.getElementById("visualization-output");
    const galleryContainer = document.getElementById("gallery-container");
    const feedbackButtons = document.getElementById("feedback-buttons");
    const goodButton = document.getElementById("good-btn");
    const badButton = document.getElementById("bad-btn");
    const feedbackInput = document.getElementById("feedback-input");
    const feedbackSubmit = document.getElementById("feedback-submit-btn");
    const logDisplay = document.getElementById("log-display");
    const uploadProgress = document.getElementById("upload-progress");
    const fileUploadProgress = document.getElementById("file-upload-progress");
    const fileUploadStatus = document.getElementById("file-upload-status");
    const refreshGalleryBtn = document.getElementById("refresh-gallery-btn");

    recordButton.addEventListener("click", async () => {
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
        try {
            // Request access to the user's microphone
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            // Start recording
            mediaRecorder.start();
            recordingStatus.style.display = "block";
            recordingStatus.textContent = "Recording...";

            audioChunks = [];
            mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
            };

            // Stop recording when the user clicks the button again
            recordButton.textContent = "🛑 Stop";
            recordButton.onclick = stopRecording;
        } catch (error) {
            console.error("Error accessing microphone:", error);
            alert("Microphone access is required for recording.");
        }
        }
    });

    async function stopRecording() {
        // Stop the media recorder
        mediaRecorder.stop();
        recordingStatus.textContent = "Processing...";
        recordingStatus.style.display = "block";

        mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        // Send audio to the backend
        try {
            const response = await fetch("/upload-audio", {
            method: "POST",
            body: formData,
            });
            const result = await response.json();
            alert(`Transcription: ${result.text}`);
        } catch (error) {
            console.error("Error uploading audio:", error);
            alert("Failed to upload the audio.");
        } finally {
            recordingStatus.style.display = "none";
            recordButton.textContent = "🎙️ Record";
            recordButton.onclick = startRecording;
        }
        };
    }

    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Function to show feedback buttons
    function showFeedbackButtons() {
        feedbackButtons.style.display = 'block';
    }

    // Function to hide feedback buttons
    function hideFeedbackButtons() {
        feedbackButtons.style.display = 'none';
    }

    // Add Message to Chat Log
    function addMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.textContent = `${sender}: ${message}`;
        messageDiv.classList.add(sender.toLowerCase() === 'you' ? 'user-message' : 'maia-message');
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;

        // Show feedback buttons only for MAIA's messages
        if (sender === 'MAIA') {
            showFeedbackButtons();
        }
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
                if (data.log) {
                    logDisplay.textContent = data.log;
                }
            })
            .catch((error) => {
                addMessage("MAIA", "Error: Unable to process your message.");
                console.error('Error:', error);
                logDisplay.textContent = `Error in processing: ${error.message}`;
            });
    }

    // Send Feedback
    function sendFeedback(feedbackType) {
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ feedback: feedbackType }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback sent:', data);
            hideFeedbackButtons();
        })
        .catch(error => console.error('Error sending feedback:', error));
    }

    // Upload Files with Progress
    function uploadFiles() {
        const files = Array.from(fileInput.files);
        if (!files.length) {
            alert("No files selected.");
            return;
        }

        uploadProgress.style.display = "block";
        files.forEach((file, index) => {
            const formData = new FormData();
            formData.append("file", file);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload', true);

            xhr.upload.addEventListener("progress", function(evt) {
                if (evt.lengthComputable) {
                    const percentComplete = Math.round((evt.loaded / evt.total) * 100);
                    fileUploadProgress.value = percentComplete;
                    fileUploadStatus.textContent = `Uploading ${file.name}: ${percentComplete}%`;
                }
            }, false);

            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4) {
                    if (xhr.status === 200) {
                        const data = JSON.parse(xhr.responseText);
                        addMessage("MAIA", `File "${file.name}" processed: ${data.message}`);
                        if (index === files.length - 1) {
                            uploadProgress.style.display = "none";
                        }
                    } else {
                        addMessage("MAIA", `Error processing file "${file.name}".`);
                        console.error(xhr.statusText);
                        if (index === files.length - 1) {
                            uploadProgress.style.display = "none";
                        }
                    }
                }
            };

            xhr.send(formData);
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
        .catch((error) => {
            visualizationOutput.textContent = `Error: Unable to generate visualization. ${error.message}`;
            console.error('Visualization Error:', error);
        });
}

// Load and Refresh Gallery Images
function loadGallery() {
    fetch("/get_gallery_images")
        .then((response) => response.json())
        .then((data) => {
            galleryContainer.innerHTML = "";
            data.images.forEach((filename) => {
                const img = document.createElement("img");
                img.src = `static/images/${filename}`;
                img.alt = filename;
                img.title = filename; // Add title for hover info
                img.addEventListener('click', () => {
                    // Simple lightbox functionality for larger view
                    const lightbox = document.createElement('div');
                    lightbox.className = 'lightbox';
                    lightbox.innerHTML = `<img src="${img.src}" alt="${img.alt}"><span class="close">×</span>`;
                    document.body.appendChild(lightbox);
                    lightbox.querySelector('.close').addEventListener('click', () => {
                        document.body.removeChild(lightbox);
                    });
                });
                galleryContainer.appendChild(img);
            });
        })
        .catch((error) => {
            galleryContainer.textContent = `Error loading gallery: ${error.message}`;
            console.error('Gallery Load Error:', error);
        });
}

// Submit Feedback to Neo4j
function submitFeedback() {
    const feedback = feedbackInput.value.trim();
    if (!feedback) {
        alert("Please enter feedback before submitting.");
        return;
    }

    fetch("/v1/submit_feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback: feedback }),
    })
        .then((response) => response.json())
        .then((data) => {
            alert(data.message || "Feedback submitted successfully!");
            feedbackInput.value = "";
        })
        .catch((error) => {
            console.error('Error:', error);
            alert("Error submitting feedback. Please try again.");
        });
}

// Event Listeners
sendButton.addEventListener("click", debounce(sendMessage, 500));
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});
uploadButton.addEventListener("click", debounce(uploadFiles, 500));
visualizationButton.addEventListener("click", debounce(generateVisualization, 500));
feedbackSubmit.addEventListener("click", debounce(submitFeedback, 500));
goodButton.addEventListener("click", () => sendFeedback('good'));
badButton.addEventListener("click", () => sendFeedback('bad'));
refreshGalleryBtn.addEventListener("click", debounce(loadGallery, 500));

// Load Gallery on Page Load
loadGallery();

// Additional UX Enhancements
// 1. Dynamic Chat Input Placeholder
function updateChatInputPlaceholder() {
    const messages = chatLog.children;
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1].textContent;
        if (lastMessage.startsWith('MAIA:')) {
            chatInput.placeholder = "Your turn...";
        } else {
            chatInput.placeholder = "M.A.I.A. is waiting for your response...";
        }
    }
}

chatLog.addEventListener('DOMSubtreeModified', debounce(updateChatInputPlaceholder, 300));

// 2. Error Handling for File Upload
fileInput.addEventListener('change', (event) => {
    const files = event.target.files;
    if (files.length > 5) {
        alert("You can upload up to 5 files at a time.");
        fileInput.value = ""; // Clear file input
    }
});

// 3. CSS for Lightbox
const style = document.createElement('style');
style.textContent = `
    .lightbox {
        position: fixed;
        z-index: 1000;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .lightbox img {
        max-width: 90%;
        max-height: 90%;
        border: 2px solid #00d8ff;
    }
    .lightbox .close {
        position: absolute;
        top: 15px;
        right: 15px;
        color: #f4f4f4;
        font-size: 24px;
        cursor: pointer;
    }
`;
document.head.appendChild(style);
});

