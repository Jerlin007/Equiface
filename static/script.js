// Cache DOM elements
const modeSelect = document.getElementById("modeSelect");
const fileUploadContainer = document.getElementById("fileUploadContainer");
const webcamContainer = document.getElementById("webcamContainer");
const uploadButton = document.getElementById("uploadButton");
const captureButton = document.getElementById("captureButton");
const recaptureButton = document.getElementById("recaptureButton");
const fileInput = document.getElementById("fileInput");
const webcam = document.getElementById("webcam");
const captureCanvas = document.getElementById("captureCanvas");
const results = document.getElementById("results");
const trackingOverlay = document.getElementById("trackingOverlay");
const webcamControls = document.getElementById("webcamControls");
const hamburger = document.querySelector('.hamburger-menu');
const navMenu = document.querySelector('.web3-menu');


let videoStream;

// Mode toggle handler
modeSelect.addEventListener("change", () => {
    const analyzedImage = document.getElementById("analyzedImage");
    analyzedImage.style.display = "none";
    analyzedImage.classList.remove('active');
    if (modeSelect.value === "file") {
        fileUploadContainer.style.display = "block";
        webcamContainer.style.display = "none";
        webcamControls.style.display = "none"; // Hide capture/recapture buttons
        stopWebcam();
        resetWebcamView();
        results.innerHTML = `<h2>Results</h2><pre id="resultOutput">No analysis yet.</pre>`;
    } else {
        fileUploadContainer.style.display = "none";
        webcamContainer.style.display = "block";
        webcamControls.style.display = "flex"; // Show the controls only in webcam mode
        startWebcam();
    }
});

// Add file input change handler to show preview
fileInput.addEventListener('change', function(e) {
    const analyzedImage = document.getElementById("analyzedImage");
    const file = e.target.files[0];
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            analyzedImage.src = e.target.result;
            analyzedImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// --- Helper Functions for UI State Management ---
function resetWebcamView() {
    // Show video, hide canvas, and adjust button visibility
    webcam.style.display = 'block';
    captureCanvas.style.display = 'none';
    captureButton.style.display = 'block';
    recaptureButton.style.display = 'none';
    // Disable capture button until a proper face alignment is detected
    captureButton.disabled = true;
}

/* 
  Instead of toggling an external #loading element, 
  we replace the results contents with our square spinner loader.
*/
function showLoading() {
    results.innerHTML = `
    <h2>Results</h2>
    <div class="custom-loader">
      <div class="square-spinner"></div>
      <span class="loader-dot" style="animation-delay: 0s;">.</span>
      <span class="loader-dot" style="animation-delay: 0.3s;">.</span>
      <span class="loader-dot" style="animation-delay: 0.6s;">.</span>
    </div>
  `;
}

// When hiding the loader, we simply clear its content.
// (The displayResults() function will overwrite it.)
function hideLoading() {
    results.innerHTML = `<h2>Results</h2>`;
}

// --- Webcam Controls ---
// Start Webcam and initiate face tracking
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => {
            videoStream = stream;
            webcam.srcObject = stream;
            resetWebcamView();
            // Begin face tracking for alignment
            trackFace();
        })
        .catch(err => displayResults({ error: "Webcam not accessible" }));
}

// Stop Webcam
function stopWebcam() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
}

// Update the uploadButton event listener
uploadButton.addEventListener("click", () => {
    const file = fileInput.files[0];
    if (file) {
        const analyzedImage = document.getElementById("analyzedImage");
        analyzedImage.style.display = 'none';
        
        // Show preview while processing
        const reader = new FileReader();
        reader.onload = (e) => {
            analyzedImage.src = e.target.result;
            analyzedImage.classList.add('active');
        };
        reader.readAsDataURL(file);

        showLoading();
        
        const formData = new FormData();
        formData.append("file", file);
        fetch("/upload", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResults(data);
                analyzedImage.classList.add('active');
            })
            .catch(err => {
                hideLoading();
                analyzedImage.style.display = 'none';
                displayResults({ error: "Error uploading file" });
            });
    } else {
        displayResults({ error: "No file selected" });
    }
});


// --- Capture & Recapture Handlers ---
// Attach a single capture handler to avoid duplicates.
captureButton.addEventListener("click", handleCapture);

recaptureButton.addEventListener("click", handleRecapture);

function handleCapture() {
    // Freeze the current frame by drawing it to the canvas
    const context = captureCanvas.getContext("2d");
    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    context.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);

    // Update UI: Hide webcam video and show captured image
    webcam.style.display = 'none';
    captureCanvas.style.display = 'block';
    // Disable capture to prevent multiple clicks
    captureButton.disabled = true;

    // Convert canvas image to Blob and send to the backend
    captureCanvas.toBlob(blob => {
        showLoading();
        const formData = new FormData();
        formData.append("file", blob, "capture.jpg");
        fetch("/upload", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResults(data);
                // Adjust button visibility for next action
                captureButton.style.display = 'none';
                recaptureButton.style.display = 'block';
            })
            .catch(err => {
                hideLoading();
                displayResults({ error: "Error capturing image" });
                // Revert to video view and allow capture again
                captureCanvas.style.display = 'none';
                webcam.style.display = 'block';
                captureButton.disabled = false;
            });
    });
}

function handleRecapture() {
    // Reset the UI so that the user can capture again
    resetWebcamView();
    // Optionally restart face tracking if your logic requires it:
    trackFace();
}

// --- Face Tracking with BlazeFace ---
async function trackFace() {
    try {
        const model = await blazeface.load();
        let faceDetectionActive = true;

        async function detectFrame() {
            if (!videoStream || !faceDetectionActive) return;

            const faces = await model.estimateFaces(webcam, false);
            if (faces.length > 0) {
                const face = faces[0];
                const [x, y] = face.topLeft;
                const [x2, y2] = face.bottomRight;
                const width = x2 - x;
                const height = y2 - y;
                const centerX = x + width / 2;
                const centerY = y + height / 2;

                // Calculate container dimensions and scaling
                const videoWidth = webcam.videoWidth;
                const videoHeight = webcam.videoHeight;
                const containerWidth = webcam.offsetWidth;
                const containerHeight = webcam.offsetHeight;

                const videoAspect = videoWidth / videoHeight;
                const containerAspect = containerWidth / containerHeight;
                let scale, offsetX = 0, offsetY = 0;

                if (containerAspect > videoAspect) {
                    scale = containerWidth / videoWidth;
                    offsetY = (containerHeight - (videoHeight * scale)) / 2;
                } else {
                    scale = containerHeight / videoHeight;
                    offsetX = (containerWidth - (videoWidth * scale)) / 2;
                }

                const scaledX = (centerX * scale) + offsetX;
                const scaledY = (centerY * scale) + offsetY;

                // Calculate distance from the center of the container
                const distance = Math.sqrt(
                    Math.pow(scaledX - containerWidth / 2, 2) +
                    Math.pow(scaledY - containerHeight / 2, 2)
                );

                // Adjust overlay styling based on alignment
                const overlayRadius = 60;
                if (distance <= overlayRadius) {
                    trackingOverlay.classList.remove('misaligned');
                    captureButton.disabled = false;
                } else {
                    trackingOverlay.classList.add('misaligned');
                    captureButton.disabled = true;
                }
            } else {
                trackingOverlay.classList.add('misaligned');
                captureButton.disabled = true;
            }

            requestAnimationFrame(detectFrame);
        }

        detectFrame();

        // Stop face detection when the video is paused (e.g., when capturing)
        webcam.addEventListener('pause', () => {
            faceDetectionActive = false;
            captureButton.disabled = true;
        });
    } catch (error) {
        console.error('Face detection error:', error);
        displayResults({ error: "Face detection failed" });
    }
}

// --- Display Results ---
function displayResults(data) {
    results.innerHTML = `<h2>Your Facial Symmetry Score</h2>`;
    Object.keys(data).forEach(feature => {
        const value = data[feature];
        const label = document.createElement("div");
        label.className = "result-item";
        label.textContent = `${feature.charAt(0).toUpperCase() + feature.slice(1)}: ${value}%`;

        const progressBar = document.createElement("div");
        progressBar.className = "progress-bar";
        progressBar.style.width = `${value}%`;

        const progressContainer = document.createElement("div");
        progressContainer.className = "progress-bar-container";
        progressContainer.appendChild(progressBar);

        results.appendChild(label);
        results.appendChild(progressContainer);
    });
}

// FAQ Toggle
document.querySelectorAll('.faq-question').forEach(item => {
    item.addEventListener('click', () => {
        const parent = item.parentElement;
        parent.classList.toggle('active');
    });
});


// Header Scroll Effect
const header = document.querySelector('.web3-header');
const appElement = document.getElementById('tips');

function updateHeaderSticky() {
    const appRect = appElement.getBoundingClientRect();
    
    // Check if the top of #app is at or above the viewport top
    if (appRect.top <= 0) {
        header.classList.add('sticky');
    } else {
        header.classList.remove('sticky');
    }
}

window.addEventListener('scroll', updateHeaderSticky);
window.addEventListener('resize', updateHeaderSticky); // Handle window resize
updateHeaderSticky(); // Initial check

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close menu when clicking outside
document.addEventListener('click', (e) => {
    if (!hamburger.contains(e.target) && !navMenu.contains(e.target)) {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');

        // If using body scroll lock
        document.body.classList.remove('menu-open');
        document.body.style.overflow = '';
    }
});

// Close menu when clicking menu links
document.querySelectorAll('.web3-link').forEach(link => {
    link.addEventListener('click', (e) => {
        // Only close if on mobile view
        if (window.innerWidth <= 768) {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
            
            // If using body scroll lock (from previous implementation)
            document.body.classList.remove('menu-open');
            document.body.style.overflow = '';
        }
    });
});