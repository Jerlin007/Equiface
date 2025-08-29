"""
===============================================================================
Flask Web Application for Facial Symmetry Analysis
-------------------------------------------------------------------------------
This application:
  - Serves a web interface for uploading images or capturing webcam shots.
  - Preprocesses images (correcting orientation, color mode, and resizing).
  - Uses MediaPipe's Face Mesh to detect facial landmarks.
  - Computes various symmetry scores based on key facial features.
  - Returns the analysis results as JSON.
  - Does NOT save uploaded files (all processing is done in-memory).
===============================================================================
"""

# -----------------------------------------------------------------------------
# Module Imports and Flask App Configuration
# -----------------------------------------------------------------------------
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageOps, ExifTags
import cv2
import mediapipe as mp
import numpy as np

# Initialize the Flask application and enable CORS for cross-domain requests.
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# MediaPipe Face Mesh Initialization
# -----------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# -----------------------------------------------------------------------------
# Image Preprocessing Function (in-memory)
# -----------------------------------------------------------------------------
def preprocess_image(pil_img):
    """
    Preprocesses a PIL image to prepare it for facial symmetry analysis.

    Steps:
      1. Correct EXIF orientation if available.
      2. Convert RGBA to RGB if needed.
      3. Resize to (310x413).
    
    Parameters:
      pil_img (PIL.Image): Input image.

    Returns:
      PIL.Image: Preprocessed image.
    """
    # Correct EXIF orientation
    try:
        exif = pil_img._getexif()
    except Exception:
        exif = None

    if exif is not None:
        orientation_tag = None
        for tag, value in ExifTags.TAGS.items():
            if value == 'Orientation':
                orientation_tag = tag
                break
        if orientation_tag in exif and exif[orientation_tag] != 1:
            pil_img = ImageOps.exif_transpose(pil_img)

    # Convert to RGB if RGBA
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")

    # Resize
    pil_img = pil_img.resize((310, 413))
    return pil_img

# -----------------------------------------------------------------------------
# Facial Symmetry Analysis Function using MediaPipe
# -----------------------------------------------------------------------------
def analyze_symmetry_mediapipe(pil_img):
    """
    Analyzes facial symmetry using MediaPipe Face Mesh.

    Parameters:
      pil_img (PIL.Image): Preprocessed PIL image.

    Returns:
      dict: Symmetry scores or error message.
    """
    pil_img = preprocess_image(pil_img)

    # Convert to OpenCV format
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark
    image_height, image_width, _ = image.shape
    points = [
        (int(landmark.x * image_width), int(landmark.y * image_height))
        for landmark in landmarks
    ]

    # Symmetry metrics
    eyes_symmetry = max(0, 100 - abs(points[33][0] - points[133][0]))
    mouth_symmetry = max(0, 100 - abs(points[62][0] - points[314][0]))
    nose_symmetry = max(0, 100 - abs(points[31][0] - points[35][0]))
    eyebrows_symmetry = max(0, 100 - abs(points[21][1] - points[22][1]))
    jawline_symmetry = max(0, 100 - abs(points[5][1] - points[11][1]))

    # Vertical symmetry
    midline_x = (points[27][0] + points[30][0]) // 2
    vertical_symmetry_diff = 0
    count = 0
    for i, j in zip(range(17), range(16, -1, -1)):
        left_point = points[i]
        right_point = points[j]
        vertical_symmetry_diff += abs((2 * midline_x) - (left_point[0] + right_point[0]))
        count += 1
    vertical_symmetry = max(0, 100 - (vertical_symmetry_diff / count))

    # Horizontal symmetry
    eye_top = (points[37][1] + points[38][1] + points[43][1] + points[44][1]) / 4
    eye_bottom = (points[40][1] + points[41][1] + points[46][1] + points[47][1]) / 4
    horizontal_symmetry_diff = abs(eye_bottom - eye_top)
    horizontal_symmetry = max(0, 100 - horizontal_symmetry_diff)

    overall_symmetry = np.mean([
        eyes_symmetry, mouth_symmetry, nose_symmetry,
        eyebrows_symmetry, jawline_symmetry,
        vertical_symmetry, horizontal_symmetry
    ])

    return {
        "eyes": eyes_symmetry,
        "mouth": mouth_symmetry,
        "nose": nose_symmetry,
        "eyebrows": eyebrows_symmetry,
        "jawline": jawline_symmetry,
        "vertical_symmetry": vertical_symmetry,
        "horizontal_symmetry": horizontal_symmetry,
        "overall": overall_symmetry
    }

# -----------------------------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("/index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.lower().endswith(('.jpeg', '.jpg')):
        try:
            pil_img = Image.open(BytesIO(file.read()))
            analysis_results = analyze_symmetry_mediapipe(pil_img)
        except Exception as e:
            return jsonify({"error": f"Error in analyzing image: {str(e)}"})
        
        return jsonify(analysis_results)

    return jsonify({"error": "Invalid file type"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
