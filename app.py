from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from PIL import Image, ImageOps
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

from PIL import Image, ImageOps

from PIL import Image, ImageOps, ExifTags

def preprocess_image(image_path):
    """
    Preprocess the image by:
      1. Correcting orientation only if needed based on EXIF data.
      2. Converting RGBA to RGB if necessary.
      3. Resizing the image to 310x413.
    The resulting image overwrites the original file at image_path.
    """
    with Image.open(image_path) as img:
        # Check if the image has EXIF orientation information
        try:
            exif = img._getexif()
        except Exception:
            exif = None
        
        # Map EXIF orientation tag if it exists.
        orientation_tag = None
        if exif is not None:
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_tag = tag
                    break

        # Apply exif_transpose only if orientation exists and is not the default (1)
        if exif is not None and orientation_tag in exif:
            if exif[orientation_tag] != 1:
                img = ImageOps.exif_transpose(img)
        
        # Convert RGBA to RGB only if needed
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        # Resize the image to the target resolution 310x413
        img = img.resize((310, 413))
        
        # Save the processed image back to the same path
        img.save(image_path)



def analyze_symmetry_mediapipe(image_path):

    """
    Analyze facial symmetry using MediaPipe after preprocessing the image.
    """
    # Preprocess the image: correct orientation, convert to RGB if needed, and resize to 310x413
    preprocess_image(image_path)

    # Read the processed image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run MediaPipe Face Mesh
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    # Get the landmarks for the first face (you can modify for multiple faces)
    landmarks = results.multi_face_landmarks[0].landmark

    # Convert landmarks to pixel coordinates
    image_height, image_width, _ = image.shape
    points = [(int(landmark.x * image_width), int(landmark.y * image_height)) for landmark in landmarks]

    # Calculate symmetry (similar to your existing method)
    eyes_symmetry = max(0, 100 - abs(points[33][0] - points[133][0]))  # Distance between eyes (example)
    mouth_symmetry = max(0, 100 - abs(points[62][0] - points[314][0]))  # Mouth width (example)
    nose_symmetry = max(0, 100 - abs(points[31][0] - points[35][0]))  # Nose width (example)
    eyebrows_symmetry = max(0, 100 - abs(points[21][1] - points[22][1]))  # Height difference of eyebrows
    jawline_symmetry = max(0, 100 - abs(points[5][1] - points[11][1]))  # Jawline comparison (example)

    # 6. Vertical symmetry (based on nose bridge and comparing other landmarks)
    midline_x = (points[27][0] + points[30][0]) // 2  # Using the nose bridge as the midline
    vertical_symmetry_diff = 0
    count = 0
    for i, j in zip(range(17), range(16, -1, -1)):  # Pairs of left and right face landmarks
        left_point = points[i]
        right_point = points[j]
        vertical_symmetry_diff += abs((2 * midline_x) - (left_point[0] + right_point[0]))
        count += 1
    vertical_symmetry = max(0, 100 - (vertical_symmetry_diff / count))  # Normalize to a score out of 100


    # 7. Horizontal symmetry (based on eye area)
    eye_top = (points[37][1] + points[38][1] + points[43][1] + points[44][1]) / 4
    eye_bottom = (points[40][1] + points[41][1] + points[46][1] + points[47][1]) / 4
    horizontal_symmetry_diff = abs(eye_bottom - eye_top)
    horizontal_symmetry = max(0, 100 - horizontal_symmetry_diff)  # Normalize to a score out of 100

    # 8. Overall symmetry as an average of individual scores
    overall_symmetry = np.mean([eyes_symmetry, mouth_symmetry, nose_symmetry,
                                eyebrows_symmetry, jawline_symmetry, vertical_symmetry, horizontal_symmetry])


    results = {
        "eyes": eyes_symmetry,
        "mouth": mouth_symmetry,
        "nose": nose_symmetry,
        "eyebrows": eyebrows_symmetry,
        "jawline": jawline_symmetry,
        "vertical_symmetry": vertical_symmetry,
        "horizontal_symmetry": horizontal_symmetry,
        "overall": overall_symmetry
    }

    return results

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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": f"Error saving file: {str(e)}"})
        
        try:
            # Preprocess and analyze the image
            preprocess_image(file_path)
            results = analyze_symmetry_mediapipe(file_path)
        except Exception as e:
            return jsonify({"error": f"Error in analyzing image: {str(e)}"})
        finally:
            # Remove the file regardless of success or error
            try:
                os.remove(file_path)
            except Exception as remove_err:
                print("Error deleting file:", remove_err)
        
        return jsonify(results)
    
    return jsonify({"error": "Invalid file type"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)





