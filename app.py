from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from PIL import Image, ImageOps
import cv2
import mediapipe as mp
import numpy as np
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from gunicorn.app.base import BaseApplication

class FlaskApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key, value)

    def load(self):
        # Preload models before worker fork
        _ = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        return self.application

app = Flask(__name__)
CORS(app)
UPLOAD_BUCKET = 'my-ml-app-bucket'  # Replace with your bucket name
app.config['UPLOAD_BUCKET'] = UPLOAD_BUCKET

# Ensure upload folder exists locally for temp processing
os.makedirs('/tmp/uploads', exist_ok=True)

# Initialize Cloud Storage client
storage_client = storage.Client()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

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
        img = img.resize((224, 224))
        
        # Save the processed image back to the same path
        img.save(image_path)



def analyze_symmetry_mediapipe(image_path):
        # Add at the start of the function
    import gc
    gc.collect()

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
    # Add before return
    del image, image_rgb, results
    gc.collect()
    return results

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and file.filename.lower().endswith(('.jpeg', '.jpg')):
        # Save temporarily locally
        file_path = os.path.join('/tmp/uploads', file.filename)
        
        try:
            file.save(file_path)
            
            # Upload to Cloud Storage
            bucket = storage_client.bucket(app.config['UPLOAD_BUCKET'])
            blob = bucket.blob(f"uploads/{file.filename}")
            blob.upload_from_filename(file_path)
            
            # Preprocess and analyze
            preprocess_image(file_path)
            results = analyze_symmetry_mediapipe(file_path)
        except GoogleCloudError as e:
            return jsonify({"error": f"Cloud Storage error: {str(e)}"})
        except Exception as e:
            return jsonify({"error": f"Error in analyzing image: {str(e)}"})
        finally:
            # Clean up local file
            try:
                os.remove(file_path)
            except Exception as remove_err:
                print("Error deleting file:", remove_err)
            # Optionally delete from Cloud Storage if not needed
            try:
                blob.delete()
            except Exception as delete_err:
                print("Error deleting from Cloud Storage:", delete_err)
        
        return jsonify(results)
    
    return jsonify({"error": "Invalid file type"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)





