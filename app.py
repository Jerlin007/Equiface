from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from PIL import Image, ImageOps, ExifTags
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
        _ = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        return self.application

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "https://your-domain.com"}})  # Restrict in production
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limit to 5MB
UPLOAD_BUCKET = os.environ.get('UPLOAD_BUCKET', 'my-ml-app-bucket')
app.config['UPLOAD_BUCKET'] = UPLOAD_BUCKET
os.makedirs('/tmp/uploads', exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
storage_client = storage.Client()

def preprocess_image(image_path):
    with Image.open(image_path) as img:
        try:
            exif = img._getexif()
        except Exception:
            exif = None
        orientation_tag = None
        if exif is not None:
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_tag = tag
                    break
        if exif is not None and orientation_tag in exif:
            if exif[orientation_tag] != 1:
                img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        img.save(image_path)

def analyze_symmetry_mediapipe(image_path):
    import gc
    gc.collect()
    preprocess_image(image_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}
    landmarks = results.multi_face_landmarks[0].landmark
    image_height, image_width, _ = image.shape
    points = [(int(landmark.x * image_width), int(landmark.y * image_height)) for landmark in landmarks]
    eyes_symmetry = max(0, 100 - abs(points[33][0] - points[133][0]))
    mouth_symmetry = max(0, 100 - abs(points[62][0] - points[314][0]))
    nose_symmetry = max(0, 100 - abs(points[31][0] - points[35][0]))
    eyebrows_symmetry = max(0, 100 - abs(points[21][1] - points[22][1]))
    jawline_symmetry = max(0, 100 - abs(points[5][1] - points[11][1]))
    midline_x = (points[27][0] + points[30][0]) // 2
    vertical_symmetry_diff = 0
    count = 0
    for i, j in zip(range(17), range(16, -1, -1)):
        left_point = points[i]
        right_point = points[j]
        vertical_symmetry_diff += abs((2 * midline_x) - (left_point[0] + right_point[0]))
        count += 1
    vertical_symmetry = max(0, 100 - (vertical_symmetry_diff / count))
    eye_top = (points[37][1] + points[38][1] + points[43][1] + points[44][1]) / 4
    eye_bottom = (points[40][1] + points[41][1] + points[46][1] + points[47][1]) / 4
    horizontal_symmetry_diff = abs(eye_bottom - eye_top)
    horizontal_symmetry = max(0, 100 - horizontal_symmetry_diff)
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
    del image, image_rgb, results
    gc.collect()
    return results

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Remove leading slash

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg'}
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"})
    file_path = os.path.join('/tmp/uploads', file.filename)
    try:
        file.save(file_path)
        bucket = storage_client.bucket(app.config['UPLOAD_BUCKET'])
        blob = bucket.blob(f"uploads/{file.filename}")
        blob.upload_from_filename(file_path)
        preprocess_image(file_path)
        results = analyze_symmetry_mediapipe(file_path)
    except GoogleCloudError as e:
        return jsonify({"error": f"Cloud Storage error: {str(e)}"})
    except Exception as e:
        return jsonify({"error": f"Error in analyzing image: {str(e)}"})
    finally:
        try:
            os.remove(file_path)
        except Exception as remove_err:
            print("Error deleting file:", remove_err)
        try:
            blob.delete()
        except Exception as delete_err:
            print("Error deleting from Cloud Storage:", delete_err)
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
