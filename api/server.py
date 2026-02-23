import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Get base directory (important for Vercel)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full paths to XML files
face_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(BASE_DIR, "haarcascade_eye.xml")

# Load cascades
face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

# Check if models loaded correctly
if face_cascade.empty():
    raise Exception("Failed to load haarcascade_frontalface_default.xml")

if eye_cascade.empty():
    raise Exception("Failed to load haarcascade_eye.xml")


# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Eye Detection API Running Successfully"
    })


# Detect route
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        results.append({
            "face_coordinates": [int(x), int(y), int(w), int(h)],
            "eyes_detected": int(len(eyes))
        })

    return jsonify({
        "total_faces": len(results),
        "results": results
    })


# Required for Vercel
app = app
