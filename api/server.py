import os
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

face_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(BASE_DIR, "haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

@app.get("/")
def home():
    return {"message": "Eye Detection API Running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        results.append({
            "face": [int(x), int(y), int(w), int(h)],
            "eyes_detected": len(eyes)
        })

    return {"faces": results}