from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import numpy as np
from PIL import Image
import os

try:
    import joblib
except ImportError:
    joblib = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_FOLDER = BASE_DIR / "model"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = MODEL_FOLDER / "retinopathy_model.pkl"

# Educational demo classifier.
# It uses a saved ML model when model/retinopathy_model.pkl exists.
# Otherwise, it uses a simple image-feature fallback so the UI can be tested.
model = None
if joblib and MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

CLASSES = {
    0: ("No Diabetic Retinopathy", "No obvious signs detected in this demo.", "Low"),
    1: ("Mild Diabetic Retinopathy", "Possible early retinal changes.", "Moderate"),
    2: ("Moderate Diabetic Retinopathy", "Possible moderate retinal changes.", "High"),
    3: ("Severe Diabetic Retinopathy", "Possible advanced retinal changes.", "High"),
    4: ("Proliferative Diabetic Retinopathy", "Possible advanced/proliferative changes.", "Very High"),
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(path):
    """Small, transparent feature vector for the demo model."""
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    features = [
        gray.mean(),
        gray.std(),
        arr[:,:,0].mean(),
        arr[:,:,1].mean(),
        arr[:,:,2].mean(),
        np.percentile(gray, 5),
        np.percentile(gray, 25),
        np.percentile(gray, 75),
        np.percentile(gray, 95),
        (gray < 0.18).mean(),
        (gray > 0.85).mean(),
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)

def demo_prediction(path):
    # This is ONLY a UI/demo fallback, not a medically valid diagnostic model.
    f = extract_features(path)[0]
    darkness = f[9]
    brightness = f[10]
    contrast = f[1]
    score = darkness * 2.0 + contrast * 0.8 + max(0, 0.25 - brightness) * 1.5
    if score < 0.20:
        cls = 0
    elif score < 0.35:
        cls = 1
    elif score < 0.50:
        cls = 2
    elif score < 0.70:
        cls = 3
    else:
        cls = 4
    confidence = min(96.0, 58.0 + score * 50.0)
    return cls, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please select a retinal image."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only JPG, JPEG and PNG images are allowed."}), 400

    safe_name = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / safe_name
    file.save(file_path)

    try:
        if model is not None:
            features = extract_features(file_path)
            pred = int(model.predict(features)[0])
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(features)) * 100)
            else:
                confidence = 80.0
        else:
            pred, confidence = demo_prediction(file_path)

        label, description, risk = CLASSES.get(pred, CLASSES[0])
        return jsonify({
            "success": True,
            "prediction": label,
            "description": description,
            "risk": risk,
            "confidence": round(confidence, 2),
            "image_url": f"/uploads/{safe_name}",
            "demo_model": model is None,
            "disclaimer": "Educational mini-project only. This result is not a medical diagnosis."
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
