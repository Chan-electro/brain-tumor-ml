from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image

from utils.preprocess import load_and_preprocess_image
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH     = "models/best_model.keras"
LABEL_MAP_PATH = "models/label_map.json"

# Fallback: support old .h5 format if new model not yet trained
_FALLBACK_MODEL_PATH = "models/tumor_classifier.h5"

model       = None
idx_to_label = {}


# ---------------------------------------------------------------------------
# Startup: load model + labels
# ---------------------------------------------------------------------------

def load_model_and_labels():
    global model, idx_to_label

    # Load label map
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        idx_to_label = {int(v): k for k, v in label_map.items()}
        print(f"Label map loaded: {idx_to_label}")
    else:
        print(f"Warning: label map not found at {LABEL_MAP_PATH}")
        idx_to_label = {}

    # Prefer new .keras format, fall back to .h5
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        model_path = _FALLBACK_MODEL_PATH
        if os.path.exists(model_path):
            print(f"Note: using fallback model {model_path}")

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Warning: no trained model found. Train the model first.")


load_model_and_labels()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def pil_to_base64(pil_image: Image.Image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mode = request.form.get("mode", "multi")
        return redirect(url_for("predict", mode=mode))
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    mode = request.args.get("mode", "multi")

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return "No file selected", 400

        if model is None:
            return "Model not loaded. Please train the model first.", 503

        try:
            img_array, pil_image = load_and_preprocess_image(file)

            # Run prediction
            preds = model.predict(img_array)[0]
            predicted_idx = int(np.argmax(preds))
            confidence = float(preds[predicted_idx])
            
            # Get label safely
            label = idx_to_label.get(predicted_idx, "Unknown")

            # Binary mode logic
            binary_label = None
            if mode == "binary":
                # Assuming 'no_tumor' is the label for no tumor
                if label.lower() == "no_tumor":
                    binary_label = "No Tumor"
                else:
                    binary_label = "Tumor"

            # Grad CAM
            # Note: 'top_conv' is a placeholder. EfficientNetB0 usually has 'top_activation' or similar.
            # We will need to verify the layer name after training.
            # For now, we'll wrap in try-except to avoid crashing if layer name is wrong
            heatmap_b64 = None
            try:
                last_conv_layer_name = "top_activation" # Common in EfficientNet
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                overlayed = overlay_heatmap(heatmap, pil_image)
                heatmap_b64 = pil_to_base64(Image.fromarray(overlayed))
            except Exception as e:
                print(f"Grad-CAM overlay error: {e}")
                heatmap_b64 = pil_to_base64(pil_image)

            original_b64 = pil_to_base64(pil_image)

            return render_template(
                "result.html",
                mode=mode,
                label=label,
                binary_label=binary_label,
                confidence=round(confidence * 100, 2),
                all_probs=all_probs,
                original_image=original_b64,
                heatmap_image=heatmap_b64,
            )

        except Exception as e:
            print(f"Prediction error: {e}")
            return f"An error occurred during prediction: {e}", 500

    return render_template("predict.html", mode=mode)


if __name__ == "__main__":
    app.run(debug=True)
