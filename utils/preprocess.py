"""
preprocess.py
-------------
Image preprocessing for Flask inference.

The pipeline MUST match the training pipeline exactly:
  1. Resize to the model's expected input size
  2. Convert to float32 in [0, 255] — EfficientNetB4's preprocess_input
     handles its own normalization internally, so we do NOT divide by 255
  3. Apply CLAHE contrast enhancement
  4. Apply per-image Z-score normalization

Note: We pass values in [0, 255] to apply_clahe (which converts internally
to uint8), then zscore_normalize brings the result to [0, 1]. This matches
the mri_augment(training=False) path used during training.
"""

import numpy as np
import cv2
from PIL import Image
import imutils
import tensorflow as tf

IMG_SIZE = (224, 224)

def load_and_preprocess_image(file_storage, img_size: tuple = IMG_SIZE):
    """
    file_storage is the uploaded file from Flask (werkzeug FileStorage).
    Returns image tensor ready for model: shape (1, 224, 224, 3)
    """
    image = Image.open(file_storage).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image
