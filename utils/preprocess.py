"""
preprocess.py
-------------
Image preprocessing for Flask inference.

Training pipeline  (train.ipynb make_dataset):
  1. Brain-contour crop          (pure OpenCV)
  2. Resize  (224, 224)          cubic interpolation
  3. float32  [0, 255]
  4. effnet_preprocess()         scales [0,255] → EfficientNet expected range

Inference pipeline (this file) must match steps 1-4 exactly.

NOTE: The model has NO Lambda layer.
      preprocess_input is applied here before model.predict().
"""

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

IMG_SIZE = (224, 224)


def crop_brain_contour(image_np: np.ndarray) -> np.ndarray:
    """
    Detect brain region via largest-contour and crop tightly around it.
    Adapted from MohamedAliHabib/Brain-Tumor-Detection.
    Pure OpenCV — no imutils needed. Compatible with OpenCV 3.x and 4.x.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts_output = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts_output[0] if len(cnts_output) == 2 else cnts_output[1]

        if not cnts:
            return image_np

        c = max(cnts, key=cv2.contourArea)
        ext_left  = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top   = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot   = tuple(c[c[:, :, 1].argmax()][0])

        cropped = image_np[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return image_np
        return cropped
    except Exception:
        return image_np


def load_and_preprocess_image(file_storage, img_size: tuple = IMG_SIZE):
    """
    Preprocess an uploaded MRI image for model inference.

    Matches training pipeline (train.ipynb → make_dataset) exactly:
      1. Open and convert to RGB
      2. Brain-contour crop
      3. Resize to img_size with cubic interpolation
      4. float32 [0, 255]
      5. effnet_preprocess()  ← must be applied since the model has no Lambda layer

    Args:
        file_storage: Werkzeug FileStorage (uploaded file from Flask)
        img_size: (width, height) tuple

    Returns:
        (image_array, pil_image)
        image_array: float32 numpy array, shape (1, H, W, 3), preprocessed
        pil_image  : PIL.Image (RGB, resized) for display
    """
    pil_image = Image.open(file_storage).convert("RGB")
    image_np = np.array(pil_image, dtype=np.uint8)

    # Step 1: Brain-contour crop (same as training)
    image_np = crop_brain_contour(image_np)

    # Step 2: Resize with cubic interpolation
    image_np = cv2.resize(image_np, img_size, interpolation=cv2.INTER_CUBIC)

    # Step 3: float32 [0, 255]
    image_array = image_np.astype(np.float32)

    # Step 4: EfficientNetB0 preprocess_input — must match make_dataset() in train.ipynb
    image_array = effnet_preprocess(image_array)       # in-place safe, returns same shape

    # Add batch dimension: (1, H, W, 3)
    image_array = image_array[np.newaxis, ...]

    # Clean PIL image for display (before preprocessing)
    display_pil = Image.fromarray(image_np)

    return image_array, display_pil



def crop_brain_contour(image_np: np.ndarray) -> np.ndarray:
    """
    Detect brain region via largest-contour and crop tightly around it.
    Adapted from MohamedAliHabib/Brain-Tumor-Detection.
    Pure OpenCV — no imutils needed.

    Args:
        image_np: uint8 RGB array
    Returns:
        Cropped uint8 array, or original if crop fails.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Compatible with OpenCV 3.x and 4.x
        cnts_output = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts_output[0] if len(cnts_output) == 2 else cnts_output[1]

        if not cnts:
            return image_np

        c = max(cnts, key=cv2.contourArea)
        ext_left  = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top   = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot   = tuple(c[c[:, :, 1].argmax()][0])

        cropped = image_np[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]

        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return image_np

        return cropped
    except Exception:
        return image_np


def load_and_preprocess_image(file_storage, img_size: tuple = IMG_SIZE):
    """
    Preprocess an uploaded MRI image for model inference.

    Steps (must match training pipeline in train.ipynb):
      1. Open and convert to RGB
      2. Brain-contour crop
      3. Resize to img_size with cubic interpolation
      4. Convert to float32 in [0, 255]  ← model's internal preprocess_input handles the rest

    Args:
        file_storage: Werkzeug FileStorage (uploaded file from Flask)
        img_size: (width, height) tuple

    Returns:
        (image_array, pil_image)
        image_array: float32 numpy array, shape (1, H, W, 3), values in [0, 255]
        pil_image  : PIL.Image (RGB, original resolution) for display
    """
    pil_image = Image.open(file_storage).convert("RGB")

    # Convert to numpy for cropping
    image_np = np.array(pil_image, dtype=np.uint8)

    # Apply brain-contour crop (same as training)
    image_np = crop_brain_contour(image_np)

    # Resize with cubic interpolation
    image_np = cv2.resize(image_np, img_size, interpolation=cv2.INTER_CUBIC)

    # Keep [0, 255] float32 — model's Lambda layer calls EfficientNet preprocess_input internally
    image_array = image_np.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # (1, H, W, 3)

    # Also return a clean PIL image for display (using the resized numpy array)
    display_pil = Image.fromarray(image_np)

    return image_array, display_pil
