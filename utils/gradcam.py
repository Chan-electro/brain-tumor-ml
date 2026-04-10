"""
gradcam.py
----------
Grad-CAM visualization utilities for the brain tumor classifier.

Key features:
  - Auto-detects the last convolutional layer — no hardcoded layer names
    (works with EfficientNetB0, B4, DenseNet121, ResNet, etc.)
  - Handles nested sub-models (e.g. EfficientNet backbone wrapped in
    a functional model)
  - Returns (heatmap, predicted_idx, pred_probs) so app.py doesn't need
    a separate model.predict() call
  - Uses intensity-weighted per-pixel alpha blending so only the regions
    the model focuses on are highlighted — no washed-out full-image color
  - Power-law contrast enhancement for crisper heatmaps
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------------
# Layer auto-detection
# ---------------------------------------------------------------------------

_CONV_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.Conv2DTranspose,
)


def _find_conv_layer_and_model(model):
    """
    Find the last convolutional layer and the (sub-)model it belongs to.

    Returns:
        (layer_name, target_model)
        target_model is the model object that directly contains the layer.
    """
    # First pass: top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, _CONV_TYPES):
            return layer.name, model

    # Second pass: nested sub-models (e.g. EfficientNetB4 as a layer)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, _CONV_TYPES):
                    return sub_layer.name, layer  # layer is the sub-model

    raise ValueError(
        "No convolutional layer found in model. "
        "Specify last_conv_layer_name manually."
    )


def get_last_conv_layer_name(model) -> str:
    """Auto-detect the name of the last convolutional layer."""
    name, _ = _find_conv_layer_and_model(model)
    return name


# ---------------------------------------------------------------------------
# Grad-CAM heatmap
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: str = None,
    pred_index: int = None,
):
    """
    Compute a Grad-CAM heatmap for the given input.

    Args:
        img_array:            numpy array, shape (1, H, W, 3), float32
        model:                compiled tf.keras.Model
        last_conv_layer_name: if None, auto-detected
        pred_index:           class index to explain. If None, uses argmax.

    Returns:
        heatmap      (H', W') float32 ndarray in [0, 1]
        predicted_idx int
        pred_probs   (C,) float32
    """
    if last_conv_layer_name is None:
        last_conv_layer_name, target_model = _find_conv_layer_and_model(model)
    else:
        # Try to find the layer in the top-level model first
        try:
            model.get_layer(last_conv_layer_name)
            target_model = model
        except ValueError:
            # Layer is inside a nested sub-model
            for layer in model.layers:
                if hasattr(layer, "layers"):
                    try:
                        layer.get_layer(last_conv_layer_name)
                        target_model = layer
                        break
                    except ValueError:
                        continue
            else:
                raise ValueError(f"Layer '{last_conv_layer_name}' not found.")

    # Get the conv layer output — resolve from the correct (sub-)model
    conv_layer = target_model.get_layer(last_conv_layer_name)
    conv_output = conv_layer.output

    # Build a sub-model: same inputs as the full model, outputs conv maps + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("Warning: GradCAM gradients are None — returning blank heatmap")
        h, w = img_array.shape[1], img_array.shape[2]
        return np.zeros((h, w), dtype=np.float32), pred_index, predictions[0].numpy()

    # Global-average-pool the gradients → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]                          # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                           # (h, w)

    # ReLU: only keep positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    # Power-law contrast boost — sharpens the hot-spot vs. background separation
    heatmap = heatmap.numpy().astype(np.float32)
    heatmap = np.power(heatmap, 1.5)

    pred_probs = predictions[0].numpy()
    return heatmap, pred_index, pred_probs


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay_heatmap(
    heatmap: np.ndarray,
    image,
    alpha: float = 0.55,
    threshold: float = 0.15,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap over the original image using per-pixel
    intensity-weighted alpha.  Only regions the model focuses on get colored;
    low-activation areas stay as the original MRI.

    Args:
        heatmap:   float32 (h, w) in [0, 1] from make_gradcam_heatmap
        image:     PIL.Image or uint8 numpy (H, W, 3)
        alpha:     max opacity for hottest region (0–1)
        threshold: suppress heatmap values below this (reduces noise)
        colormap:  OpenCV colormap constant (JET shows hot=red, cold=blue)

    Returns:
        uint8 (H, W, 3) RGB overlay
    """
    # --- Convert input image ---
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        image_np = np.asarray(image, dtype=np.uint8)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    h, w = image_np.shape[:2]

    # --- Resize heatmap to image size (float, before colorizing) ---
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )

    # --- Threshold: suppress low activations ---
    heatmap_resized = np.clip(heatmap_resized - threshold, 0, 1)
    max_val = heatmap_resized.max()
    if max_val > 0:
        heatmap_resized = heatmap_resized / max_val  # re-normalize to [0, 1]

    # --- Colorize ---
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # --- Per-pixel alpha mask (float32, shape H×W×1) ---
    # Use the heatmap intensity itself as the alpha so only hot regions show color
    pixel_alpha = (heatmap_resized * alpha)[..., np.newaxis]  # (H, W, 1)

    # --- Blend: original * (1 - a) + colored_heatmap * a ---
    blended = (
        image_np.astype(np.float32) * (1.0 - pixel_alpha)
        + heatmap_colored_rgb.astype(np.float32) * pixel_alpha
    )

    return np.clip(blended, 0, 255).astype(np.uint8)
