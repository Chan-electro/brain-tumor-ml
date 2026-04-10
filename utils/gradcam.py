"""
gradcam.py
----------
Grad-CAM visualization utilities for the brain tumor classifier.

Key improvements over the original:
  - Auto-detects the last convolutional layer — no hardcoded layer names
    (works with EfficientNetB0, B4, DenseNet121, ResNet, etc.)
  - Returns (heatmap, predicted_idx, pred_probs) so app.py doesn't need
    a separate model.predict() call
  - Uses cv2.COLORMAP_TURBO (perceptually more uniform than JET)
  - Uses cv2.addWeighted for artifact-free alpha blending
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------------
# Layer auto-detection
# ---------------------------------------------------------------------------

def get_last_conv_layer_name(model) -> str:
    """
    Auto-detect the name of the last convolutional layer in a Keras model.

    Searches the model's layer list in reverse. Handles both flat models
    and models that wrap a sub-model (e.g. EfficientNet backbone inside
    a custom functional model).

    Supported layer types: Conv2D, DepthwiseConv2D, SeparableConv2D,
    Conv2DTranspose.

    Args:
        model: a compiled tf.keras.Model

    Returns:
        Name of the last convolutional layer as a string.

    Raises:
        ValueError: if no convolutional layer is found.
    """
    _conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Conv2DTranspose,
    )

    # First pass: top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, _conv_types):
            return layer.name

    # Second pass: nested sub-models (e.g. EfficientNetB4 as a layer)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, _conv_types):
                    return sub_layer.name

    raise ValueError(
        "No convolutional layer found in model. "
        "Specify last_conv_layer_name manually."
    )


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
        last_conv_layer_name: if None, auto-detected via get_last_conv_layer_name()
        pred_index:           class index to explain. If None, uses argmax prediction.

    Returns:
        heatmap      (H', W') float32 ndarray in [0, 1] — spatial attention map
        predicted_idx int     — argmax class index
        pred_probs   (C,)     — softmax probabilities for all classes
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    # Build a sub-model that outputs both the conv feature maps and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pool the gradients → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight each feature map channel by its importance
    conv_outputs = conv_outputs[0]                        # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                         # (h, w)

    # ReLU: only keep positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    pred_probs = predictions[0].numpy()  # (C,) float32
    return heatmap.numpy(), pred_index, pred_probs


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay_heatmap(
    heatmap: np.ndarray,
    image,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Resize the Grad-CAM heatmap and blend it over the original MRI image.

    Args:
        heatmap: float32 ndarray in [0, 1], shape (h, w) — from make_gradcam_heatmap
        image:   PIL.Image or uint8 numpy array (H, W, 3)
        alpha:   heatmap opacity (0 = invisible, 1 = full heatmap)

    Returns:
        overlayed: uint8 numpy array (H, W, 3), RGB
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB")).astype(np.uint8)
    else:
        image_np = np.array(image).astype(np.uint8)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    h, w = image_np.shape[:2]

    # Colorize heatmap using TURBO colormap (perceptually more uniform than JET)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match the original image dimensions
    heatmap_resized = cv2.resize(heatmap_colored_rgb, (w, h),
                                  interpolation=cv2.INTER_LINEAR)

    # Blend using cv2.addWeighted (avoids float overflow/clipping artifacts)
    overlayed = cv2.addWeighted(
        image_np,        1.0 - alpha,
        heatmap_resized, alpha,
        0
    )
    return overlayed
