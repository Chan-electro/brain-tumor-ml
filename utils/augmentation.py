"""
augmentation.py
---------------
MRI-specific image augmentation and normalization functions.

All functions operate on float32 numpy arrays with pixel values in [0, 1].
They are designed to be called via tf.py_function inside a tf.data pipeline.

Usage in tf.data:
    from utils.augmentation import mri_augment

    def augment_train(image, label):
        image = tf.py_function(
            lambda img: mri_augment(img.numpy(), training=True),
            [image], tf.float32
        )
        image.set_shape([380, 380, 3])
        return image, label
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def apply_clahe(image_np: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to each
    RGB channel independently.

    CLAHE enhances local contrast in MRI images, making tumor boundaries
    and tissue gradients more distinguishable.

    Args:
        image_np: float32 array in [0, 1], shape (H, W, 3)

    Returns:
        CLAHE-enhanced float32 array in [0, 1], same shape.
    """
    img_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)

    # Ensure the image is 3D with exactly 3 channels.
    # MRI images may arrive as (H, W), (H, W, 1), or (H, W, 3).
    # tf.py_function can sometimes squeeze dimensions unexpectedly.
    if img_uint8.ndim == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)
    elif img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
        img_uint8 = np.concatenate([img_uint8] * 3, axis=-1)
    elif img_uint8.ndim == 3 and img_uint8.shape[2] != 3:
        # Unexpected channel count — take first channel and replicate
        img_uint8 = np.stack([img_uint8[:, :, 0]] * 3, axis=-1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_uint8)
    enhanced_channels = [clahe.apply(ch) for ch in channels]
    enhanced = cv2.merge(enhanced_channels)
    return (enhanced / 255.0).astype(np.float32)


def zscore_normalize(image_np: np.ndarray) -> np.ndarray:
    """
    Per-image Z-score normalization.

    Normalizes pixel intensity relative to the image's own mean and std.
    This is more robust than global /255 normalization for MRI, where
    scanner intensity calibration varies between acquisitions.

    Steps:
        1. Compute mean and std across all pixels
        2. Subtract mean, divide by std
        3. Clip to [-3, 3] (removes outlier intensities)
        4. Rescale to [0, 1]

    Args:
        image_np: float32 array, shape (H, W, 3)

    Returns:
        Normalized float32 array in [0, 1], same shape.
    """
    mean = image_np.mean()
    std = image_np.std() + 1e-7
    normalized = (image_np - mean) / std
    normalized = np.clip(normalized, -3.0, 3.0)
    normalized = (normalized + 3.0) / 6.0
    return normalized.astype(np.float32)


def add_gaussian_noise(image_np: np.ndarray, std: float = 0.015) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to simulate MRI scanner thermal noise.

    Args:
        image_np: float32 array in [0, 1], shape (H, W, 3)
        std: noise standard deviation (0.015 = subtle; 0.05 = visible)

    Returns:
        Noisy float32 array clipped to [0, 1], same shape.
    """
    noise = np.random.normal(0.0, std, image_np.shape).astype(np.float32)
    return np.clip(image_np + noise, 0.0, 1.0)


def apply_random_blur(image_np: np.ndarray, max_sigma: float = 0.8) -> np.ndarray:
    """
    Apply Gaussian blur with randomly sampled sigma to simulate scanner
    resolution variation.

    Args:
        image_np: float32 array in [0, 1], shape (H, W, 3)
        max_sigma: upper bound for blur sigma (0.8 is subtle)

    Returns:
        Blurred float32 array in [0, 1], same shape.
    """
    sigma = np.random.uniform(0.0, max_sigma)
    if sigma < 0.1:
        return image_np
    img_uint8 = (image_np * 255).astype(np.uint8)
    # kernel size must be odd and at least 3
    k = max(3, int(2 * round(2 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(img_uint8, (k, k), sigma)
    return (blurred / 255.0).astype(np.float32)


def elastic_transform(
    image_np: np.ndarray, alpha: float = 25.0, sigma: float = 4.0
) -> np.ndarray:
    """
    Random elastic deformation to simulate subtle brain tissue shape variation.

    Uses random displacement fields smoothed with a Gaussian kernel to create
    locally coherent deformations that preserve global anatomy.

    Args:
        image_np: float32 array in [0, 1], shape (H, W, 3)
        alpha: displacement magnitude (25 = subtle; >50 = visible distortion)
        sigma: Gaussian smoothing for the displacement field (4 = smooth)

    Returns:
        Deformed float32 array in [0, 1], same shape.
    """
    h, w = image_np.shape[:2]

    # Generate random displacement fields
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1),
        (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1),
        (0, 0), sigma
    ) * alpha

    # Create sampling maps
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)

    img_uint8 = (image_np * 255).astype(np.uint8)
    distorted = cv2.remap(
        img_uint8, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return (distorted / 255.0).astype(np.float32)


def random_brightness_contrast(
    image_np: np.ndarray,
    brightness_range: float = 0.15,
    contrast_range: float = 0.15
) -> np.ndarray:
    """
    Random brightness and contrast jitter.

    Mimics intensity variation between MRI acquisitions.

    Args:
        image_np: float32 array in [0, 1], shape (H, W, 3)
        brightness_range: maximum additive brightness shift (±0.15)
        contrast_range: maximum multiplicative contrast shift (±0.15)

    Returns:
        Adjusted float32 array clipped to [0, 1], same shape.
    """
    alpha = 1.0 + np.random.uniform(-contrast_range, contrast_range)
    beta  = np.random.uniform(-brightness_range, brightness_range)
    adjusted = image_np * alpha + beta
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full augmentation pipeline
# ---------------------------------------------------------------------------

def mri_augment(image_np: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Full MRI preprocessing and augmentation pipeline.

    Always applied (train + inference):
        1. CLAHE — contrast enhancement
        2. Z-score normalization — per-image intensity standardization

    Applied only during training (probabilistic):
        3. Gaussian noise    (p=0.50)
        4. Random blur       (p=0.30)
        5. Elastic transform (p=0.40)
        6. Brightness/contrast jitter (p=0.40)

    Args:
        image_np: float32 array. Can be in [0, 1] or [0, 255]; values >1 are
                  automatically rescaled to [0, 1] before processing.
        training: if True, apply probabilistic augmentations.

    Returns:
        Processed float32 array in [0, 1], same spatial shape.
    """
    # Defensive: accept [0, 255] input from tf.data loaders
    if image_np.max() > 1.5:
        image_np = image_np / 255.0
    image_np = image_np.astype(np.float32)

    # Ensure 3-channel RGB — some MRI scans are grayscale
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[-1] == 1:
        image_np = np.concatenate([image_np] * 3, axis=-1)

    # Always apply deterministic preprocessing
    image_np = apply_clahe(image_np)
    image_np = zscore_normalize(image_np)

    if not training:
        return image_np

    # --- Probabilistic augmentations (training only) ---
    if np.random.rand() < 0.50:
        image_np = add_gaussian_noise(image_np, std=0.015)

    if np.random.rand() < 0.30:
        image_np = apply_random_blur(image_np, max_sigma=0.8)

    if np.random.rand() < 0.40:
        image_np = elastic_transform(image_np, alpha=25.0, sigma=4.0)

    if np.random.rand() < 0.40:
        image_np = random_brightness_contrast(
            image_np, brightness_range=0.12, contrast_range=0.12
        )

    return image_np
