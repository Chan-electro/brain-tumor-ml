"""
finetune.py
-----------
Local fine-tune script for Brain Tumor Classifier.

Loads the existing best_model.keras and continues training with
gradual unfreezing to squeeze out extra performance.

Usage:
    python scripts/finetune.py

Requirements:
    - pip install -r requirements.txt
    - models/best_model.keras must exist
    - data/train, data/val directories must exist
"""

import os
import sys
import json
import math
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, regularizers

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODEL_IN    = ROOT / "models" / "best_model.keras"
MODEL_OUT   = ROOT / "models" / "best_model.keras"     # overwrite in-place
MODEL_BKUP  = ROOT / "models" / "best_model_pretune.keras"
LABEL_MAP   = ROOT / "models" / "label_map.json"
DATA_DIR    = ROOT / "data"
LOGS_DIR    = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # must match inference pipeline
BATCH_SIZE  = 8            # keep small for CPU — increase to 16+ if using GPU
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Fine-tune phases
FINETUNE_PHASES = [
    # (fraction_unfrozen, lr,   epochs)
    (0.25,               2e-5,  8),   # Phase A: top 25% unfrozen
    (0.50,               1e-5,  8),   # Phase B: top 50% unfrozen
    (1.00,               5e-6,  5),   # Phase C: full model (BN frozen)
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_dataset(split: str, shuffle: bool = False):
    """Load a split from data/{split}/ into a tf.data.Dataset."""
    ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / split),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED if shuffle else None,
        class_names=CLASS_NAMES,
    )
    # EfficientNet preprocessing
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def unfreeze_top_fraction(model, fraction: float):
    """Unfreeze the top `fraction` of backbone layers (BatchNorm stays frozen)."""
    # Find the EfficientNet backbone sub-model
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is None:
        print("  [!] No backbone sub-model found — unfreezing entire model top fraction")
        target = model
    else:
        target = backbone

    target.trainable = True
    total = len(target.layers)
    freeze_until = int(total * (1.0 - fraction))

    frozen_count = 0
    trainable_count = 0
    for i, layer in enumerate(target.layers):
        if i < freeze_until or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_count += 1
        else:
            layer.trainable = True
            trainable_count += 1

    print(f"  Backbone: {trainable_count} trainable / {frozen_count} frozen "
          f"({fraction*100:.0f}% unfrozen)")


def make_callbacks(phase_label: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            min_delta=0.001,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(LOGS_DIR / f"finetune_{phase_label}.csv"),
            append=False,
        ),
    ]


def compute_class_weights(train_ds):
    """Compute balanced class weights from the training dataset."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = []
    for _, batch_labels in train_ds.unbatch():
        labels.append(int(np.argmax(batch_labels.numpy())))
    labels = np.array(labels)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=labels)
    weights = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", {CLASS_NAMES[k]: round(v, 3) for k, v in weights.items()})
    return weights


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Brain Tumor Classifier — Local Fine-Tune")
    print("=" * 60)

    # 1. Verify prerequisites
    if not MODEL_IN.exists():
        print(f"\n[ERROR] Model not found at: {MODEL_IN}")
        print("Please train the model first using the Kaggle notebook.")
        sys.exit(1)

    for split in ["train", "val"]:
        if not (DATA_DIR / split).exists():
            print(f"[ERROR] Missing data split: {DATA_DIR / split}")
            sys.exit(1)

    # 2. Backup original model
    if not MODEL_BKUP.exists():
        import shutil
        shutil.copy2(MODEL_IN, MODEL_BKUP)
        print(f"\nBackup saved → {MODEL_BKUP.name}")
    else:
        print(f"\nBackup already exists: {MODEL_BKUP.name} (skipping)")

    # 3. Load model
    print(f"\nLoading model: {MODEL_IN.name} ...", flush=True)
    model = tf.keras.models.load_model(str(MODEL_IN))
    print("Model loaded.")

    # 4. Build datasets (without augmentation for fast CPU run)
    print("\nBuilding data pipelines ...")
    # Raw dataset (no custom augmentation to keep CPU time reasonable)
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
        class_names=CLASS_NAMES,
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=CLASS_NAMES,
    )

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Light augmentation layer (built-in Keras, runs on-graph)
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    def prep_train(x, y):
        x = augment(x, training=True)
        return preprocess(x), y

    def prep_val(x, y):
        return preprocess(x), y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds_raw
                .map(prep_train, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds   = (val_ds_raw
                .map(prep_val, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))

    # Count samples for reporting
    train_count = sum(
        len(list((DATA_DIR / "train" / c).glob("*")))
        for c in CLASS_NAMES if (DATA_DIR / "train" / c).exists()
    )
    val_count = sum(
        len(list((DATA_DIR / "val" / c).glob("*")))
        for c in CLASS_NAMES if (DATA_DIR / "val" / c).exists()
    )
    print(f"  Train: {train_count} images | Val: {val_count} images")

    # 5. Class weights
    print("\nComputing class weights ...")
    class_weights = compute_class_weights(val_ds_raw)

    # 6. Evaluate baseline
    print("\n── Baseline evaluation (before fine-tuning) ──")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"  Val accuracy: {acc*100:.2f}%  |  Val loss: {loss:.4f}")
    best_acc = acc

    # 7. Fine-tune phases
    for phase_idx, (fraction, lr, epochs) in enumerate(FINETUNE_PHASES, start=1):
        phase_label = f"phase_{phase_idx}"
        print(f"\n{'='*60}")
        print(f"  Fine-tune Phase {phase_idx}: "
              f"top {fraction*100:.0f}% unfrozen | lr={lr} | max {epochs} epochs")
        print("=" * 60)

        unfreeze_top_fraction(model, fraction)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=make_callbacks(phase_label),
            class_weight=class_weights,
            verbose=1,
        )

        phase_best = max(history.history["val_accuracy"])
        print(f"\n  Phase {phase_idx} best val accuracy: {phase_best*100:.2f}%")
        if phase_best > best_acc:
            best_acc = phase_best
            print(f"  ✅ New overall best: {best_acc*100:.2f}%")
        else:
            print(f"  Previous best still holds: {best_acc*100:.2f}%")

    # 8. Final evaluation
    print("\n── Final evaluation (after fine-tuning) ──")
    final_model = tf.keras.models.load_model(str(MODEL_OUT))
    final_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    loss, acc = final_model.evaluate(val_ds, verbose=0)
    print(f"  Final val accuracy: {acc*100:.2f}%  |  Val loss: {loss:.4f}")

    print("\n" + "=" * 60)
    print(f"  Fine-tuning complete!")
    print(f"  Best model saved to: {MODEL_OUT}")
    print(f"  Original backed up at: {MODEL_BKUP.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
