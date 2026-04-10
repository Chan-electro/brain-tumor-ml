"""
evaluation.py
-------------
Evaluation utilities for the brain tumor classifier.

Provides:
  - evaluate_on_split()         — batch inference, returns arrays
  - plot_confusion_matrix()     — raw + normalized heatmaps
  - plot_roc_curves()           — per-class ROC with AUC
  - plot_training_history()     — multi-phase accuracy + loss curves
  - print_classification_report() — sklearn report to stdout and file

All plots are saved to the reports/ directory at the project root.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

REPORTS_DIR = Path("reports")
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
_CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def evaluate_on_split(model, dataset, verbose: bool = True):
    """
    Run full inference on a tf.data dataset split.

    Args:
        model:   Loaded Keras model.
        dataset: tf.data.Dataset yielding (images, one-hot labels) batches.
        verbose: Print progress.

    Returns:
        y_true_idx   (N,)  — integer ground-truth class indices
        y_pred_idx   (N,)  — integer predicted class indices
        y_pred_probs (N, C) — softmax probability vectors
    """
    y_true_list, y_pred_list = [], []

    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images, verbose=0)
        y_pred_list.append(preds)
        y_true_list.append(batch_labels.numpy())

    y_true = np.concatenate(y_true_list, axis=0)   # (N, C) one-hot
    y_pred_probs = np.concatenate(y_pred_list, axis=0)  # (N, C)

    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    accuracy = np.mean(y_true_idx == y_pred_idx)
    if verbose:
        print(f"  Accuracy: {accuracy * 100:.2f}%  ({int(accuracy * len(y_true_idx))}/{len(y_true_idx)} correct)")

    return y_true_idx, y_pred_idx, y_pred_probs


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = CLASS_NAMES,
    split: str = "test",
):
    """
    Plot side-by-side raw count and row-normalized confusion matrices.

    Saves to reports/{split}_confusion_matrix.png.
    """
    _ensure_reports_dir()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — {split} set", fontsize=14, fontweight="bold")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_title("Row-Normalized (Recall per Class)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    save_path = REPORTS_DIR / f"{split}_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true_idx: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list = CLASS_NAMES,
    split: str = "test",
) -> float:
    """
    Plot one-vs-rest ROC curves for each class with individual AUC values
    and the macro-averaged AUC in the title.

    Saves to reports/{split}_roc_curves.png.

    Returns:
        macro_auc (float)
    """
    _ensure_reports_dir()
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    auc_values = []

    for i, (cls_name, color) in enumerate(zip(class_names, _CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{cls_name}  (AUC = {roc_auc:.3f})")

    macro_auc = float(np.mean(auc_values))

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curves — {split} set  |  Macro AUC = {macro_auc:.3f}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = REPORTS_DIR / f"{split}_roc_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}  (Macro AUC = {macro_auc:.3f})")
    plt.show()
    return macro_auc


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(histories: dict):
    """
    Plot accuracy and loss curves across multiple training phases as a
    single continuous line per metric.

    Args:
        histories: dict mapping phase label → Keras History object.
                   Example: {"Phase 1 (frozen)": h1, "Phase 2 (25%)": h2, ...}

    Saves to reports/training_history.png.
    """
    _ensure_reports_dir()
    phase_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History Across All Phases", fontsize=14, fontweight="bold")

    epoch_offset = 0
    for (phase_name, history), color in zip(histories.items(), phase_colors):
        hist = history.history
        n_epochs = len(hist["accuracy"])
        epochs = range(epoch_offset + 1, epoch_offset + n_epochs + 1)

        # Accuracy
        axes[0].plot(epochs, hist["accuracy"],
                     color=color, lw=2, label=f"{phase_name} (train)")
        axes[0].plot(epochs, hist["val_accuracy"],
                     color=color, lw=2, linestyle="--", label=f"{phase_name} (val)")

        # Loss
        axes[1].plot(epochs, hist["loss"],
                     color=color, lw=2, label=f"{phase_name} (train)")
        axes[1].plot(epochs, hist["val_loss"],
                     color=color, lw=2, linestyle="--", label=f"{phase_name} (val)")

        # Phase boundary vertical lines
        if epoch_offset > 0:
            axes[0].axvline(epoch_offset + 1, color="gray", lw=0.8, linestyle=":")
            axes[1].axvline(epoch_offset + 1, color="gray", lw=0.8, linestyle=":")

        epoch_offset += n_epochs

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = REPORTS_DIR / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = CLASS_NAMES,
    split: str = "test",
) -> str:
    """
    Print and save a sklearn classification report with per-class
    precision, recall, F1-score, and support.

    Saves to reports/{split}_classification_report.txt.

    Returns:
        report string
    """
    _ensure_reports_dir()
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(f"\n{'='*60}")
    print(f"Classification Report — {split} set")
    print("=" * 60)
    print(report)

    save_path = REPORTS_DIR / f"{split}_classification_report.txt"
    with open(save_path, "w") as f:
        f.write(f"Classification Report — {split} set\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  Saved: {save_path}")
    return report
