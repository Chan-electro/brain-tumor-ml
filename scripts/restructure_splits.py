"""
restructure_splits.py
---------------------
Pools all images from data/train/ and data/val/, then creates a proper
stratified 70% / 15% / 15% (train / val / test) split per class.

Run once from the project root:
    python scripts/restructure_splits.py

WARNING: This script overwrites the existing train/ and val/ folders and
creates a new test/ folder. Back up your data/ directory before running.
"""

import os
import random
import shutil
from pathlib import Path

SEED = 42
DATA_ROOT = Path("data")
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remainder to avoid rounding gaps

random.seed(SEED)


def collect_all_images(cls: str) -> list:
    """Pool images from both existing train/ and val/ for a given class."""
    all_files = []
    for split in ["train", "val"]:
        folder = DATA_ROOT / split / cls
        if folder.exists():
            all_files.extend(
                list(folder.glob("*.jpg"))
                + list(folder.glob("*.jpeg"))
                + list(folder.glob("*.png"))
            )
    return all_files


def copy_files(files: list, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


def main():
    print("=" * 60)
    print("Brain Tumor Dataset — Stratified 70/15/15 Resplit")
    print("=" * 60)

    # Step 1: Collect all images per class into a temp staging area
    staged = {}
    for cls in CLASSES:
        files = collect_all_images(cls)
        random.shuffle(files)
        staged[cls] = files
        print(f"  {cls}: {len(files)} total images found")

    print()

    # Step 2: Copy files to a temporary staging directory so we don't
    # corrupt the source while reading from it
    staging_dir = DATA_ROOT / "_staging"
    print("Staging images to temporary directory...")
    for cls, files in staged.items():
        stage_cls = staging_dir / cls
        stage_cls.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, stage_cls / f.name)

    # Step 3: Remove old train/ and val/ directories
    print("Removing old train/ and val/ directories...")
    for split in ["train", "val"]:
        split_dir = DATA_ROOT / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    # Step 4: Create new splits from staging
    print("\nCreating new splits:")
    total_counts = {"train": 0, "val": 0, "test": 0}

    for cls in CLASSES:
        files = sorted((staging_dir / cls).glob("*"))
        files = [f for f in files if f.is_file()]
        n = len(files)

        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        # test gets remainder
        splits = {
            "train": files[:n_train],
            "val":   files[n_train : n_train + n_val],
            "test":  files[n_train + n_val :],
        }

        print(f"  {cls}: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

        for split_name, split_files in splits.items():
            copy_files(split_files, DATA_ROOT / split_name / cls)
            total_counts[split_name] += len(split_files)

    # Step 5: Remove staging directory
    shutil.rmtree(staging_dir)

    print()
    print("Split summary:")
    for split_name, count in total_counts.items():
        print(f"  {split_name}: {count} images total")
    print()
    print("Done! The test/ set is your held-out evaluation set.")
    print("Never tune hyperparameters based on test set results.")


if __name__ == "__main__":
    main()
