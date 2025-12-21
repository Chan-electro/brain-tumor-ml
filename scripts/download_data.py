import requests
import os
import json
import random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO

# Configuration
DATASET_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "benschill/brain-tumor-collection"
CONFIG = "original"
SPLIT = "train"
TOTAL_ROWS = 2870 # From API response
PAGE_SIZE = 100
VAL_SPLIT = 0.2
OUTPUT_DIR = "data"

# Class mapping from API features
# 0: Glioma Tumor, 1: Meningioma Tumor, 2: Pituitary Tumor, 3: No Tumor
# We will simplify names for folders
CLASS_MAPPING = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary",
    3: "no_tumor"
}

def create_dirs():
    for split in ["train", "val"]:
        for class_name in CLASS_MAPPING.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

def download_image(row):
    try:
        row_idx = row["row_idx"]
        data = row["row"]
        label_idx = data["label"]
        image_url = data["image"]["src"]
        
        class_name = CLASS_MAPPING.get(label_idx, "unknown")
        
        # Random split
        split = "val" if random.random() < VAL_SPLIT else "train"
        
        # Filename
        filename = f"{row_idx}.jpg"
        filepath = os.path.join(OUTPUT_DIR, split, class_name, filename)
        
        if os.path.exists(filepath):
            return # Skip if exists
            
        # Download
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            # Convert to RGB to avoid issues with some formats
            image = image.convert("RGB")
            image.save(filepath)
            # print(f"Saved {filepath}")
        else:
            print(f"Failed to download {image_url}: {response.status_code}")
            
    except Exception as e:
        print(f"Error processing row {row.get('row_idx')}: {e}")

def main():
    create_dirs()
    print("Created directories.")
    
    print(f"Fetching {TOTAL_ROWS} rows...")
    
    all_rows = []
    
    # Fetch all rows metadata first
    for offset in range(0, TOTAL_ROWS, PAGE_SIZE):
        print(f"Fetching metadata offset {offset}...")
        params = {
            "dataset": DATASET_NAME,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": PAGE_SIZE
        }
        try:
            response = requests.get(DATASET_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                rows = data.get("rows", [])
                all_rows.extend(rows)
            else:
                print(f"Error fetching metadata at offset {offset}: {response.status_code}")
        except Exception as e:
            print(f"Exception fetching metadata at offset {offset}: {e}")

    print(f"Total rows fetched: {len(all_rows)}")
    
    # Download images in parallel
    print("Downloading images...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_image, all_rows)
        
    print("Download complete.")

if __name__ == "__main__":
    main()
