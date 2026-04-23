# datasplit.py — Camera 1: split labeled data into train/valid/test (70/20/10)

import os
import shutil
import random

IMAGE_DIR = "./package_data_raw"
LABEL_DIR = "./package_data_labeled"
OUTPUT_BASE = "."

TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
paired = []
for img_file in image_files:
    label_file = os.path.splitext(img_file)[0] + ".txt"
    if os.path.exists(os.path.join(LABEL_DIR, label_file)):
        paired.append((img_file, label_file))

if not paired:
    print("No matched image-label pairs found!")
    exit(1)

print(f"Camera 2: found {len(paired)} labeled images")

random.seed(RANDOM_SEED)
random.shuffle(paired)

n = len(paired)
n_train = int(n * TRAIN_RATIO)
n_valid = int(n * VALID_RATIO)

splits = {
    "train": paired[:n_train],
    "valid": paired[n_train:n_train + n_valid],
    "test":  paired[n_train + n_valid:]
}

for split_name, pairs in splits.items():
    img_out = os.path.join(OUTPUT_BASE, split_name, "images")
    lbl_out = os.path.join(OUTPUT_BASE, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_file, label_file in pairs:
        shutil.copy2(os.path.join(IMAGE_DIR, img_file), os.path.join(img_out, img_file))
        shutil.copy2(os.path.join(LABEL_DIR, label_file), os.path.join(lbl_out, label_file))
    print(f"  {split_name}: {len(pairs)} images")

print("Camera 2 split complete.")
