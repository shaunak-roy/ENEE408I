# batch_undistort.py — Undistort all images in a folder using saved K and D
import cv2
import numpy as np
import os
import glob

# ===== CONFIG =====
INPUT_DIR  = "./package_data_raw"
OUTPUT_DIR = "./package_data_undistorted"
K_PATH     = "cam_K.npy"
D_PATH     = "cam_D.npy"
BALANCE    = 0.0                      # 0.0 = tight crop, 1.0 = full FOV
# ==================

K = np.load(K_PATH)
D = np.load(D_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
                glob.glob(os.path.join(INPUT_DIR, "*.png")))

if not images:
    print(f"No images found in {INPUT_DIR}")
    exit(1)

# Precompute maps from first image (all images must be same resolution)
first = cv2.imread(images[0])
h, w = first.shape[:2]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=BALANCE
)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
)

print(f"Undistorting {len(images)} images ({w}x{h}, balance={BALANCE})...")

for i, path in enumerate(images):
    img = cv2.imread(path)
    result = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    filename = os.path.basename(path)
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, result)

    print(f"  [{i+1}/{len(images)}] {filename}")

print(f"\nDone. Undistorted images saved to '{OUTPUT_DIR}/'")