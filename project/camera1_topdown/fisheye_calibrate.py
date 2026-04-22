"""
USBFHD01M Fisheye Camera Calibration Script
============================================
Edit the parameters at the top, then run:
  python3 calibrate_fisheye.py

Requirements: pip install opencv-contrib-python numpy
"""

import cv2
import numpy as np
import glob
import sys
import os


# ── Edit these ──────────────────────────────────────────────────────
CAM           = 0                       # camera index
OUT           = "calibration_data_raw"  # folder to save/read calibration images
N             = 30                      # number of images to capture
PREFIX        = "cam"                   # output prefix → cam_K.npy, cam_D.npy
BALANCE       = 0.0                     # undistort balance: 0=tight crop, 1=full FOV
IMAGE         = ""                      # test image for UNDISTORT — leave blank to auto-pick

CHECKERBOARD  = (8, 5)                  # inner corners: (cols-1, rows-1) of your printed board
SQUARE_SIZE   = 3.0                     # square size in cm
# ────────────────────────────────────────────────────────────────────


def calibrate(images_dir, prefix):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 1, 3), np.float64)
    objp[:, 0, :2] = (np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
                      .T.reshape(-1, 2) * SQUARE_SIZE)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    img_shape: tuple[int, int] | None = None
    rejected = 0

    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                   glob.glob(os.path.join(images_dir, "*.jpg")))

    if not paths:
        print(f"ERROR: No images found in '{images_dir}'")
        sys.exit(1)

    print(f"Processing {len(paths)} images...")

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  ✗  Could not read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"  ✓  {os.path.basename(path)}")
        else:
            rejected += 1
            print(f"  ✗  {os.path.basename(path)}  (corners not found)")

    if len(objpoints) < 10:
        print(f"\nERROR: Only {len(objpoints)} valid images (need ≥10). Recapture more.")
        sys.exit(1)

    print(f"\n{len(objpoints)} valid / {rejected} rejected")

    if img_shape is None:
        print("ERROR: No valid images found.")
        sys.exit(1)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
         cv2.fisheye.CALIB_FIX_SKEW             |
         cv2.fisheye.CALIB_FIX_K2               |
         cv2.fisheye.CALIB_FIX_K3               |
         cv2.fisheye.CALIB_FIX_K4)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D,
        rvecs=None, tvecs=None, flags=flags
    )

    print(f"\n{'='*50}")
    print(f"RMS reprojection error: {rms:.4f}  (good if < 1.0)")
    print(f"\nK (intrinsic matrix):\n{K}")
    print(f"\nD (distortion coefficients [k1,k2,k3,k4]):\n{D.ravel()}")
    print(f"{'='*50}\n")

    np.save(f"{prefix}_K.npy", K)
    np.save(f"{prefix}_D.npy", D)
    np.savetxt(f"{prefix}_K.txt", K, fmt="%.6f")
    np.savetxt(f"{prefix}_D.txt", D.ravel(), fmt="%.6f", header="k1 k2 k3 k4")

    print(f"Saved: {prefix}_K.npy / {prefix}_K.txt")
    print(f"Saved: {prefix}_D.npy / {prefix}_D.txt")

    if rms > 1.0:
        print("\nWARNING: RMS > 1.0 — consider recapturing with more varied board poses.")

if __name__ == "__main__":
    calibrate(OUT, PREFIX)