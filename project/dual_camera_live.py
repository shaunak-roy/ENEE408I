# dual_camera_live.py — Combined detection: package spacing (cam1) + stacked (cam2→cam1)
#
# Run from project/ directory:
#   python3 dual_camera_live.py
#
# Prerequisites:
#   - Trained models at camera1_topdown/runs/detect/camera1_topdown/weights/best.pt
#                     and camera2_angled/runs/detect/camera2_angled/weights/best.pt
#   - homography_cam2_to_cam1.npy  (run homography_calibrate.py once)
#   - camera1_topdown/cam_K.npy + cam_D.npy  (fisheye undistortion for cam1)
#
# What it does:
#   Camera 1 (top-down):
#     - Detects all packages with YOLO
#     - Marks pairs with a gap smaller than SPACING_THRESHOLD_RATIO * avg_box_width as "too close"
#   Camera 2 (angled):
#     - Detects stacked packages with YOLO
#     - Maps each stacked detection center through H → cam1 pixel space
#     - Finds the cam1 package box that contains (or is nearest to) that mapped point
#     - Highlights it as "stacked" on the cam1 display
#   Final output: single annotated cam1 window
#
# Keybindings while running:
#   q  — quit
#   +/- — raise/lower spacing threshold (shown in window title)

import cv2
import numpy as np
import os
from itertools import combinations

from ultralytics import YOLO

# ===== CONFIG =====
CAM1_INDEX   = 0
CAM2_INDEX   = 4
MODEL1_PATH  = "camera1_topdown/runs/detect/camera1_topdown/weights/best.pt"
MODEL2_PATH  = "camera2_angled/runs/detect/camera2_angled/weights/best.pt"
HOMOGRAPHY_PATH = "homography_cam2_to_cam1.npy"
CAM1_K_PATH  = "camera1_topdown/cam_K.npy"
CAM1_D_PATH  = "camera1_topdown/cam_D.npy"

CONF_THRESHOLD = 0.25
DEVICE = 0                    # set to 'cpu' if no CUDA GPU

# Spacing: gap between nearest box edges < this fraction of avg box width → "too close"
SPACING_THRESHOLD_RATIO = 0.15

# How far a mapped cam2 point can be outside a box and still match it (pixels).
# Increase if your homography has some error. Set to 0 for strict containment only.
STACKED_MATCH_MARGIN = 40
# Maximum center-distance for fallback matching when no box contains the mapped point.
STACKED_FALLBACK_MAX_DIST = 200
# ==================

# Colors (BGR)
_C_PACKAGE      = (0, 200, 0)
_C_POOR_SPACING = (0, 140, 255)   # orange
_C_STACKED      = (0, 0, 230)     # red
_C_BOTH         = (160, 0, 200)   # purple — stacked AND too close


# ── Undistortion ─────────────────────────────────────────────────────────────

def build_undistort_maps(K, D, w, h):
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    return m1, m2


def apply_undistort(frame, maps):
    if maps is None:
        return frame
    return cv2.remap(frame, maps[0], maps[1],
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ── Spacing detection ─────────────────────────────────────────────────────────

def _edge_gap(b1, b2):
    """Euclidean distance between nearest edges. Returns 0 if boxes overlap."""
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    gx = max(0.0, max(x1a, x1b) - min(x2a, x2b))
    gy = max(0.0, max(y1a, y1b) - min(y2a, y2b))
    return (gx * gx + gy * gy) ** 0.5


def find_spacing_pairs(boxes, threshold_ratio):
    """Return set of (i,j) index pairs whose edge gap < threshold_ratio * avg_box_width."""
    if len(boxes) < 2:
        return set()
    avg_w = float(np.mean([b[2] - b[0] for b in boxes]))
    thresh = threshold_ratio * avg_w
    return {(i, j)
            for i, j in combinations(range(len(boxes)), 2)
            if _edge_gap(boxes[i], boxes[j]) < thresh}


# ── Homography mapping ────────────────────────────────────────────────────────

def map_point(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def find_matching_box(px, py, boxes, margin, fallback_max_dist):
    """
    Return index of the box that best matches mapped point (px, py).

    First tries boxes expanded by `margin` pixels (handles homography error).
    Falls back to nearest box center if within fallback_max_dist.
    Returns None if nothing is close enough.
    """
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin):
            return i

    if not boxes:
        return None
    cx_arr = np.array([(b[0] + b[2]) / 2 for b in boxes])
    cy_arr = np.array([(b[1] + b[3]) / 2 for b in boxes])
    dists = np.hypot(cx_arr - px, cy_arr - py)
    idx = int(np.argmin(dists))
    return idx if dists[idx] < fallback_max_dist else None


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_box_with_label(img, x1, y1, x2, y2, text, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_hud(img, n_pkg, n_close, n_stacked, thresh_ratio):
    lines = [
        f"packages: {n_pkg}",
        f"too close: {n_close}  (gap thresh: {thresh_ratio:.2f}x width)",
        f"stacked: {n_stacked}",
        "q=quit  +/-=threshold",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spacing_ratio = SPACING_THRESHOLD_RATIO

    model1 = YOLO(MODEL1_PATH)
    model2 = YOLO(MODEL2_PATH)

    # Homography
    H = None
    if os.path.exists(HOMOGRAPHY_PATH):
        H = np.load(HOMOGRAPHY_PATH)
        print(f"Loaded homography: {HOMOGRAPHY_PATH}")
    else:
        print(f"WARNING: {HOMOGRAPHY_PATH} not found. "
              "Run homography_calibrate.py first to enable stacked cross-mapping.")

    # Cam1 undistortion maps
    undistort_maps = None
    if os.path.exists(CAM1_K_PATH) and os.path.exists(CAM1_D_PATH):
        K = np.load(CAM1_K_PATH)
        D = np.load(CAM1_D_PATH)
        # Build maps from a live frame so we have the correct resolution
        cam_probe = cv2.VideoCapture(CAM1_INDEX)
        cam_probe.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam_probe.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, probe_frame = cam_probe.read()
        cam_probe.release()
        if ret:
            h, w = probe_frame.shape[:2]
            undistort_maps = build_undistort_maps(K, D, w, h)
            print(f"Cam1 undistortion maps ready ({w}x{h})")
    else:
        print("WARNING: cam_K.npy/cam_D.npy not found — cam1 frames won't be undistorted.")

    # Open cameras
    cam1 = cv2.VideoCapture(CAM1_INDEX)
    if not cam1.isOpened():
        print(f"ERROR: Cannot open camera {CAM1_INDEX}")
        return

    cam2 = None
    if H is not None:
        cam2 = cv2.VideoCapture(CAM2_INDEX)
        if not cam2.isOpened():
            print(f"WARNING: Cannot open camera {CAM2_INDEX}. Stacked detection disabled.")
            cam2 = None

    for cam in [cam1, cam2]:
        if cam:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "Camera 1 — Package Detection"
    print(f"\nRunning. Press 'q' to quit, '+'/'-' to adjust spacing threshold.")

    while True:
        ret1, raw1 = cam1.read()
        if not ret1:
            continue

        frame1 = apply_undistort(raw1, undistort_maps)

        # ── Cam1: package detection ──
        res1 = model1(frame1, device=DEVICE, conf=CONF_THRESHOLD, verbose=False)
        package_boxes = []   # (x1, y1, x2, y2) of class 'package' only
        for box in res1[0].boxes:
            if model1.names[int(box.cls[0])] == 'package':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                package_boxes.append((x1, y1, x2, y2))

        # ── Cam2: stacked detection → map to cam1 ──
        stacked_indices = set()
        if cam2 is not None and H is not None:
            ret2, frame2 = cam2.read()
            if ret2:
                res2 = model2(frame2, device=DEVICE, conf=CONF_THRESHOLD, verbose=False)
                for box in res2[0].boxes:
                    if model2.names[int(box.cls[0])] == 'stacked':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx2 = (x1 + x2) / 2
                        cy2 = (y1 + y2) / 2
                        px, py = map_point(H, cx2, cy2)
                        idx = find_matching_box(px, py, package_boxes,
                                                STACKED_MATCH_MARGIN,
                                                STACKED_FALLBACK_MAX_DIST)
                        if idx is not None:
                            stacked_indices.add(idx)

        # ── Spacing analysis ──
        spacing_pairs = find_spacing_pairs(package_boxes, spacing_ratio)
        spacing_indices = {i for pair in spacing_pairs for i in pair}

        # ── Annotate cam1 frame ──
        annotated = frame1.copy()
        for i, (x1, y1, x2, y2) in enumerate(package_boxes):
            is_stacked = i in stacked_indices
            is_close   = i in spacing_indices
            if is_stacked and is_close:
                color, label = _C_BOTH, "stacked+close"
            elif is_stacked:
                color, label = _C_STACKED, "stacked"
            elif is_close:
                color, label = _C_POOR_SPACING, "too close"
            else:
                color, label = _C_PACKAGE, "package"
            draw_box_with_label(annotated, x1, y1, x2, y2, label, color)

        draw_hud(annotated,
                 n_pkg=len(package_boxes),
                 n_close=len(spacing_indices),
                 n_stacked=len(stacked_indices),
                 thresh_ratio=spacing_ratio)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            spacing_ratio = min(spacing_ratio + 0.05, 1.0)
            print(f"Spacing threshold: {spacing_ratio:.2f}")
        elif key == ord('-'):
            spacing_ratio = max(spacing_ratio - 0.05, 0.0)
            print(f"Spacing threshold: {spacing_ratio:.2f}")
        elif cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam1.release()
    if cam2:
        cam2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
