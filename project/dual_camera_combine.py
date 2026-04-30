import cv2
import numpy as np
import os
from itertools import combinations
from ultralytics import YOLO

# ===== CONFIG =====
CAM1_INDEX   = 4
CAM2_INDEX   = 6
MODEL1_PATH  = "camera1_topdown/runs/detect/camera1_topdown/weights/best.pt"
MODEL2_PATH  = "camera2_angled/runs/detect/camera2_angled/weights/best.pt"
HOMOGRAPHY_PATH = "homography_cam2_to_cam1.npy"
CAM1_K_PATH  = "camera1_topdown/cam_K.npy"
CAM1_D_PATH  = "camera1_topdown/cam_D.npy"

CONF_THRESHOLD = 0.25
DEVICE = 0

SPACING_THRESHOLD_RATIO = 0.15
STACKED_MATCH_MARGIN = 40
STACKED_FALLBACK_MAX_DIST = 200
# ==================

_C_POOR_SPACING = (0, 140, 255)
_C_STACKED      = (0, 0, 230)
_C_BOTH         = (160, 0, 200)
_C_EMPTY        = (255, 200, 0)


def _edge_gap(b1, b2):
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    gx = max(0.0, max(x1a, x1b) - min(x2a, x2b))
    gy = max(0.0, max(y1a, y1b) - min(y2a, y2b))
    return (gx * gx + gy * gy) ** 0.5


def find_spacing_pairs(boxes, threshold_ratio):
    if len(boxes) < 2:
        return set()
    avg_w = float(np.mean([b[2] - b[0] for b in boxes]))
    thresh = threshold_ratio * avg_w
    return {(i, j)
            for i, j in combinations(range(len(boxes)), 2)
            if _edge_gap(boxes[i], boxes[j]) < thresh}


def map_point(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def find_matching_box(px, py, boxes, margin, fallback_max_dist):
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


def draw_box(img, x1, y1, x2, y2, text, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main():
    spacing_ratio = SPACING_THRESHOLD_RATIO

    model1 = YOLO(MODEL1_PATH)
    model2 = YOLO(MODEL2_PATH)

    H = None
    if os.path.exists(HOMOGRAPHY_PATH):
        H = np.load(HOMOGRAPHY_PATH)
        print(f"Loaded homography: {HOMOGRAPHY_PATH}")
    else:
        print(f"WARNING: {HOMOGRAPHY_PATH} not found.")

    cam1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_V4L2)
    cam2 = cv2.VideoCapture(CAM2_INDEX, cv2.CAP_V4L2)

    if not cam1.isOpened():
        print(f"ERROR: Cannot open camera {CAM1_INDEX}")
        return
    if not cam2.isOpened():
        print(f"ERROR: Cannot open camera {CAM2_INDEX}")
        return

    # Precompute undistortion maps
    umap1, umap2 = None, None
    if os.path.exists(CAM1_K_PATH):
        K = np.load(CAM1_K_PATH)
        D = np.zeros((4, 1))
        ret, probe = cam1.read()
        if ret:
            h, w = probe.shape[:2]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=0.0
            )
            umap1, umap2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            print(f"Undistortion ready ({w}x{h})")
    else:
        print("WARNING: K not found — skipping undistortion.")

    print("Running. q=quit  +/-=spacing threshold")

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1 or not ret2:
            continue

        if umap1 is not None:
            frame1 = cv2.remap(frame1, umap1, umap2, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)

        # Cam1: package + empty_space detection
        res1 = model1(frame1, device=DEVICE, conf=CONF_THRESHOLD, verbose=False)
        package_boxes = []
        empty_boxes = []
        for box in res1[0].boxes:
            cls_name = model1.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_name == 'package':
                package_boxes.append((x1, y1, x2, y2))
            elif cls_name == 'empty_space':
                empty_boxes.append((x1, y1, x2, y2))

        # Cam2: stacked detection
        stacked_indices = set()
        cam2_stacked_boxes = []
        if H is not None:
            res2 = model2(frame2, device=DEVICE, conf=CONF_THRESHOLD, verbose=False)
            for box in res2[0].boxes:
                if model2.names[int(box.cls[0])] == 'stacked':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cam2_stacked_boxes.append((x1, y1, x2, y2))
                    cx2 = (x1 + x2) / 2
                    cy2 = (y1 + y2) / 2
                    px, py = map_point(H, cx2, cy2)
                    idx = find_matching_box(px, py, package_boxes,
                                            STACKED_MATCH_MARGIN,
                                            STACKED_FALLBACK_MAX_DIST)
                    if idx is not None:
                        stacked_indices.add(idx)

        # Spacing analysis
        spacing_pairs = find_spacing_pairs(package_boxes, spacing_ratio)
        spacing_indices = {i for pair in spacing_pairs for i in pair}

        # Cam1 feed: show too-close and empty_space
        disp1 = frame1.copy()
        for i, (x1, y1, x2, y2) in enumerate(package_boxes):
            if i in spacing_indices:
                draw_box(disp1, x1, y1, x2, y2, "too close", _C_POOR_SPACING)
        for (x1, y1, x2, y2) in empty_boxes:
            draw_box(disp1, x1, y1, x2, y2, "empty", _C_EMPTY)

        # Cam2 feed: show stacked boxes
        disp2 = frame2.copy()
        for (x1, y1, x2, y2) in cam2_stacked_boxes:
            draw_box(disp2, x1, y1, x2, y2, "stacked", _C_STACKED)

        # Combined: show all issues
        combined = frame1.copy()
        for i, (x1, y1, x2, y2) in enumerate(package_boxes):
            is_stacked = i in stacked_indices
            is_close = i in spacing_indices
            if is_stacked and is_close:
                draw_box(combined, x1, y1, x2, y2, "stacked+close", _C_BOTH)
            elif is_stacked:
                draw_box(combined, x1, y1, x2, y2, "stacked", _C_STACKED)
            elif is_close:
                draw_box(combined, x1, y1, x2, y2, "too close", _C_POOR_SPACING)
        for (x1, y1, x2, y2) in empty_boxes:
            draw_box(combined, x1, y1, x2, y2, "empty", _C_EMPTY)

        cv2.imshow("Cam1 - Spacing", disp1)
        cv2.imshow("Cam2 - Stacked", disp2)
        cv2.imshow("Combined", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            spacing_ratio = min(spacing_ratio + 0.05, 1.0)
            print(f"Spacing threshold: {spacing_ratio:.2f}")
        elif key == ord('-'):
            spacing_ratio = max(spacing_ratio - 0.05, 0.0)
            print(f"Spacing threshold: {spacing_ratio:.2f}")

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()