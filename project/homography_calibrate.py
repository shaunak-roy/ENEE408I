# homography_calibrate.py — Compute the cam2→cam1 homography matrix
#
# Run from project/ directory:
#   python3 homography_calibrate.py
#
# Workflow:
#   1. Both camera feeds open.
#   2. Press SPACE to freeze both frames simultaneously.
#   3. Click 4+ corresponding physical points (tape marks, conveyor corners, etc.)
#      on EACH window in the same order.
#   4. Press 'c' to compute homography and see reprojection error.
#   5. Press 's' to save, 'r' to reset and retry, 'q' to quit.
#
# Output: homography_cam2_to_cam1.npy (used by dual_camera_live.py)
#
# Note: cam1 frames are undistorted before display so that the saved H maps
# from raw cam2 space → undistorted cam1 space, matching the live inference pipeline.

import cv2
import numpy as np
import os

# ===== CONFIG =====
CAM1_INDEX = 0
CAM2_INDEX = 4
CAM1_K_PATH = "camera1_topdown/cam_K.npy"
CAM1_D_PATH = "camera1_topdown/cam_D.npy"
OUTPUT = "homography_cam2_to_cam1.npy"
MIN_POINTS = 4
# ==================

cam1_pts = []
cam2_pts = []
frozen = False
frozen1 = None
frozen2 = None
undistort_maps = None  # (map1, map2) for cam1 fisheye undistortion


def build_undistort_maps(K, D, w, h):
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2


def undistort(frame, maps):
    if maps is None:
        return frame
    return cv2.remap(frame, maps[0], maps[1],
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def draw_pts(img, pts, color):
    out = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), 7, color, -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 2)
        cv2.putText(out, str(i + 1), (int(x) + 10, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def mouse_cam1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frozen:
        cam1_pts.append([x, y])
        print(f"  Cam1 point {len(cam1_pts)}: ({x}, {y})")


def mouse_cam2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frozen:
        cam2_pts.append([x, y])
        print(f"  Cam2 point {len(cam2_pts)}: ({x}, {y})")


def reprojection_error(H, src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    dst = np.array(dst_pts, dtype=np.float32)
    errs = np.linalg.norm(projected - dst, axis=1)
    return errs.mean(), errs.max()


def main():
    global frozen, frozen1, frozen2, cam1_pts, cam2_pts, undistort_maps

    cam1 = cv2.VideoCapture(CAM1_INDEX)
    cam2 = cv2.VideoCapture(CAM2_INDEX)

    if not cam1.isOpened():
        print(f"ERROR: Cannot open camera {CAM1_INDEX}")
        return
    if not cam2.isOpened():
        print(f"ERROR: Cannot open camera {CAM2_INDEX}")
        return

    for cam in [cam1, cam2]:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load undistortion maps for cam1
    if os.path.exists(CAM1_K_PATH) and os.path.exists(CAM1_D_PATH):
        K = np.load(CAM1_K_PATH)
        D = np.load(CAM1_D_PATH)
        ret, frame = cam1.read()
        if ret:
            h, w = frame.shape[:2]
            undistort_maps = build_undistort_maps(K, D, w, h)
            print(f"Cam1 undistortion maps loaded ({w}x{h})")
    else:
        print("WARNING: cam_K.npy/cam_D.npy not found — cam1 frames won't be undistorted.")

    cv2.namedWindow("Camera 1 (top-down — click here first)")
    cv2.namedWindow("Camera 2 (angled — click here second)")
    cv2.setMouseCallback("Camera 1 (top-down — click here first)", mouse_cam1)
    cv2.setMouseCallback("Camera 2 (angled — click here second)", mouse_cam2)

    print("\nInstructions:")
    print("  SPACE  — freeze/unfreeze frames")
    print("  Click  — add corresponding point (same physical point, same order in each window)")
    print("  c      — compute homography")
    print("  s      — save homography_cam2_to_cam1.npy")
    print("  r      — reset points")
    print("  q      — quit")
    print(f"\nNeed at least {MIN_POINTS} point pairs.")

    H = None

    while True:
        if not frozen:
            ret1, f1 = cam1.read()
            ret2, f2 = cam2.read()
            if ret1:
                frozen1 = undistort(f1, undistort_maps)
            if ret2:
                frozen2 = f2.copy()

        disp1 = draw_pts(frozen1, cam1_pts, (0, 255, 255)) if frozen1 is not None \
            else np.zeros((480, 640, 3), np.uint8)
        disp2 = draw_pts(frozen2, cam2_pts, (255, 100, 50)) if frozen2 is not None \
            else np.zeros((480, 640, 3), np.uint8)

        hud = (f"{'FROZEN' if frozen else 'LIVE'} | "
               f"Cam1: {len(cam1_pts)} pts | Cam2: {len(cam2_pts)} pts"
               + (f" | H ready — press s to save" if H is not None else ""))
        cv2.putText(disp1, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(disp2, "Click same points in the same order as cam1",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

        cv2.imshow("Camera 1 (top-down — click here first)", disp1)
        cv2.imshow("Camera 2 (angled — click here second)", disp2)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            frozen = not frozen
            if frozen:
                print(f"\nFrozen. Click {MIN_POINTS}+ matching points on each window.")
            else:
                print("Unfrozen.")

        elif key == ord('c'):
            n1, n2 = len(cam1_pts), len(cam2_pts)
            if n1 < MIN_POINTS or n2 < MIN_POINTS:
                print(f"Need at least {MIN_POINTS} points on each. Got cam1={n1}, cam2={n2}.")
            elif n1 != n2:
                print(f"Point count mismatch: cam1={n1}, cam2={n2}. They must be equal.")
            else:
                src = np.array(cam2_pts, dtype=np.float32)
                dst = np.array(cam1_pts, dtype=np.float32)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    inliers = int(mask.sum())
                    mean_err, max_err = reprojection_error(H, src, dst)
                    print(f"\nHomography OK — {inliers}/{n1} RANSAC inliers")
                    print(f"Reprojection error: mean={mean_err:.2f}px  max={max_err:.2f}px")
                    if mean_err > 20:
                        print("WARNING: high error — consider re-clicking with more spread-out points.")
                    print("Press 's' to save or 'r' to reset.")
                else:
                    print("ERROR: findHomography failed. Try different or more spread-out points.")

        elif key == ord('s'):
            if H is not None:
                np.save(OUTPUT, H)
                print(f"Saved: {OUTPUT}")
            else:
                print("No homography yet. Press 'c' first.")

        elif key == ord('r'):
            cam1_pts.clear()
            cam2_pts.clear()
            frozen = False
            H = None
            print("Reset.")

        elif key == ord('q'):
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
