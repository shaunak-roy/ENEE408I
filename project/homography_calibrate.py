import cv2
import numpy as np
import os

# ===== CONFIG =====
CAM1_INDEX = 4
CAM2_INDEX = 6
CAM1_K_PATH = "camera1_topdown/cam_K.npy"
CAM1_D_PATH = "camera1_topdown/cam_D.npy"
OUTPUT = "homography_cam2_to_cam1.npy"
MIN_POINTS = 4
BALANCE = 0.0
# ==================

cam1_pts = []
cam2_pts = []
frozen = False
frozen1 = None
frozen2 = None


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


def main():
    global frozen, frozen1, frozen2

    cam1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_V4L2)
    cam2 = cv2.VideoCapture(CAM2_INDEX, cv2.CAP_V4L2)

    if not cam1.isOpened():
        print(f"ERROR: Cannot open camera {CAM1_INDEX}")
        return
    if not cam2.isOpened():
        print(f"ERROR: Cannot open camera {CAM2_INDEX}")
        return

    # Precompute undistortion maps from first frame
    map1, map2 = None, None
    if os.path.exists(CAM1_K_PATH) and os.path.exists(CAM1_D_PATH):
        K = np.load(CAM1_K_PATH)
        D = np.zeros((4, 1))
        ret, frame = cam1.read()
        if ret:
            h, w = frame.shape[:2]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=BALANCE
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            print(f"Undistortion maps ready ({w}x{h}, balance={BALANCE})")
    else:
        print("WARNING: K/D not found — skipping undistortion.")

    cv2.namedWindow("Cam1")
    cv2.namedWindow("Cam2")
    cv2.setMouseCallback("Cam1", mouse_cam1)
    cv2.setMouseCallback("Cam2", mouse_cam2)

    print("\nSPACE=freeze  c=compute  s=save  r=reset  q=quit")
    print(f"Need {MIN_POINTS}+ point pairs.\n")

    H = None

    while True:
        if not frozen:
            ret1, f1 = cam1.read()
            ret2, f2 = cam2.read()
            if ret1:
                if map1 is not None:
                    frozen1 = cv2.remap(f1, map1, map2, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
                else:
                    frozen1 = f1.copy()
            if ret2:
                frozen2 = f2.copy()

        disp1 = draw_pts(frozen1, cam1_pts, (0, 255, 255)) if frozen1 is not None \
            else np.zeros((480, 640, 3), np.uint8)
        disp2 = draw_pts(frozen2, cam2_pts, (255, 100, 50)) if frozen2 is not None \
            else np.zeros((480, 640, 3), np.uint8)

        status = f"{'FROZEN' if frozen else 'LIVE'} | Cam1:{len(cam1_pts)} Cam2:{len(cam2_pts)}"
        if H is not None:
            status += " | H ready, press s"
        cv2.putText(disp1, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Cam1", disp1)
        cv2.imshow("Cam2", disp2)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            frozen = not frozen
            print("Frozen." if frozen else "Unfrozen.")

        elif key == ord('c'):
            n1, n2 = len(cam1_pts), len(cam2_pts)
            if n1 < MIN_POINTS or n2 < MIN_POINTS:
                print(f"Need {MIN_POINTS}+ points each. Got cam1={n1}, cam2={n2}.")
            elif n1 != n2:
                print(f"Mismatch: cam1={n1}, cam2={n2}.")
            else:
                src = np.array(cam2_pts, dtype=np.float32)
                dst = np.array(cam1_pts, dtype=np.float32)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    mean_err, max_err = reprojection_error(H, src, dst)
                    print(f"H OK — {int(mask.sum())}/{n1} inliers, err: {mean_err:.2f}px mean, {max_err:.2f}px max")
                else:
                    print("findHomography failed. Try more spread-out points.")

        elif key == ord('s'):
            if H is not None:
                np.save(OUTPUT, H)
                print(f"Saved: {OUTPUT}")
            else:
                print("No H yet. Press c first.")

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


def reprojection_error(H, src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    dst = np.array(dst_pts, dtype=np.float32)
    errs = np.linalg.norm(projected - dst, axis=1)
    return errs.mean(), errs.max()


if __name__ == "__main__":
    main()