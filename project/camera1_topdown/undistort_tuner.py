"""
Live Fisheye Undistortion Tuner
================================
Reads K and D from cam_K.npy and cam_D.npy.
Tunes k1 and cx/cy optically (saved back to cam_K/D.npy).
Zoom and balance are display-only and saved separately.

Press S to save. Press Q to quit.
"""

import cv2
import numpy as np

# ── Edit these ──────────────────────────────────────────────────────
CAM    = 1
K_PATH = "cam_K.npy"
D_PATH = "cam_D.npy"

# Starting slider positions
START_BALANCE   =   0   # 0-100  display only
START_ZOOM      =  100   # 0-100  display only
START_CX_OFFSET = 200   # 0-400  (200 = no offset) — optical, saved to K
START_CY_OFFSET = 200   # 0-400  (200 = no offset) — optical, saved to K
START_K1_SCALE  = 0   # 0-200  (100 = no change) — optical, saved to D
# ────────────────────────────────────────────────────────────────────

WIN_ORIG  = "Original (fisheye)"
WIN_UNDST = "Undistorted - S=save  Q=quit"


def build_maps(K, D, w, h, balance, zoom, cx_offset, cy_offset, k1_scale):
    # These are the TRUE optical values — cx/cy offset and k1 scale are real corrections
    K_mod = K.copy()
    K_mod[0, 2] += cx_offset
    K_mod[1, 2] += cy_offset

    D_mod = D.copy()
    D_mod[0, 0] *= k1_scale

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_mod, D_mod, (w, h), np.eye(3), balance=balance
    )
    # Zoom is display-only — applied to new_K only, not saved
    new_K[0, 0] *= zoom
    new_K[1, 1] *= zoom

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_mod, D_mod, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2, K_mod, D_mod


def main():
    K = np.load(K_PATH)
    D = np.load(D_PATH)
    print("Loaded K:\n", K)
    print("Loaded D:\n", D.ravel())

    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM}")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from camera")
    h, w = frame.shape[:2]
    print(f"Camera {CAM} opened at {w}x{h}\n")

    cv2.namedWindow(WIN_ORIG,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_UNDST, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Balance x100 (display)",  WIN_UNDST, START_BALANCE,   100, lambda x: None)
    cv2.createTrackbar("Zoom x100    (display)",  WIN_UNDST, START_ZOOM,      100, lambda x: None)
    cv2.createTrackbar("Cx offset    (optical)",  WIN_UNDST, START_CX_OFFSET, 400, lambda x: None)
    cv2.createTrackbar("Cy offset    (optical)",  WIN_UNDST, START_CY_OFFSET, 400, lambda x: None)
    cv2.createTrackbar("k1 scale x100 (optical)", WIN_UNDST, START_K1_SCALE,  200, lambda x: None)

    map1, map2, _, _ = build_maps(K, D, w, h,
                                   balance=START_BALANCE / 100.0,
                                   zoom=max(START_ZOOM / 100.0, 0.1),
                                   cx_offset=START_CX_OFFSET - 200,
                                   cy_offset=START_CY_OFFSET - 200,
                                   k1_scale=START_K1_SCALE / 100.0)
    prev_params = None
    frame_count = 0
    K_mod = K.copy()
    D_mod = D.copy()

    print("Optical sliders (Cx, Cy, k1) → saved to cam_K/D.npy when you press S")
    print("Display sliders (Balance, Zoom) → not saved, just for viewing\n")
    print("S=save  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed")
            break

        balance   = cv2.getTrackbarPos("Balance x100 (display)",  WIN_UNDST) / 100.0
        zoom      = max(cv2.getTrackbarPos("Zoom x100    (display)", WIN_UNDST) / 100.0, 0.1)
        cx_offset = cv2.getTrackbarPos("Cx offset    (optical)",  WIN_UNDST) - 200
        cy_offset = cv2.getTrackbarPos("Cy offset    (optical)",  WIN_UNDST) - 200
        k1_scale  = cv2.getTrackbarPos("k1 scale x100 (optical)", WIN_UNDST) / 100.0

        params = (balance, zoom, cx_offset, cy_offset, k1_scale)

        if params != prev_params:
            map1, map2, K_mod, D_mod = build_maps(
                K, D, w, h, balance, zoom, cx_offset, cy_offset, k1_scale
            )
            prev_params = params
            print(f"\rbalance={balance:.2f}(display)  zoom={zoom:.2f}(display)  "
                  f"cx={K_mod[0,2]:.1f}  cy={K_mod[1,2]:.1f}  "
                  f"k1={D_mod[0,0]:.5f}  ", end="", flush=True)

        undist = cv2.remap(frame, map1, map2,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)

        overlay = undist.copy()
        cv2.putText(overlay,
                    f"cx={K_mod[0,2]:.1f}  cy={K_mod[1,2]:.1f}  k1={D_mod[0,0]:.5f}"
                    f"  balance={balance:.2f}  zoom={zoom:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, "OPTICAL: cx/cy/k1 saved on S   DISPLAY: balance/zoom not saved",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        cv2.imshow(WIN_ORIG,  frame)
        cv2.imshow(WIN_UNDST, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save only the TRUE optical K and D (cx/cy and k1 corrections)
            # Do NOT save zoom or balance — those are display only
            np.save(K_PATH, K_mod)
            np.save(D_PATH, D_mod)
            np.savetxt(K_PATH.replace(".npy", ".txt"), K_mod, fmt="%.8f")
            np.savetxt(D_PATH.replace(".npy", ".txt"), D_mod.ravel(), fmt="%.8f",
                       header="k1 k2 k3 k4")

            # Save display params separately
            np.savetxt("display_params.txt",
                       [balance, zoom], fmt="%.4f",
                       header="balance zoom (display only, not optical)")

            orig_path  = f"saved_original_{frame_count:04d}.png"
            undst_path = f"saved_undistorted_{frame_count:04d}.png"
            cv2.imwrite(orig_path,  frame)
            cv2.imwrite(undst_path, undist)

            print(f"\n\nSaved optical K  → {K_PATH} + {K_PATH.replace('.npy','.txt')}")
            print(f"Saved optical D  → {D_PATH} + {D_PATH.replace('.npy','.txt')}")
            print(f"Saved display    → display_params.txt")
            print(f"Saved frames     → {orig_path}, {undst_path}")
            print(f"\nOptical K:\n{K_mod}")
            print(f"Optical D: {D_mod.ravel()}")
            print(f"Display balance={balance}  zoom={zoom}  (NOT in K/D)\n")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()