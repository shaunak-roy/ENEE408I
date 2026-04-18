"""
Live Fisheye Undistortion Tuner - CUDA accelerated
====================================================
Reads K and D from cam_K.npy and cam_D.npy.
Uses CUDA for remap if available, falls back to CPU.
Optical sliders (fx, fy, cx, cy, k1, k2) saved to cam_K/D.npy on S.
Display sliders (balance, zoom) saved to display_params.txt only.

Press S to save. Press Q to quit.
Zoom slider: lower = more zoomed in, higher = more zoomed out
"""

import cv2
import numpy as np
import threading
import time

# ── Edit these ──────────────────────────────────────────────────────
CAM    = 0
K_PATH = "cam_K.npy"
D_PATH = "cam_D.npy"

START_BALANCE   =   0
START_ZOOM      =  100
START_CX_OFFSET = 200
START_CY_OFFSET = 200
START_FX_SCALE  = 100
START_FY_SCALE  = 100
START_K1_SCALE  = 100
START_K2_SCALE  = 100
# ────────────────────────────────────────────────────────────────────

WIN_ORIG  = "Original (fisheye)"
WIN_UNDST = "Undistorted - S=save  Q=quit"

CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA: {'YES' if CUDA_AVAILABLE else 'NO - falling back to CPU'}")


class CameraThread:
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fps_count = 0
        fps_start = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                fps_count += 1
                if fps_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    print(f"\rCamera FPS: {30/elapsed:.1f}  ", end="", flush=True)
                    fps_start = time.time()
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class UndistortThread:
    def __init__(self):
        self.input_frame  = None
        self.output_frame = None
        self.map1         = None
        self.map2         = None
        self.gpu_map1     = None
        self.gpu_map2     = None
        self.input_lock   = threading.Lock()
        self.output_lock  = threading.Lock()
        self.maps_lock    = threading.Lock()
        self.running      = True
        self.new_frame    = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()

            with self.input_lock:
                frame = self.input_frame
            with self.maps_lock:
                map1, map2 = self.map1, self.map2
                gpu_map1, gpu_map2 = self.gpu_map1, self.gpu_map2

            if frame is None or map1 is None or map2 is None:
                continue

            if CUDA_AVAILABLE and gpu_map1 is not None and gpu_map2 is not None:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_result = cv2.cuda.remap(gpu_frame, gpu_map1, gpu_map2,
                                            interpolation=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                result = gpu_result.download()
            else:
                result = cv2.remap(frame, map1, map2,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)

            with self.output_lock:
                self.output_frame = result

    def push_frame(self, frame):
        with self.input_lock:
            self.input_frame = frame
        self.new_frame.set()

    def update_maps(self, map1, map2):
        with self.maps_lock:
            self.map1 = map1
            self.map2 = map2
            if CUDA_AVAILABLE:
                self.gpu_map1 = cv2.cuda_GpuMat()
                self.gpu_map2 = cv2.cuda_GpuMat()
                self.gpu_map1.upload(map1)
                self.gpu_map2.upload(map2)

    def get_result(self):
        with self.output_lock:
            return self.output_frame.copy() if self.output_frame is not None else None

    def stop(self):
        self.running = False


def build_maps(K, D, w, h, balance, zoom,
               cx_offset, cy_offset,
               fx_scale, fy_scale,
               k1_scale, k2_scale):
    K_mod = K.copy()
    K_mod[0, 0] *= fx_scale
    K_mod[1, 1] *= fy_scale
    K_mod[0, 2] += cx_offset
    K_mod[1, 2] += cy_offset

    D_mod = D.copy()
    D_mod[0, 0] *= k1_scale
    D_mod[1, 0] *= k2_scale

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_mod, D_mod, (w, h), np.eye(3), balance=balance
    )
    new_K[0, 0] /= zoom
    new_K[1, 1] /= zoom

    map_type = cv2.CV_32FC1 if CUDA_AVAILABLE else cv2.CV_16SC2
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_mod, D_mod, np.eye(3), new_K, (w, h), map_type
    )
    return map1, map2, K_mod, D_mod


def main():
    K = np.load(K_PATH)
    D = np.load(D_PATH)
    print("Loaded K:\n", K)
    print("Loaded D:\n", D.ravel())

    cam      = CameraThread(CAM)
    undistor = UndistortThread()

    print("Waiting for camera...")
    while cam.read() is None:
        time.sleep(0.05)

    # Get actual camera resolution
    first_frame = cam.read()
    while first_frame is None:
        first_frame = cam.read()
    h, w = first_frame.shape[:2]
    print(f"Camera {CAM} running at {w}x{h}\n")

    cv2.namedWindow(WIN_ORIG,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_UNDST, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Balance  (display)", WIN_UNDST, START_BALANCE,   100, lambda x: None)
    cv2.createTrackbar("Zoom     (display)", WIN_UNDST, START_ZOOM,      100, lambda x: None)
    cv2.createTrackbar("Cx off   (optical)", WIN_UNDST, START_CX_OFFSET, 400, lambda x: None)
    cv2.createTrackbar("Cy off   (optical)", WIN_UNDST, START_CY_OFFSET, 400, lambda x: None)
    cv2.createTrackbar("fx scale (optical)", WIN_UNDST, START_FX_SCALE,  200, lambda x: None)
    cv2.createTrackbar("fy scale (optical)", WIN_UNDST, START_FY_SCALE,  200, lambda x: None)
    cv2.createTrackbar("k1 scale (optical)", WIN_UNDST, START_K1_SCALE,  200, lambda x: None)
    cv2.createTrackbar("k2 scale (optical)", WIN_UNDST, START_K2_SCALE,  200, lambda x: None)

    map1, map2, K_mod, D_mod = build_maps(K, D, w, h,
                                           balance=START_BALANCE / 100.0,
                                           zoom=max(START_ZOOM / 100.0, 0.01),
                                           cx_offset=START_CX_OFFSET - 200,
                                           cy_offset=START_CY_OFFSET - 200,
                                           fx_scale=START_FX_SCALE / 100.0,
                                           fy_scale=START_FY_SCALE / 100.0,
                                           k1_scale=START_K1_SCALE / 100.0,
                                           k2_scale=START_K2_SCALE / 100.0)
    undistor.update_maps(map1, map2)

    prev_params = None
    frame_count = 0
    last_frame  = None
    last_undist = None

    print("OPTICAL sliders (cx/cy/fx/fy/k1/k2) -> saved to cam_K/D.npy on S")
    print("DISPLAY sliders (balance/zoom)       -> saved to display_params.txt only")
    print("S=save  Q=quit\n")

    while True:
        balance   = cv2.getTrackbarPos("Balance  (display)", WIN_UNDST) / 100.0
        zoom      = max(cv2.getTrackbarPos("Zoom     (display)", WIN_UNDST) / 100.0, 0.01)
        cx_offset = cv2.getTrackbarPos("Cx off   (optical)", WIN_UNDST) - 200
        cy_offset = cv2.getTrackbarPos("Cy off   (optical)", WIN_UNDST) - 200
        fx_scale  = max(cv2.getTrackbarPos("fx scale (optical)", WIN_UNDST) / 100.0, 0.1)
        fy_scale  = max(cv2.getTrackbarPos("fy scale (optical)", WIN_UNDST) / 100.0, 0.1)
        k1_scale  = cv2.getTrackbarPos("k1 scale (optical)", WIN_UNDST) / 100.0
        k2_scale  = cv2.getTrackbarPos("k2 scale (optical)", WIN_UNDST) / 100.0

        params = (balance, zoom, cx_offset, cy_offset,
                  fx_scale, fy_scale, k1_scale, k2_scale)

        if params != prev_params:
            map1, map2, K_mod, D_mod = build_maps(
                K, D, w, h, balance, zoom,
                cx_offset, cy_offset,
                fx_scale, fy_scale,
                k1_scale, k2_scale
            )
            undistor.update_maps(map1, map2)
            prev_params = params
            print(f"\rfx={K_mod[0,0]:.1f} fy={K_mod[1,1]:.1f} "
                  f"cx={K_mod[0,2]:.1f} cy={K_mod[1,2]:.1f} "
                  f"k1={D_mod[0,0]:.5f} k2={D_mod[1,0]:.5f} "
                  f"bal={balance:.2f} zoom={zoom:.2f}  ",
                  end="", flush=True)

        frame = cam.read()
        if frame is not None:
            last_frame = frame
            undistor.push_frame(frame)

        undist = undistor.get_result()
        if undist is not None:
            last_undist = undist

        if last_frame is not None:
            cv2.imshow(WIN_ORIG, last_frame)

        if last_undist is not None:
            overlay = last_undist.copy()
            cv2.putText(overlay,
                        f"fx={K_mod[0,0]:.1f} fy={K_mod[1,1]:.1f} "
                        f"cx={K_mod[0,2]:.1f} cy={K_mod[1,2]:.1f} "
                        f"k1={D_mod[0,0]:.4f} k2={D_mod[1,0]:.4f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(overlay,
                        f"balance={balance:.2f}(disp) zoom={zoom:.2f}(disp) "
                        f"{'[CUDA]' if CUDA_AVAILABLE else '[CPU]'}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.imshow(WIN_UNDST, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_frame is not None and last_undist is not None:
            np.save(K_PATH, K_mod)
            np.save(D_PATH, D_mod)
            np.savetxt(K_PATH.replace(".npy", ".txt"), K_mod, fmt="%.8f")
            np.savetxt(D_PATH.replace(".npy", ".txt"), D_mod.ravel(), fmt="%.8f",
                       header="k1 k2 k3 k4")
            np.savetxt("display_params.txt", [balance, zoom], fmt="%.4f",
                       header="balance zoom (display only, not optical)")

            orig_path  = f"saved_original_{frame_count:04d}.png"
            undst_path = f"saved_undistorted_{frame_count:04d}.png"
            cv2.imwrite(orig_path,  last_frame)
            cv2.imwrite(undst_path, last_undist)

            print(f"\n\nSaved K  -> {K_PATH} + {K_PATH.replace('.npy', '.txt')}")
            print(f"Saved D  -> {D_PATH} + {D_PATH.replace('.npy', '.txt')}")
            print(f"Saved display -> display_params.txt")
            print(f"Saved frames  -> {orig_path}, {undst_path}")
            print(f"\nK:\n{K_mod}")
            print(f"D: {D_mod.ravel()}")
            print(f"balance={balance}  zoom={zoom}  (display only)\n")
            frame_count += 1

    cam.stop()
    undistor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()