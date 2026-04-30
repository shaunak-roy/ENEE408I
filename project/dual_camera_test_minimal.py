import cv2
import numpy as np
import os

cam1 = cv2.VideoCapture(4, cv2.CAP_V4L2)
cam2 = cv2.VideoCapture(6, cv2.CAP_V4L2)

# Precompute undistortion maps
map1, map2 = None, None
K_PATH = "camera1_topdown/cam_K.npy"
D_PATH = "camera1_topdown/cam_D.npy"

if os.path.exists(K_PATH) and os.path.exists(D_PATH):
    K = np.load(K_PATH)
    D = np.zeros((4, 1))
    ret, frame = cam1.read()
    if ret:
        h, w = frame.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        print(f"Undistortion ready ({w}x{h})")

while True:
    ret1, f1 = cam1.read()
    ret2, f2 = cam2.read()
    if ret1:
        if map1 is not None:
            f1 = cv2.remap(f1, map1, map2, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("Cam1", f1)
    if ret2:
        cv2.imshow("Cam2", f2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()