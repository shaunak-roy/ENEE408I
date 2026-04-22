import cv2
import numpy as np

# ── Paste your values here ──────────────────
IMAGE = "package_data_raw/frame_1.jpg"   # any image from your camera
K_PATH = "cam_K.npy"
D_PATH = "cam_D.npy"
K = np.load(K_PATH)
D = np.load(D_PATH)
BALANCE = 0.0
ZOOM    = 1.0
# ────────────────────────────────────────────

img = cv2.imread(IMAGE)
if img is None:
    print(f"ERROR: Could not read '{IMAGE}'")
    exit(1)
h, w = img.shape[:2]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=BALANCE
)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
)
result = cv2.remap(img, map1, map2,
                   interpolation=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", result)
cv2.imwrite("test_undistorted.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()