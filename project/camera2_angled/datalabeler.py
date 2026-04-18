# datalabeler.py — Camera 1 (Top-down) labeling tool
# Click twice to draw a bounding box.
# 'n' = next/save | 'u' = undo | 's' = skip | 'q' = quit

import cv2
import os
import re

# ===== CONFIG =====
IMAGE_DIR = "./data_raw"
LABEL_DIR = "./data_labeled"
CLASS_ID = 0   # 0 = package (update data.yaml if you add more classes)

os.makedirs(LABEL_DIR, exist_ok=True)

drawing = False
ix, iy = -1, -1
boxes = []


def extract_index(filename):
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else -1


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            ix, iy = x, y
            drawing = True
        else:
            boxes.append((ix, iy, x, y))
            drawing = False


def draw_boxes(img):
    temp = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return temp


def save_labels(img_shape, img_name):
    h, w, _ = img_shape
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    with open(label_path, "w") as f:
        for (x1, y1, x2, y2) in boxes:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"Saved: {label_path} ({len(boxes)} boxes)")


images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")],
    key=extract_index
)

if not images:
    print(f"No .jpg images found in {IMAGE_DIR}")
    print("Run datacollector.py first to capture images.")
    exit(1)

print(f"Found {len(images)} images from Camera 1 (Top-Down)")

for idx, img_name in enumerate(images):
    img_path = os.path.join(IMAGE_DIR, img_name)
    current_img = cv2.imread(img_path)

    if current_img is None:
        print(f"Skipping unreadable: {img_name}")
        continue

    # skip if already labeled
    label_name = os.path.splitext(img_name)[0] + ".txt"
    if os.path.exists(os.path.join(LABEL_DIR, label_name)):
        print(f"Already labeled, skipping: {img_name}")
        continue

    boxes = []
    cv2.namedWindow("Camera 1 - Label")
    cv2.setMouseCallback("Camera 1 - Label", mouse_callback)

    print(f"\n[{idx+1}/{len(images)}] {img_name}")
    print("click x2 = box | n = save/next | u = undo | s = skip | q = quit")

    while True:
        display = draw_boxes(current_img)
        cv2.imshow("Camera 1 - Label", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            save_labels(current_img.shape, img_name)
            break
        elif key == ord('s'):
            print(f"Skipped: {img_name}")
            break
        elif key == ord('u'):
            if boxes:
                boxes.pop()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("Camera 1 labeling complete!")
