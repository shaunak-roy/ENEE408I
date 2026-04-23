# datalabeler.py — Multi-class labeling tool
# Reads classes from data.yaml automatically.
# Click twice to draw a bounding box.
# Number keys = select class | n = next/save | u = undo | s = skip | q = quit

import cv2
import os
import re
import yaml

# ===== CONFIG =====
IMAGE_DIR    = "./package_data_raw"
LABEL_DIR    = "./package_data_labeled"
YAML_PATH    = "./data.yaml"
NUM_WORKERS  = 3                      # total number of people labeling
WORKER_ID    = 1                      # this person's ID (1 to NUM_WORKERS)
# ==================

# Load classes from data.yaml
with open(YAML_PATH, "r") as f:
    config = yaml.safe_load(f)
CLASS_NAMES = config["names"]
print(f"Loaded {len(CLASS_NAMES)} classes from {YAML_PATH}:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  [{i}] {name}")

os.makedirs(LABEL_DIR, exist_ok=True)

drawing = False
ix, iy = -1, -1
boxes = []         # list of (x1, y1, x2, y2, class_id)
current_class = 0


def extract_index(filename):
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else -1


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, boxes, current_class
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            ix, iy = x, y
            drawing = True
        else:
            boxes.append((ix, iy, x, y, current_class))
            drawing = False


def draw_boxes(img):
    temp = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]
    for (x1, y1, x2, y2, cid) in boxes:
        color = colors[cid % len(colors)]
        cv2.rectangle(temp, (x1, y1), (x2, y2), color, 2)
        cv2.putText(temp, CLASS_NAMES[cid], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return temp


def draw_hud(img):
    """Draw class selector and instructions at the top."""
    temp = img.copy()
    y = 25
    for i, name in enumerate(CLASS_NAMES):
        marker = ">>" if i == current_class else "  "
        color = (0, 255, 0) if i == current_class else (200, 200, 200)
        cv2.putText(temp, f"{marker} [{i}] {name}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
    cv2.putText(temp, "click x2=box | 0-9=class | n=save | u=undo | s=skip | q=quit",
                (10, temp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return temp


def save_labels(img_shape, img_name):
    h, w, _ = img_shape
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)
    with open(label_path, "w") as f:
        for (x1, y1, x2, y2, cid) in boxes:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"Saved: {label_path} ({len(boxes)} boxes)")


images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))],
    key=extract_index
)

if not images:
    print(f"No images found in {IMAGE_DIR}")
    exit(1)

# Split images by worker ID
if WORKER_ID < 1 or WORKER_ID > NUM_WORKERS:
    print(f"ERROR: WORKER_ID must be 1 to {NUM_WORKERS}")
    exit(1)

print(f"\n{len(images)} total images, ~{len(images)//NUM_WORKERS} per worker:")
for w in range(NUM_WORKERS):
    chunk = images[w::NUM_WORKERS]
    marker = " <<< you" if w + 1 == WORKER_ID else ""
    print(f"  Worker {w+1}: {len(chunk)} images{marker}")

my_images = images[WORKER_ID - 1::NUM_WORKERS]
print(f"\nWorker {WORKER_ID}/{NUM_WORKERS} — {len(my_images)} images to label")

for idx, img_name in enumerate(my_images):
    img_path = os.path.join(IMAGE_DIR, img_name)
    current_img = cv2.imread(img_path)
    if current_img is None:
        print(f"Skipping unreadable: {img_name}")
        continue

    label_name = os.path.splitext(img_name)[0] + ".txt"
    if os.path.exists(os.path.join(LABEL_DIR, label_name)):
        print(f"Already labeled, skipping: {img_name}")
        continue

    boxes = []
    cv2.namedWindow("Label")
    cv2.setMouseCallback("Label", mouse_callback)

    print(f"\n[{idx+1}/{len(my_images)}] {img_name}  |  class: [{current_class}] {CLASS_NAMES[current_class]}")

    while True:
        display = draw_boxes(current_img)
        display = draw_hud(display)
        cv2.imshow("Label", display)

        key = cv2.waitKey(1) & 0xFF

        # Number keys to switch class
        if ord('0') <= key <= ord('9'):
            num = key - ord('0')
            if num < len(CLASS_NAMES):
                current_class = num
                print(f"  Class → [{current_class}] {CLASS_NAMES[current_class]}")
        elif key == ord('n'):
            save_labels(current_img.shape, img_name)
            break
        elif key == ord('s'):
            print(f"Skipped: {img_name}")
            break
        elif key == ord('u'):
            if boxes:
                removed = boxes.pop()
                print(f"  Undo: {CLASS_NAMES[removed[4]]} box removed")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("\nLabeling complete!")