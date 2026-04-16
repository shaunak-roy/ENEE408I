# datacollector.py — Camera 2 (45-degree Angled View)
# Capture training frames from the side-mounted angled camera.
# Press 'p' to save a frame, 'q' to quit.

import os
import cv2
import re

# ===== CONFIG =====
CAMERA_INDEX = 1         # Angled camera — usually the second camera (change to 0 / 2 if needed)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SAVE_DIR = "./data_raw"
WINDOW_NAME = "Camera 2 (Angled) - Data Collector"


def get_starting_index(folder):
    pattern = re.compile(r"frame_(\d+)\.jpg")
    max_index = -1
    if not os.path.exists(folder):
        return 0
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    return max_index + 1


if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print(f"ERROR: Could not open camera at index {CAMERA_INDEX}")
        print("Try changing CAMERA_INDEX — if only one camera is plugged in, this should be 0.")
        exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera 2 (Angled) opened: {actual_w}x{actual_h}")

    frame_count = get_starting_index(SAVE_DIR)
    print(f"Starting from frame index: {frame_count}")
    print("Controls: 'p' = save frame | 'q' = quit")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame. Retrying...")
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('p') or key == ord('P'):
                filename = os.path.join(SAVE_DIR, f"frame_{frame_count}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                frame_count += 1

            elif key == ord('q'):
                print(f"\nDone. Saved {frame_count} total frames.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")

    finally:
        cam.release()
        cv2.destroyAllWindows()
