# datacollector.py — Camera 1 (Top-down USBFHD01M)
# Capture training frames from the overhead camera.
# Press 'p' to save a frame, 'q' to quit.

import os
import cv2
import re

# ===== CONFIG =====
CAMERA_INDEX = 0         # Top-down USBFHD01M — change if needed (try 0, 1, 2)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SAVE_DIR = "./package_data_raw"
WINDOW_NAME = "Camera 1 (Top-Down) - Data Collector"


def get_starting_index(folder):
    """Find next available frame number so we don't overwrite."""
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
        print("Try changing CAMERA_INDEX to 1 or 2 if you have multiple cameras connected.")
        exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera 1 (Top-Down) opened: {actual_w}x{actual_h}")

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
