import os
import cv2
import re

CAMERA_INDEX = 4
SAVE_DIR = "./calibration_data_raw"

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
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cam.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}")
        exit(1)

    frame_count = get_starting_index(SAVE_DIR)
    print(f"Starting from frame {frame_count}. p=save q=quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        cv2.imshow("Cam1", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            filename = os.path.join(SAVE_DIR, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            frame_count += 1
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()