# datatrainer.py — Camera 1 (Top-down) YOLO training / testing / live inference
# Set mode: 0 = train, 1 = test, 2 = live video

from ultralytics import YOLO
import cv2

CAMERA_INDEX = 0   # must match datacollector.py

def main():
    mode = 0  # 0 = train, 1 = test, 2 = video

    if mode == 0:
        model = YOLO("yolov8n.pt")
        model.train(
            data="data.yaml",
            epochs=300,
            imgsz=640,
            device=0,     # set to 'cpu' if no CUDA GPU
            plots=True,
            workers=0,
            batch=8,
            name="camera1_topdown"
        )

    elif mode == 1:
        model = YOLO("runs/detect/camera1_topdown/weights/best.pt")
        metrics = model.val(split='test')
        print(f"Camera 1 mAP50:    {metrics.box.map50:.4f}")
        print(f"Camera 1 mAP50-95: {metrics.box.map:.4f}")

    elif mode == 2:
        model = YOLO("runs/detect/camera1_topdown/weights/best.pt")
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened():
            print(f"Camera at index {CAMERA_INDEX} did not open.")
            return
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Camera 1 live inference. Press 'q' to quit.")
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            results = model(frame, device=0, verbose=False)
            annotated = results[0].plot()
            cv2.imshow("Camera 1 - YOLO Live", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty("Camera 1 - YOLO Live", cv2.WND_PROP_VISIBLE) < 1:
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
