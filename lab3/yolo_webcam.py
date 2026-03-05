from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")  # small + fast pretrained model

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera did not open.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        results = model(frame, verbose=False)  # inference
        annotated = results[0].plot()          # draw boxes

        cv2.imshow("YOLO Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
