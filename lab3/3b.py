from ultralytics import YOLO
import cv2
import yaml

def main():

    modes = {0: 'train', 1: 'test', 2: 'video'}
    mode = 2

    match modes[mode]:
        case 'train':
            model = YOLO("yolov8n.pt")
            model.train(data="data.yaml", epochs=300, imgsz=640, device=0, plots=True)
            return
        
        case 'test':
            model = YOLO("runs/detect/train/weights/best.pt")
            metrics = model.val(split='test') 
            print(metrics.box.map)
            return
        
        case 'video':
            model = YOLO("runs/detect/train/weights/best.pt")

            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                print("Camera did not open.")
                return

            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                
                results = model(frame, device=0, verbose=False)
                annotated = results[0].plot()

                cv2.imshow("YOLO Webcam", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if cv2.getWindowProperty("YOLO Webcam", cv2.WND_PROP_VISIBLE) < 1:
                    break

            cam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
