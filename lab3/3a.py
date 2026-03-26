import cv2
import numpy as np

def detectLine(frame):
    """
    Detect a white line in the image and return:
      - lineCenter in [-1, 1] (left=-1, center=0, right=+1)
      - newFrame (frame with rectangle + center dot drawn)
    """

    newFrame = frame.copy()
    _, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 40, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lineCenter = 0.0

    if contours:
        c = max(contours, key=cv2.contourArea)

        x, y, cw, ch = cv2.boundingRect(c)

        cx = x + cw // 2
        cy = y + ch // 2

        lineCenter = (cx - (w / 2)) / (w / 2)

        cv2.rectangle(newFrame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
        cv2.circle(newFrame, (cx, cy), 6, (0, 0, 255), -1)

    return float(lineCenter), newFrame


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera did not open.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        lineCenter, newFrame = detectLine(frame)

        cv2.imshow("Line Tracking", newFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Line Tracking", cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
