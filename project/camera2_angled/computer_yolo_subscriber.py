# ros2_yolo_viewer.py — Subscribe to Pi camera via ROS2 and run YOLO live inference
#
# Usage:
#   python3 ros2_yolo_viewer.py
#   python3 ros2_yolo_viewer.py --ros-args -p input_topic:=/camera2/image/compressed

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
import cv2
import numpy as np


class YOLOViewer(Node):
    def __init__(self):
        super().__init__('yolo_viewer')

        # ROS2 parameter for the topic
        self.declare_parameter('input_topic', '/camera/image/compressed')
        input_topic = self.get_parameter('input_topic').value
        if input_topic is None:
            return

        # Load trained YOLO model
        self.model = YOLO("runs/detect/camera2_angled/weights/best.pt")

        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(f"Subscribed to: {input_topic}")
        self.get_logger().info("Running YOLO inference (showing 'stacked' only). Press 'q' to quit.")

    def image_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Run YOLO inference
        results = self.model(frame, device=0, verbose=False)

        # Draw only 'stacked' detections manually
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label != 'stacked':
                continue

            # Extract box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"stacked {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 0, 255), -1)
            cv2.putText(frame, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Camera 2 - YOLO Live', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()