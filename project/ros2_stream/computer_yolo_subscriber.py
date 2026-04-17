# computer_yolo_subscriber.py — Runs on your computer (needs GPU / CUDA / PyTorch)
#
# Subscribes to compressed camera frames from the Pi, runs YOLO inference,
# and displays the annotated result. Optionally publishes the annotated frames
# back so the Pi (or other nodes) can react to detections.
#
# Usage (on computer):
#   python3 computer_yolo_subscriber.py
#
# Or with custom parameters:
#   python3 computer_yolo_subscriber.py --ros-args \
#       -p model_path:=../camera1_topdown/runs/detect/camera1_topdown/weights/best.pt \
#       -p input_topic:=/camera1/image/compressed \
#       -p output_topic:=/camera1/image/annotated

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')

        # ===== PARAMETERS =====
        self.declare_parameter('model_path', '../camera1_topdown/runs/detect/camera1_topdown/weights/best.pt')
        self.declare_parameter('input_topic', '/camera/image/compressed')
        self.declare_parameter('output_topic', '/yolo/annotated/compressed')
        self.declare_parameter('device', 0)           # 0 = GPU, 'cpu' = CPU
        self.declare_parameter('conf_threshold', 0.25) # minimum confidence to display
        self.declare_parameter('show_window', True)   # display the live annotated feed
        self.declare_parameter('publish_annotated', True)

        model_path       = self.get_parameter('model_path').value
        input_topic      = self.get_parameter('input_topic').value
        output_topic     = self.get_parameter('output_topic').value
        self.device      = self.get_parameter('device').value
        self.conf        = self.get_parameter('conf_threshold').value
        self.show_window = self.get_parameter('show_window').value
        self.publish_annotated = self.get_parameter('publish_annotated').value

        # ===== LOAD MODEL =====
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.get_logger().info(f"Model loaded. Device: {self.device}")

        # ===== SUBSCRIBER =====
        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to: {input_topic}")

        # ===== OPTIONAL PUBLISHER FOR ANNOTATED FRAMES =====
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(CompressedImage, output_topic, 10)
            self.get_logger().info(f"Publishing annotated frames to: {output_topic}")

        self.frame_count = 0

    def image_callback(self, msg):
        # Decode JPEG -> OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("Failed to decode incoming JPEG")
            return

        # Run YOLO inference
        results = self.model(frame, device=self.device, conf=self.conf, verbose=False)

        # Get annotated image (bounding boxes drawn by Ultralytics)
        annotated = results[0].plot()

        # Log detections occasionally
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            self.get_logger().info(f"Frame {self.frame_count}: {num_detections} detections")

        # Display locally
        if self.show_window:
            cv2.imshow('YOLO Inference (from Pi)', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()

        # Re-publish annotated frames so other nodes (or the Pi) can consume them
        if self.publish_annotated:
            result, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if result:
                out_msg = CompressedImage()
                out_msg.header = msg.header  # preserve timestamp
                out_msg.format = 'jpeg'
                out_msg.data = encoded.tobytes()
                self.annotated_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSubscriber()
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
