# pi_camera_publisher.py — Runs on the Raspberry Pi
#
# Captures frames from a USB camera connected to the Pi, JPEG-compresses them,
# and publishes over ROS2 so the computer can subscribe and run YOLO inference.
#
# Usage (on Pi):
#   python3 pi_camera_publisher.py
#
# Or with custom parameters:
#   python3 pi_camera_publisher.py --ros-args -p camera_index:=0 -p fps:=15.0 \
#       -p topic_name:=/camera1/image/compressed

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class PiCameraPublisher(Node):
    def __init__(self):
        super().__init__('pi_camera_publisher')

        # ===== PARAMETERS (overridable via --ros-args) =====
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('topic_name', '/camera/image/compressed')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('jpeg_quality', 75)   # 0-100, higher = better quality, more bandwidth
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        camera_index = self.get_parameter('camera_index').value
        topic_name   = self.get_parameter('topic_name').value
        fps          = self.get_parameter('fps').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        frame_width  = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value

        # ===== CAMERA SETUP =====
        self.cam = cv2.VideoCapture(camera_index)
        if not self.cam.isOpened():
            self.get_logger().error(f"Could not open camera at index {camera_index}")
            raise RuntimeError("Camera failed to open")

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        actual_w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera opened: {actual_w}x{actual_h}")
        self.get_logger().info(f"Publishing to: {topic_name} at {fps} FPS, JPEG quality {self.jpeg_quality}")

        # ===== PUBLISHER + TIMER =====
        self.publisher = self.create_publisher(CompressedImage, topic_name, 10)
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

        self.frame_count = 0

    def publish_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return

        # JPEG-encode the frame to minimize network bandwidth
        result, encoded = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not result:
            self.get_logger().warn("JPEG encoding failed")
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        self.publisher.publish(msg)
        self.frame_count += 1

        if self.frame_count % 30 == 0:
            self.get_logger().info(f"Published {self.frame_count} frames")

    def destroy_node(self):
        self.cam.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
