# computer_live_viewer.py — Simple live feed viewer (no YOLO)
#
# Subscribes to the Pi's camera stream and just displays it. Use this to verify
# the Pi is publishing correctly BEFORE trying YOLO inference.
#
# Usage:
#   python3 computer_live_viewer.py
#   python3 computer_live_viewer.py --ros-args -p input_topic:=/camera1/image/compressed

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class LiveViewer(Node):
    def __init__(self):
        super().__init__('live_viewer')

        self.declare_parameter('input_topic', '/camera/image/compressed')
        input_topic = self.get_parameter('input_topic').value

        if input_topic is None:
            return  # or raise, or log

        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to: {input_topic}")
        self.get_logger().info("Press 'q' in the window to quit.")

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        cv2.imshow('Pi Camera Live Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LiveViewer()
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
