#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


IMAGE_TOPIC = "/camera/image_raw"  


class VisionFollower(Node):
    def __init__(self):
        super().__init__("vision_follower")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)
        self.bridge = CvBridge()

        # controller params
        self.kp = 0.8
        self.forward_speed = 0.05
        self.search_spin = 0.3

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Example: green object 
        lower = np.array([35, 80, 80], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cmd = Twist()

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 300:  # ignore tiny noise
                x, y, cw, ch = cv2.boundingRect(c)
                cx = x + cw // 2

                # error in [-1, 1]
                err = (cx - (w / 2)) / (w / 2)

                cmd.linear.x = float(self.forward_speed)
                cmd.angular.z = float(-self.kp * err)

                # visualize while debugging
                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                cv2.circle(frame, (cx, y + ch // 2), 6, (0, 0, 255), -1)
            else:
                # blob too small -> search
                cmd.angular.z = float(self.search_spin)
        else:
            # nothing found -> slow rotate to search
            cmd.angular.z = float(self.search_spin)

        self.pub.publish(cmd)

        # debug window
        cv2.imshow("vision", frame)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = VisionFollower()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()