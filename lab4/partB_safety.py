#!/usr/bin/env python3
import sys, select, termios, tty
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def get_key(timeout=0.1):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return None


class TeleopSafety(Node):
    def __init__(self):
        super().__init__("teleop_safety")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        self.lin = 0.0
        self.ang = 0.0
        self.lin_step = 0.05
        self.ang_step = 0.2

        self.safety_thresh = 0.30  # meters
        self.front_min = float("inf")  # updated by scan callback

        self.timer = self.create_timer(0.1, self.tick)

    def scan_cb(self, msg: LaserScan):
        # Look at a small window around "straight ahead" (angle ~ 0)
        # Convert angle to index: i = (angle - angle_min) / angle_increment
        if msg.angle_increment == 0.0:
            return

        center_i = int((0.0 - msg.angle_min) / msg.angle_increment)
        window = 15  # ~15 samples on each side (tune if needed)

        vals = []
        for i in range(center_i - window, center_i + window + 1):
            if 0 <= i < len(msg.ranges):
                r = msg.ranges[i]
                if not math.isinf(r) and not math.isnan(r):
                    vals.append(r)

        self.front_min = min(vals) if vals else float("inf")

    def tick(self):
        key = get_key()

        if key == "w":
            self.lin += self.lin_step
        elif key == "s":
            self.lin -= self.lin_step
        elif key == "a":
            self.ang += self.ang_step
        elif key == "d":
            self.ang -= self.ang_step
        elif key == " ":
            self.lin = 0.0
            self.ang = 0.0
        elif key == "q":
            rclpy.shutdown()
            return

        # SAFETY: block forward/back if too close
        if abs(self.lin) > 1e-6 and self.front_min < self.safety_thresh:
            self.lin = 0.0

        msg = Twist()
        msg.linear.x = float(self.lin)
        msg.angular.z = float(self.ang)
        self.pub.publish(msg)


def main():
    rclpy.init()
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        node = TeleopSafety()
        rclpy.spin(node)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()