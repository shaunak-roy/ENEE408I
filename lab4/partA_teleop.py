#!/usr/bin/env python3
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Teleop(Node):
    """
    Part A: Keyboard teleop for TurtleBot3 (ROS2).
    Publishes geometry_msgs/Twist to /cmd_vel.

    Controls:
      w = forward
      s = backward
      a = turn left
      d = turn right
      space = stop
      q = quit
    """

    def __init__(self):
        super().__init__("partA_teleop")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # current commanded speeds
        self.lin = 0.0
        self.ang = 0.0

        # step sizes when you press keys
        self.lin_step = 0.05
        self.ang_step = 0.2

        # publish at 10 Hz
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("Teleop ready: w/s/a/d, space=stop, q=quit")

    def loop(self):
        key = get_key_nonblocking()

        # update speeds from keypress
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
            # stop before exit
            self.pub.publish(Twist())
            rclpy.shutdown()
            return

        # build and publish Twist message
        msg = Twist()
        msg.linear.x = float(self.lin)
        msg.angular.z = float(self.ang)
        self.pub.publish(msg)


def get_key_nonblocking():
    """
    Reads one keypress without blocking.
    Returns '' if no key was pressed.
    """
    dr, _, _ = select.select([sys.stdin], [], [], 0.0)
    if dr:
        return sys.stdin.read(1)
    return ""


def main(args=None):
    # Save terminal settings and enter raw mode so we can read single keys
    settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    rclpy.init(args=args)
    node = Teleop()

    try:
        rclpy.spin(node)
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()