#!/usr/bin/env python3

import sys
import select
import termios
import tty
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class TurtleBot3SafetyTeleop(Node):
    def __init__(self):
        super().__init__('turtlebot3_safety_teleop')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.vel_msg = Twist()

        self.linear_step = 0.05
        self.angular_step = 0.20

        self.max_linear = 0.4
        self.min_linear = -0.4
        self.max_angular = 1.5
        self.min_angular = -1.5

        self.safety_threshold = 0.10
        self.front_distance = float('inf')
        self.obstacle_detected = False

        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.get_logger().info('Keyboard Teleop with LiDAR Safety Stop Started')
        print("-----------------------")
        print("W: increase forward speed")
        print("S: increase backward speed / reduce forward")
        print("A: turn left")
        print("D: turn right")
        print("SPACE or X: complete stop")
        print("Q: quit")
        print(f"Safety stop threshold = {self.safety_threshold:.2f} m")

    def get_key(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

    def scan_callback(self, msg: LaserScan):
        """
        Check the front LiDAR region and find the closest valid obstacle distance.
        We inspect a small window around the front direction.
        """
        ranges = msg.ranges
        n = len(ranges)

        if n == 0:
            self.front_distance = float('inf')
            self.obstacle_detected = False
            return

        
        center = n // 2
        window = 20  
        front_ranges = ranges[max(0, center - window): min(n, center + window + 1)]

        valid_ranges = [
            r for r in front_ranges
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]

        if valid_ranges:
            self.front_distance = min(valid_ranges)
            self.obstacle_detected = self.front_distance < self.safety_threshold
        else:
            self.front_distance = float('inf')
            self.obstacle_detected = False

    def control_loop(self):
        key = self.get_key()

        if key is not None:
            key = key.lower()

            if key == 'w':
                self.vel_msg.linear.x += self.linear_step
            elif key == 's':
                self.vel_msg.linear.x -= self.linear_step
            elif key == 'a':
                self.vel_msg.angular.z += self.angular_step
            elif key == 'd':
                self.vel_msg.angular.z -= self.angular_step
            elif key == ' ' or key == 'x':
                self.vel_msg.linear.x = 0.0
                self.vel_msg.angular.z = 0.0
            elif key == 'q':
                self.stop_robot()
                self.cleanup_terminal()
                rclpy.shutdown()
                return

        # Clamp values
        self.vel_msg.linear.x = max(min(self.vel_msg.linear.x, self.max_linear), self.min_linear)
        self.vel_msg.angular.z = max(min(self.vel_msg.angular.z, self.max_angular), self.min_angular)

        # Safety override:
        # Lab says override forward/backward commands by setting linear velocity to 0.0
        # when obstacle is within threshold.
        if self.obstacle_detected:
            self.vel_msg.linear.x = 0.0

        self.cmd_pub.publish(self.vel_msg)

        status = (
            f"\rlinear.x = {self.vel_msg.linear.x:.2f} m/s | "
            f"angular.z = {self.vel_msg.angular.z:.2f} rad/s | "
            f"front = {self.front_distance:.2f} m | "
            f"safety = {'ON' if self.obstacle_detected else 'OFF'}   "
        )
        print(status, end='')

    def stop_robot(self):
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        self.cmd_pub.publish(self.vel_msg)
        print("\nRobot stopped.")

    def cleanup_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot3SafetyTeleop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.cleanup_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()