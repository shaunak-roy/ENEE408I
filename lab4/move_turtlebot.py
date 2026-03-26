import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Turtlebot3Move(Node):
    def __init__(self):
        super().__init__('turtlebot3_autonomous_move')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

    def timer_callback(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.2
        vel_msg.angular.z = 0.0

        self.publisher_.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)

    node = Turtlebot3Move()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()