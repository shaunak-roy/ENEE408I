#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty


def get_key():
    """Read one key without blocking."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def move():
    rospy.init_node('turtlebot3_keyboard_teleop', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    vel_msg = Twist()

    # Save terminal settings so we can restore them later
    settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    # Step sizes
    linear_step = 0.05
    angular_step = 0.20

    # Limits
    max_linear = 0.4
    min_linear = -0.4
    max_angular = 1.5
    min_angular = -1.5

    print("Keyboard Teleop Started")
    print("-----------------------")
    print("W: increase forward speed")
    print("S: increase backward speed / reduce forward")
    print("A: turn left")
    print("D: turn right")
    print("SPACE or X: complete stop")
    print("Q: quit")

    try:
        while not rospy.is_shutdown():
            key = get_key()

            if key is not None:
                key = key.lower()

                if key == 'w':
                    vel_msg.linear.x += linear_step
                elif key == 's':
                    vel_msg.linear.x -= linear_step
                elif key == 'a':
                    vel_msg.angular.z += angular_step
                elif key == 'd':
                    vel_msg.angular.z -= angular_step
                elif key == ' ' or key == 'x':
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.0
                elif key == 'q':
                    break

                # Clamp values so speeds do not grow forever
                vel_msg.linear.x = max(min(vel_msg.linear.x, max_linear), min_linear)
                vel_msg.angular.z = max(min(vel_msg.angular.z, max_angular), min_angular)

                print(
                    f"\rlinear.x = {vel_msg.linear.x:.2f} m/s | angular.z = {vel_msg.angular.z:.2f} rad/s   ",
                    end=''
                )

            pub.publish(vel_msg)
            rate.sleep()

    finally:
        # Stop robot before exiting
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        pub.publish(vel_msg)

        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("\nTeleop ended. Robot stopped.")


if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass