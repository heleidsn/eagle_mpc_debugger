#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64

class JointInitializer:
    def __init__(self):
        # Define joint names and their initial positions
        self.joint_initial_positions = {
            'joint1': 0.0,  # Initial position for joint 1
            'joint2': 1.57,  # Initial position for joint 2 (in radians)
            'joint3': -1.57,  # Initial position for joint 3 (in radians)
            # Add more joints as needed
        }

        # Initialize ROS publishers for each joint
        self.publishers = {}
        for joint_name in self.joint_initial_positions.keys():
            # Create a publisher for each joint to set the position
            self.publishers[joint_name] = rospy.Publisher(f'/{joint_name}/position_controller/command', Float64, queue_size=10)

    def set_initial_positions(self):
        # Set initial positions for each joint
        for joint_name, initial_position in self.joint_initial_positions.items():
            rospy.loginfo(f"Setting {joint_name} to initial position: {initial_position}")
            self.publishers[joint_name].publish(initial_position)

if __name__ == "__main__":
    rospy.init_node('joint_initializer_node')

    # Create an instance of the JointInitializer
    initializer = JointInitializer()

    # Set initial positions
    initializer.set_initial_positions()

    rospy.loginfo("Joint initialization complete.")