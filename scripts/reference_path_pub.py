#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class ReferencePathPublisher:
    def __init__(self):
        self.reference_pose_topic = rospy.get_param("~reference_pose_topic", "/reference/pose")
        self.reference_path_topic = rospy.get_param("~reference_path_topic", "/reference/path")
        self.path_frame_id = rospy.get_param("~path_frame_id", "map")
        self.max_path_length = rospy.get_param("~max_path_length", 2000)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.path_frame_id

        self.path_pub = rospy.Publisher(self.reference_path_topic, Path, queue_size=10)
        self.pose_sub = rospy.Subscriber(self.reference_pose_topic, PoseStamped, self.pose_callback, queue_size=50)

        rospy.loginfo("reference_path_pub started: %s -> %s", self.reference_pose_topic, self.reference_path_topic)

    def pose_callback(self, pose_msg):
        stamped_pose = PoseStamped()
        stamped_pose.header = pose_msg.header
        stamped_pose.pose = pose_msg.pose

        if not stamped_pose.header.frame_id:
            stamped_pose.header.frame_id = self.path_frame_id

        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.header.frame_id = stamped_pose.header.frame_id
        self.path_msg.poses.append(stamped_pose)

        if len(self.path_msg.poses) > self.max_path_length:
            self.path_msg.poses = self.path_msg.poses[-self.max_path_length :]

        self.path_pub.publish(self.path_msg)


if __name__ == "__main__":
    rospy.init_node("reference_path_pub")
    ReferencePathPublisher()
    rospy.spin()
