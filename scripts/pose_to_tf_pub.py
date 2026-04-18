#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import tf2_ros
from geometry_msgs.msg import TransformStamped


class PoseToTfPublisher:
    def __init__(self):
        self.pose_topic = rospy.get_param("~pose_topic", "/mavros/vision_pose/pose")
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "base_link")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=50)

        rospy.loginfo("pose_to_tf_pub started: %s -> %s/%s", self.pose_topic, self.parent_frame, self.child_frame)

    def pose_callback(self, msg):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        tf_msg.header.frame_id = self.parent_frame
        tf_msg.child_frame_id = self.child_frame

        tf_msg.transform.translation.x = msg.pose.position.x
        tf_msg.transform.translation.y = msg.pose.position.y
        tf_msg.transform.translation.z = msg.pose.position.z

        tf_msg.transform.rotation.x = msg.pose.orientation.x
        tf_msg.transform.rotation.y = msg.pose.orientation.y
        tf_msg.transform.rotation.z = msg.pose.orientation.z
        tf_msg.transform.rotation.w = msg.pose.orientation.w

        self.tf_broadcaster.sendTransform(tf_msg)


if __name__ == "__main__":
    rospy.init_node("pose_to_tf_pub")
    PoseToTfPublisher()
    rospy.spin()
