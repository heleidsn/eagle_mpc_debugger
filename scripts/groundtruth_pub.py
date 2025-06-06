#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped

class GroundTruthPublisher:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("groundtruth_to_mavros")

        # 创建发布器
        self.pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=10)

        # 等待 s500 模型加载完成
        rospy.loginfo("Waiting for s500 model to be loaded in Gazebo...")
        self.iris_idx = self.wait_for_s500()

        # 创建订阅器
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.groundtruth_to_mavros)
        rospy.loginfo("GroundTruthPublisher initialized successfully.")

    def wait_for_s500(self):
        """等待 s500 模型加载完成"""
        while not rospy.is_shutdown():
            try:
                # 等待 /gazebo/model_states 消息
                msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                # 尝试获取 s500 的索引
                return msg.name.index("s500")
            except rospy.ROSException:
                rospy.logwarn("Timeout while waiting for /gazebo/model_states message...")
            except ValueError:
                rospy.logwarn("s500 model not found in /gazebo/model_states. Retrying...")

    def groundtruth_to_mavros(self, msg):
        """将 Gazebo 的 ground truth 转换为 MAVROS 的 PoseStamped 消息"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose = msg.pose[self.iris_idx]  # 使用 Gazebo 的 ground truth

        self.pub.publish(pose_msg)

if __name__ == "__main__":
    GroundTruthPublisher()
    rospy.spin()