#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped

class GroundTruthPublisher:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("groundtruth_to_mavros")

        # 从参数服务器获取模型名称，默认为 "s500"
        self.model_name = rospy.get_param("~model_name", "s500")

        # 创建发布器
        self.pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=10)

        # 等待模型加载完成
        rospy.loginfo(f"Waiting for {self.model_name} model to be loaded in Gazebo...")
        self.iris_idx = self.wait_for_model()

        # 创建订阅器
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.groundtruth_callback)
        
        # 创建定时器，100Hz
        self.rate = rospy.Rate(100)
        
        # 存储最新的位姿数据
        self.latest_pose = None
        
        rospy.loginfo("GroundTruthPublisher initialized successfully.")
        
        # 开始发布循环
        self.publish_loop()

    def wait_for_model(self):
        """等待模型加载完成"""
        while not rospy.is_shutdown():
            try:
                # 等待 /gazebo/model_states 消息
                msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                # 尝试获取模型的索引
                return msg.name.index(self.model_name)
            except rospy.ROSException:
                rospy.logwarn("Timeout while waiting for /gazebo/model_states message...")
            except ValueError:
                rospy.logwarn(f"{self.model_name} model not found in /gazebo/model_states. Retrying...")

    def groundtruth_callback(self, msg):
        """存储最新的位姿数据"""
        self.latest_pose = msg.pose[self.iris_idx]

    def publish_loop(self):
        """以固定频率发布位姿数据"""
        while not rospy.is_shutdown():
            if self.latest_pose is not None:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"
                pose_msg.pose = self.latest_pose
                self.pub.publish(pose_msg)
            self.rate.sleep()

if __name__ == "__main__":
    GroundTruthPublisher()
    rospy.spin()