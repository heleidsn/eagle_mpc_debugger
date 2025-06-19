#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import RCIn
from sensor_msgs.msg import JointState

class RCToJointStateMapper:
    def __init__(self):
        rospy.init_node('rc_to_joint_state_node', log_level=rospy.INFO)
        rospy.loginfo("Starting RC to JointState Mapper...")

        # 关节数量
        self.num_joints = rospy.get_param('~num_joints', 3)
        self.using_soft_ee = rospy.get_param('~using_soft_ee', True)

        # 关节角度范围（弧度），按顺序给出
        self.min_angles = [-1.2, -0.8, 0.0]   # rad
        self.max_angles = [1.2, 0.8, 0.6]   # rad
        
        self.action_velocity = 1.5 # 关节速度

        # RC输入通道索引
        self.rc_channels = [9, 10, 11]

        if self.using_soft_ee:
            self.joint_names = ['joint_1', 'joint_2', 'joint_4']  # 关节名称
        else:
            self.joint_names = ['joint_1', 'joint_2', 'joint_3']  # 关节名称

        # 发布器
        self.joint_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)

        # 订阅RC输入
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback)

        rospy.loginfo("RC to JointState node initialized.")
        rospy.spin()

    def rc_callback(self, msg):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.joint_names
        joint_state.position = []
        
        for i, ch in enumerate(self.rc_channels):
            if ch >= len(msg.channels):
                rospy.logwarn(f"RC channel index {ch} out of range.")
                joint_state.position.append(0.0)
                continue

            pwm = msg.channels[ch-1]
            # rospy.loginfo("RC channel {} PWM: {}".format(ch, pwm))
            pwm = max(1000, min(2000, pwm))  # 限制范围
            norm = (pwm - 1000) / 1000.0  # 归一化到 [0, 1]

            angle = self.min_angles[i] + norm * (self.max_angles[i] - self.min_angles[i])
            joint_state.position.append(angle)
            joint_state.velocity.append(self.action_velocity)  # 速度可以设置为0或其他值
            
        rospy.loginfo("Publishing joint state: {}".format(joint_state.position))

        self.joint_pub.publish(joint_state)

if __name__ == '__main__':
    try:
        RCToJointStateMapper()
    except rospy.ROSInterruptException:
        pass
