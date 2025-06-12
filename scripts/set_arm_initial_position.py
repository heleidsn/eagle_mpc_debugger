#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
import time

def set_initial_positions():
    # 初始化节点
    rospy.init_node('set_initial_positions')
    
    # 创建发布器
    publishers = {
        'joint_1': rospy.Publisher('/arm_controller/joint_1_position_controller/command', Float64, queue_size=1),
        'joint_2': rospy.Publisher('/arm_controller/joint_2_position_controller/command', Float64, queue_size=1),
        'joint_3': rospy.Publisher('/arm_controller/joint_3_position_controller/command', Float64, queue_size=1),
        'joint_4': rospy.Publisher('/arm_controller/joint_4_position_controller/command', Float64, queue_size=1)
    }
    
    # 等待发布器初始化
    time.sleep(1)
    
    # 设置初始位置
    initial_positions = {
        'joint_1': -1.2,  # radians
        'joint_2': -1.0,  # radians
        'joint_3': 0.0,   # radians
        'joint_4': 0.0    # radians
    }
    
    # 发布初始位置命令
    for joint, position in initial_positions.items():
        publishers[joint].publish(Float64(position))
        rospy.loginfo(f"Setting {joint} to {position} radians")
        time.sleep(0.5)  # 给每个关节一些时间到达位置

if __name__ == '__main__':
    try:
        set_initial_positions()
    except rospy.ROSInterruptException:
        pass