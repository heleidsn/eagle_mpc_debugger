#!/usr/bin/env python3

import rospy
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client
from std_msgs.msg import Float64
from eagle_mpc_debugger.cfg.JointPIDConfig import JointPIDConfig

class PIDReconfigure:
    def __init__(self):
        # 初始化节点
        rospy.init_node('pid_reconfigure_node')
        
        # 创建发布器
        self.publishers = {}
        self.joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4']
        for joint in self.joints:
            self.publishers[joint] = rospy.Publisher(
                f'/{joint}_position_controller/pid_gains',
                Float64,
                queue_size=1
            )
        
        # 创建动态重配置服务器
        self.srv = Server(JointPIDConfig, self.reconfigure_callback)
        
        # 创建动态重配置客户端
        self.clients = {}
        for joint in self.joints:
            self.clients[joint] = Client(f'/{joint}_position_controller')
        
        rospy.loginfo("PID Reconfigure node started")
    
    def reconfigure_callback(self, config, level):
        """处理动态重配置回调"""
        for joint in self.joints:
            try:
                # 获取该关节的PID参数
                p = config[f'{joint}_p']
                i = config[f'{joint}_i']
                d = config[f'{joint}_d']
                
                # 更新控制器参数
                self.update_pid(joint, p, i, d)
            except Exception as e:
                rospy.logerr(f"Failed to update {joint} PID: {str(e)}")
        
        return config
        
    def update_pid(self, joint, p, i, d):
        """更新指定关节的PID参数"""
        try:
            config = {
                'p': p,
                'i': i,
                'd': d
            }
            self.clients[joint].update_configuration(config)
            rospy.loginfo(f"Updated {joint} PID: P={p}, I={i}, D={d}")
        except Exception as e:
            rospy.logerr(f"Failed to update {joint} PID: {str(e)}")

if __name__ == '__main__':
    try:
        node = PIDReconfigure()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 