'''
Author: Lei He
Date: 2025-04-02 09:55:01
LastEditTime: 2025-04-02 16:41:57
Description: publish arm test cmd using dynamixel_interface
Github: https://github.com/heleidsn
'''
#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState

import time
import eagle_mpc
import crocoddyl

class TrajectoryPublisher:
    def __init__(self):
        # 初始化节点
        rospy.init_node('trajectory_publisher_node')

        # 获取参数
        self.robot_name = rospy.get_param('~robot_name', 's500_uam')
        self.trajectory_pub = rospy.Publisher('/{}/trajectory'.format(self.robot_name), Float64MultiArray, queue_size=10)
        self.position_control_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)
        
        self.trajectory_name = rospy.get_param('~trajectory_name', 'arm_test')
        self.dt_traj_opt = rospy.get_param('~dt_traj_opt', 50)  # ms
        self.use_squash = rospy.get_param('~use_squash', True)
        self.yaml_path = rospy.get_param('~yaml_path', '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml')

        # 服务来启动和初始化轨迹
        self.start_service = rospy.Service('start_trajectory', Trigger, self.start_trajectory)
        self.init_service = rospy.Service('initialize_trajectory', Trigger, self.initialize_trajectory)

        # 轨迹数据
        self.trajectory_data = Float64MultiArray()
        self.is_trajectory_initialized = False
        
        self.get_trajectory()
        # self.initialize_trajectory()
        
    def get_opt_traj(self, robotName, trajectoryName, dt_traj_opt, useSquash, yaml_file_path):
        '''
        description: get optimized trajectory
        '''
        trajectory_config_path = '{}/trajectories/{}_{}.yaml'.format(yaml_file_path, robotName, trajectoryName)
        trajectory = eagle_mpc.Trajectory()
        trajectory.autoSetup(trajectory_config_path)
        problem = trajectory.createProblem(dt_traj_opt, useSquash, "IntegratedActionModelEuler")

        if useSquash:
            solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)
        else:
            solver = crocoddyl.SolverBoxFDDP(problem)

        # solver.convergence_init = 1e-4
        
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
        start_time = time.time()
        solver.solve([], [], maxiter=400)
        end_time = time.time()
        
        print("Time taken for trajectory optimization: {:.2f} ms".format((end_time - start_time)*1000))
        
        traj_state_ref = solver.xs
        
        return solver, traj_state_ref, problem, trajectory 
        
    def get_trajectory(self):
        """Load and initialize trajectory"""
        try:
            # Get trajectory from eagle_mpc
            self.traj_solver, self.traj_state_ref, traj_problem, self.trajectory_obj = self.get_opt_traj(
                self.robot_name,
                self.trajectory_name,
                self.dt_traj_opt,
                self.use_squash,
                self.yaml_path
            )
            
            self.trajectory_duration = self.trajectory_obj.duration  # ms
            rospy.loginfo(f"Loaded trajectory with duration: {self.trajectory_duration}ms")
        except Exception as e:
            rospy.logerr(f"Failed to load trajectory: {str(e)}")
            raise

    def initialize_trajectory(self, req):
        # 初始化轨迹数据
        msg = JointState()
        
        # 设置header
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        
        # 设置关节名称
        msg.name = ['joint_1', 'joint_2']
        
        # 从优化轨迹中获取初始关节位置
        initial_state = self.traj_state_ref[0]
        # 假设前6个状态是关节角度
        msg.position = initial_state[7:9].tolist()
        # msg.velocity = [0.0, 0.0]
        # msg.effort = [0.0, 0.0]
        
        # 发布初始关节状态
        self.position_control_pub.publish(msg)
        
        self.is_trajectory_initialized = True
        rospy.loginfo("Trajectory initialized with initial joint states: {}".format(msg.position))
        return TriggerResponse(success=True, message="Trajectory initialized.")

    def start_trajectory(self, req):
        if not self.is_trajectory_initialized:
            return TriggerResponse(success=False, message="Trajectory not initialized.")

        # 发布轨迹数据
        total_time_ms = self.trajectory_duration
        dt_ms = self.dt_traj_opt
        rate = rospy.Rate(1000/dt_ms)  # 10 Hz
        traj_len = len(self.traj_state_ref)
        for i in range(traj_len):  # 假设发布100个点
            
            position = self.traj_state_ref[i][7:9]
            velocity = self.traj_state_ref[i][-2:]   
                     
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ['joint_1', 'joint_2']
            msg.position = position
            msg.velocity = velocity
            
            self.position_control_pub.publish(msg)
            
            rospy.loginfo("Trajectory initialized with initial joint states: pose {} vel {}".format(msg.position, msg.velocity))
            rate.sleep()

        return TriggerResponse(success=True, message="Trajectory started.")

if __name__ == "__main__":
    try:
        trajectory_publisher = TrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass