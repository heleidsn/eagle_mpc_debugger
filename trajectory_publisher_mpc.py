#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Trajectory publisher for geometric controller, using MPC trajectory
'''

import rospy
import numpy as np
import time
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from eagle_mpc_msgs.msg import MpcState
from utils.create_problem import get_opt_traj, create_mpc_controller
from utils.u_convert import thrustToForceTorqueAll

# from controller_msgs.msg import FlatTarget
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float32, Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from eagle_mpc_msgs.msg import SolverPerformance, MpcState, MpcControl
from l1_control.L1AdaptiveController_v1 import L1AdaptiveController_V1
from l1_control.L1AdaptiveController_v2 import L1AdaptiveControllerAll
from l1_control.L1AdaptiveController_v3 import L1AdaptiveControllerRefactored
from typing import Dict, Any, Tuple
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates

import crocoddyl
import pinocchio as pin

from eagle_mpc_viz import MpcController
from eagle_mpc_viz import WholeBodyStatePublisher
from eagle_mpc_viz import WholeBodyTrajectoryPublisher

from collections import deque


class TrajectoryPublisher:
    def __init__(self): 
        # Initialize ROS node
        rospy.init_node('trajectory_publisher_mpc', anonymous=False, log_level=rospy.INFO)
        
        # Dynamic reconfigure server
        self.enable_l1_control = False
        self.using_controller_v1 = False
        self.using_controller_v3 = False

        # Get parameters
        self.robot_name = rospy.get_param('~robot_name', 's500_uam')     # s500, s500_uam, hexacopter370_flying_arm_3
        self.trajectory_name = rospy.get_param('~trajectory_name', 'catch_vicon_real_new')   # displacement, catch_vicon
        self.dt_traj_opt = rospy.get_param('~dt_traj_opt', 20)  # ms
        self.use_squash = rospy.get_param('~use_squash', True)
        self.yaml_path = rospy.get_param('~yaml_path', '/home/jetson/catkin_ams/src/eagle_mpc_ros/eagle_mpc_yaml')
        self.control_rate = rospy.get_param('~control_rate', 50.0)  # Hz
        
        self.odom_source = rospy.get_param('~odom_source', 'mavros')  # mavros, gazebo  
        
        self.control_mode = rospy.get_param('~control_mode', 'MPC')  # MPC, Geometric, PX4, MPC_L1
        self.arm_enabled = rospy.get_param('~arm_enabled', True)
        self.arm_control_mode = rospy.get_param('~arm_control_mode', 'position_velocity')  # position, position_velocity, position_velocity_effort, effort
        
        self.max_thrust = rospy.get_param('~max_thrust', 8.0664 * 4)
        
        # for L1 controller
        self.l1_version = rospy.get_param('~l1_version', 'v2')  # v1, v2, v3
        self.As_coef = rospy.get_param('~As_coef', -1)
        self.filter_time_constant = rospy.get_param('~filter_time_constant', 0.4)
        
        self.mpc_iter_num = 0
        self.mpc_start_cost = 0.0
        self.mpc_final_cost = 0.0
        self.mpc_ref_index = 0
        self.traj_ref_index = 0
        
        self.mpc_controller = None
        self.l1_controller = None
        
        self.using_position_error_feedback = False
        
        self.last_times = deque(maxlen=100)  # 存储最近100次回调时间
        
        # set numpy print precision
        # np.set_printoptions(precision=2, suppress=True)
        np.set_printoptions(formatter={'float': lambda x: f"{x:>4.2f}"})  # 固定 6 位小数
        
        # Load trajectory
        self.load_trajectory()
        if self.control_mode == 'MPC':
            self.init_mpc_controller()
            self._init_l1_controller()
        
        # Subscriber
        self.current_state = State()
        self.arm_state = JointState()
        self.mav_state_sub = rospy.Subscriber('/mavros/state', State, self.mav_state_callback)
        self.arm_state_sub = rospy.Subscriber('/joint_states', JointState, self.arm_state_callback)
        
        # print("Waiting for MAVROS state...")
        # rospy.wait_for_message("/mavros/state", State, timeout=5)
        # print("MAVROS state received")
        
        if self.odom_source == 'mavros':
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback_model_local_position)
        else:
            self.odom_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state_gazebo)
            # print("Waiting for MAVROS local position...")
            # rospy.wait_for_message("/mavros/local_position/odom", Odometry, timeout=5)
            # print("MAVROS local position received")
        
        # Publishers
        self.pose_pub = rospy.Publisher('/reference/pose', PoseStamped, queue_size=10)
        # self.flat_target_pub = rospy.Publisher('/reference/flatsetpoint', FlatTarget, queue_size=10)
        self.mavros_setpoint_raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.yaw_pub = rospy.Publisher('/reference/yaw', Float32, queue_size=10)
        self.body_rate_thrust_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.mpc_state_pub = rospy.Publisher("/mpc/state", MpcState, queue_size=10)
        self.arm_control_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)
        
        self.partialTrajectoryPub = WholeBodyTrajectoryPublisher('whole_body_partial_trajectory_current',
                                                                     self.mpc_controller.robot_model,
                                                                     self.mpc_controller.platform_params,
                                                                     frame_id="world")
        
        self.statePub_target = WholeBodyStatePublisher('whole_body_state_target',
                                                self.mpc_controller.robot_model,
                                                self.mpc_controller.platform_params,
                                                frame_id="world")
        
        self.statePub_current = WholeBodyStatePublisher('whole_body_state_current',
                                                self.mpc_controller.robot_model,
                                                self.mpc_controller.platform_params,
                                                frame_id="world")
        
        # Initialize Path publisher
        self.path_pub = rospy.Publisher('uav_path', Path, queue_size=10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        # Services  
        # !Note: the service is only used for arm test, do not use it in real flight
        self.start_service = rospy.Service('start_arm_test', Trigger, self.start_arm_test)
        self.init_service = rospy.Service('initialize_trajectory', Trigger, self.initialize_trajectory)
        
        self.start_l1_control_service = rospy.Service('start_l1_control', Trigger, self.start_l1_control)
        self.stop_l1_control_service = rospy.Service('stop_l1_control', Trigger, self.stop_l1_control)
        self.start_trajectory_service = rospy.Service('start_trajectory', Trigger, self.start_trajectory)
        
        # --------------------------------------timer--------------------------------------
        # Timer 1: for publishing trajectory
        self.controller_started = False
        self.controller_started_last = False
        self.traj_finished = False
        self.timer = rospy.Timer(rospy.Duration(1.0/self.control_rate), self.controller_callback)
        
        # timer 2: 1 Hz state check to start MPC controller
        self.mpc_status_timer = rospy.Timer(rospy.Duration(1), self.mpc_status_time_callback)
        
        # Low pass filter parameters for joint effort
        self.filter_time_constant_arm_control = 1  # seconds
        self.filtered_effort = np.zeros(2)  # Store filtered effort values
        self.last_filter_time = rospy.Time.now()
        
        rospy.loginfo("Trajectory publisher initialized")
        
    def init_mpc_controller(self):
        # create mpc controller to get tau_f
        mpc_name = "rail"
        mpc_yaml = '{}/mpc/{}_mpc_real.yaml'.format(self.yaml_path, self.robot_name)
        
        # Initialize trajectory tracking MPC
        self.trajectory_mpc = create_mpc_controller(
            mpc_name,
            self.trajectory_obj,
            self.traj_state_ref,
            self.dt_traj_opt,
            mpc_yaml
        )
        
        # Create hover reference state (all states are the same as initial state)
        # note: self.traj_state_ref[0] is not the initial state, it is the first state of the trajectory
        
        # get initial state
        initial_state = self.trajectory_obj.initial_state
        
        hover_state_ref = [initial_state] * len(self.traj_state_ref)
        
        # Initialize hover MPC
        self.hover_mpc = create_mpc_controller(
            mpc_name,
            self.trajectory_obj,
            hover_state_ref,  # Use hover reference state
            self.dt_traj_opt,
            mpc_yaml
        )
        
        # Set initial MPC controller
        self.mpc_controller = self.hover_mpc
        
        self.state = initial_state.copy()
        
        self.thrust_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.speed_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.total_thrust = 0.0
        
        self.arm_joint_number = self.mpc_controller.robot_model.nq - 7
        
    def _init_l1_controller(self) -> None:
        """Initialize L1 adaptive controller."""
        dt_controller = 1.0/self.control_rate
        robot_model = self.mpc_controller.robot_model

        if self.l1_version == 'v3':
            self.l1_controller = L1AdaptiveControllerRefactored(
                dt=dt_controller, 
                robot_model=robot_model, 
                As_coef=self.As_coef,
                filter_time_constant=self.filter_time_constant
            )
        elif self.l1_version == 'v1':
            self.l1_controller = L1AdaptiveController_V1(
                dt=dt_controller, 
                robot_model=robot_model, 
                As_coef=self.As_coef,
                filter_time_constant=self.filter_time_constant
            )
        elif self.l1_version == 'v2':
            self.l1_controller = L1AdaptiveControllerAll(
                dt=dt_controller, 
                robot_model=robot_model, 
                As_coef=self.As_coef,
                filter_time_constant=self.filter_time_constant
            )
        else:
            rospy.logwarn("Invalid L1 controller version")
            return

        self.l1_controller.init_controller()

    def load_trajectory(self):
        """Load and initialize trajectory"""
        try:
            # Get trajectory from eagle_mpc
            self.traj_solver, self.traj_state_ref, traj_problem, self.trajectory_obj = get_opt_traj(
                self.robot_name,
                self.trajectory_name,
                self.dt_traj_opt,
                self.use_squash,
                self.yaml_path
            )
            
            self.trajectory_duration = self.trajectory_obj.duration  # ms
            rospy.loginfo(f"Loaded trajectory with duration: {self.trajectory_duration}ms")
            
            # Pre-calculate accelerations using finite differences, only used for other controllers
            self.accelerations = []
            dt = self.dt_traj_opt / 1000.0  # Convert to seconds
            
            for i in range(len(self.traj_state_ref)):
                # Get current velocities in body frame
                vel_body = self.traj_state_ref[i][7:10]  # [vx, vy, vz]
                quat = self.traj_state_ref[i][3:7]  # [qx, qy, qz, qw]
                
                # Convert body velocities to world frame
                R = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])[:3, :3]
                vel_world = R @ vel_body
                
                # Calculate acceleration using finite differences
                if i == 0:
                    # Forward difference for first point
                    vel_next = R @ self.traj_state_ref[i+1][7:10]
                    acc = (vel_next - vel_world) / dt
                elif i == len(self.traj_state_ref) - 1:
                    # Backward difference for last point
                    vel_prev = R @ self.traj_state_ref[i-1][7:10]
                    acc = (vel_world - vel_prev) / dt
                else:
                    # Central difference for interior points
                    vel_next = R @ self.traj_state_ref[i+1][7:10]
                    vel_prev = R @ self.traj_state_ref[i-1][7:10]
                    acc = (vel_next - vel_prev) / (2 * dt)
                
                self.accelerations.append(acc)
            
        except Exception as e:
            rospy.logerr(f"Failed to load trajectory: {str(e)}")
            raise

    def controller_callback(self, event):
        """Timer callback to publish trajectory setpoints"""
        
        now = time.time()
        self.last_times.append(now)

        # 计算频率
        if len(self.last_times) >= 2:
            dt_list = [t2 - t1 for t1, t2 in zip(self.last_times, list(self.last_times)[1:])]
            avg_dt = sum(dt_list) / len(dt_list)
            freq = 1.0 / avg_dt if avg_dt > 0 else 0
            rospy.loginfo_throttle(1.0, f"controller running freq: {freq:.2f} Hz")
        
        # Three conditions: not_started, started, finished       
        if self.controller_started and not self.traj_finished:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0)
            self.traj_ref_index = int(self.mpc_ref_index / self.dt_traj_opt)
            # if self.traj_ref_index == 0:
            #     self._init_l1_controller()
            #     print("L1 controller initialized")
            
            if self.traj_ref_index >= len(self.traj_state_ref):
                self.traj_finished = True
                self.controller_started_last = False
                self.traj_ref_index = len(self.traj_state_ref)-1
                rospy.loginfo("Trajectory finished")
                
            rospy.loginfo(f"Trajectory index: {self.traj_ref_index}/ {len(self.traj_state_ref)}")
                
        elif self.traj_finished:
            self.traj_ref_index = len(self.traj_state_ref)-1
        else:
            self.traj_ref_index = 0
            
        # Get current reference state
        ref_state = self.traj_state_ref[self.traj_ref_index]
        
        # Get rotation matrix from quaternion
        R = quaternion_matrix([ref_state[3], ref_state[4], ref_state[5], ref_state[6]])[:3, :3]
        
        # Convert body velocities to world frame
        vel_body = ref_state[7:10]
        vel_world = R @ vel_body
        
        # Get pre-calculated acceleration
        acc_world = self.accelerations[self.traj_ref_index]
        
        # Get yaw from quaternion
        quat = ref_state[3:7]
        yaw = euler_from_quaternion([quat[0], quat[1], quat[2], quat[3]])[2]
        
        if self.traj_finished:
            vel_world = np.zeros(3)
            acc_world = np.zeros(3)
        
        # Publish control command
        if self.control_mode == 'PX4':
            # using default PX4 controller, recieve p, v, a
            self.publish_mavros_setpoint_raw(ref_state[0:3], vel_world, acc_world, yaw, 0)  
        elif self.control_mode == 'Geometric':
            # using geometric controller
            self.publish_flat_target(ref_state[0:3], vel_world, acc_world)
            self.publish_reference_yaw(yaw)
        elif self.control_mode == 'MPC':
            # using MPC controller
            t0 = time.time()
            self.get_mpc_command()
            t1 = time.time()
            
            if self.enable_l1_control:
                # using L1 controller
                self.get_l1_control(self.state, self.mpc_ref_index)
            else:
                # using MPC controller
                self._init_l1_controller()
            t2 = time.time()
            
            self.publish_l1_control_command(self.l1_controller.u_mpc, self.l1_controller.u_ad, self.l1_controller.u_tracking)
            
            if self.arm_enabled:
                self.publish_arm_control_command()
                
            # debug info
            self.publish_mpc_l1_debug_data()

            t3 = time.time()
            
            rospy.loginfo_throttle(1.0, f"MPC calculate time: {(t1-t0)*1000:.3f} ms")
            rospy.loginfo_throttle(1.0, f"L1 state prep time: {(t2-t1)*1000:.3f} ms")
            rospy.loginfo_throttle(1, f"MPC calculate time: {(t1-t0)*1000:.3f} ms, L1 calculate time: {(t2-t1)*1000:.3f} ms, publish time: {(t3-t2)*1000:.3f} ms")
        else:
            rospy.logwarn("Invalid control mode")
            
        # Publish whole body state
        self.publish_wholebody_state_target()
        self.publish_wholebody_state_current()

    def get_mpc_command(self):
        self.mpc_controller.problem.x0 = self.state
            
        if self.controller_started:
            # Switch to trajectory tracking MPC
            self.mpc_controller = self.trajectory_mpc
            # self.controller_time = rospy.Time.now() - self.controller_start_time
            # self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0)
            # self.traj_ref_index = int(self.mpc_ref_index / self.dt_traj_opt)
        else:
            # Use hover MPC
            self.mpc_controller = self.hover_mpc
            # self.mpc_ref_index = 0
            # self.traj_ref_index = 0
            
        # update problem
        self.mpc_controller.updateProblem(self.mpc_ref_index)   # update problem using current time in ms
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()
        
        # get MPC debug info
        self.mpc_iter_num = self.mpc_controller.solver.iter
        self.solving_time = (time_end - time_start).to_sec()
        self.mpc_start_cost = self.mpc_controller.logger.costs[0]
        self.mpc_final_cost = self.mpc_controller.logger.costs[-1]
        rospy.logdebug("MPC index: {} Solving time: {}".format(self.mpc_ref_index, self.solving_time))
         
        # get mpc control command
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        
        rospy.logdebug('MPC state          : {}'.format(self.state))
        rospy.logdebug('MPC reference state: {}'.format(self.traj_state_ref[self.traj_ref_index]))
        rospy.logdebug('MPC control command: {}'.format(self.control_command))
        
        # get planned state
        # if self.controller_started:
        #     self.state_ref = self.mpc_controller.solver.xs[1]
        # else:
        #     self.state_ref = self.traj_state_ref[self.traj_ref_index]
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        # get body rate command from planned next state
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        # get thrust command
        self.total_thrust = np.sum(self.thrust_command)
      
    def get_l1_control(self, current_state: np.ndarray, time_step: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get control command from L1 controller."""
        
        # Get the baseline control from mpc_controller_l1
        baseline_control = np.copy(self.mpc_controller.solver.us_squash[0])
        
        # Convert to force/torque
        baseline_control_ft = thrustToForceTorqueAll(
                baseline_control,
                self.mpc_controller.platform_params.tau_f
            )
        
        index_plan = self.traj_ref_index
        
        if self.l1_version == 'v3':
            self.l1_controller.compute_control(current_state.copy(), baseline_control_ft)
        elif self.l1_version == 'v1':
            # Update current state and reference
            self.l1_controller.current_state = current_state.copy()
            self.l1_controller.z_ref_all = self.traj_state_ref[index_plan].copy()
            self.l1_controller.z_ref = self.l1_controller.get_state_angle_single_rad(self.l1_controller.z_ref_all)[self.l1_controller.state_dim_euler:]
            
            # update z_real using current state
            self.l1_controller.z_real = self.l1_controller.get_state_angle_single_rad(self.l1_controller.current_state)[self.l1_controller.state_dim_euler:]
            
            # transfer z_ref and z_measure to anglself.control_command
            self.l1_controller.u_mpc = baseline_control_ft.copy()
            
            # 1. Update state predictor
            self.l1_controller.update_z_hat()
            
            # 2. Update state predictor error
            self.l1_controller.update_z_tilde()
            
            # 3. Estimate disturbance
            self.l1_controller.update_sig_hat_v1()
            
            # 4. Filter the matched uncertainty estimate
            self.l1_controller.update_u_ad()
        else:
            
            # Update current state and reference
            t1 = time.time()
            self.l1_controller.current_state = current_state.copy()
            self.l1_controller.z_ref_all = self.traj_state_ref[index_plan].copy()
            self.l1_controller.z_ref = self.l1_controller.get_state_angle_single_rad(self.l1_controller.z_ref_all)
            
            # update z_real using current state
            self.l1_controller.z_real = self.l1_controller.get_state_angle_single_rad(self.l1_controller.current_state)
            
            # transfer z_ref and z_measure to anglself.control_command
            self.l1_controller.u_mpc = baseline_control_ft.copy()
            
            t2 = time.time()
            # 1. Update state predictor
            self.l1_controller.update_z_hat()
            
            t3 = time.time()
            
            # 2. Update state predictor error
            self.l1_controller.update_z_tilde()
            
            t4 = time.time()
            # 3. Estimate disturbance
            self.l1_controller.update_sig_hat_all_v2_new()
            t5 = time.time()
            # 4. Filter the matched uncertainty estimate
            self.l1_controller.update_u_ad()
            t6 = time.time()
            
            # rospy.loginfo(f"L1 state prep time: {(t2-t1)*1000:.3f} ms")
            # rospy.loginfo(f"L1 update_z_hat time: {(t3-t2)*1000:.3f} ms")
            # rospy.loginfo(f"L1 update_z_tilde time: {(t4-t3)*1000:.3f} ms")
            # rospy.loginfo(f"L1 update_sig_hat time: {(t5-t4)*1000:.3f} ms")
            # rospy.loginfo(f"L1 update_u_ad time: {(t6-t5)*1000:.3f} ms")
            # rospy.loginfo(f"L1 total time: {(t6-t1)*1000:.3f} ms")
            
              
    def publish_mpc_control_command(self):
        '''
        发布 MPC 控制指令
            self.control_command: 优化求解器得到的控制指令，包括推力和机械臂控制
            self.thrust_command: 优化求解器得到的推力指令
        '''
        
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 设置 type_mask，忽略姿态，仅使用角速度 + 推力
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE 
        
        # 机体系角速度 (rad/s)
        att_msg.body_rate = Vector3(self.roll_rate_ref, self.pitch_rate_ref, self.yaw_rate_ref)  # 仅绕 Z 轴旋转 0.1 rad/s
        
        # 推力值 (范围 0 ~ 1)
        att_msg.thrust = self.total_thrust / self.max_thrust  # 60% 油门
        
        # 对推力进行限幅
        att_msg.thrust = np.clip(att_msg.thrust, 0, 1)

        self.body_rate_thrust_pub.publish(att_msg)
    
    def publish_arm_control_command(self):
        '''
        description: publish arm control command
        return {*}
        '''        
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint_1', 'joint_2']
        
        # get reference state
        ref_state = self.traj_state_ref[self.traj_ref_index]
        ref_control = self.traj_solver.us[min(self.traj_ref_index, len(self.traj_solver.us)-1)]
        
        if self.arm_control_mode == 'position':
            joint_msg.position = -ref_state[7:9]
            joint_msg.velocity = [0.2, 0.2]
            joint_msg.effort = [0.0, 0.0]
        elif self.arm_control_mode == 'position_velocity':
            joint_msg.position = -ref_state[7:9]
            joint_msg.velocity = -ref_state[-2:]
            joint_msg.effort = [0.0, 0.0]
        elif self.arm_control_mode == 'position_velocity_effort':  # this mode is not working, effort is not used
            joint_msg.position = -ref_state[7:9]
            joint_msg.velocity = -ref_state[-2:]
            if joint_msg.velocity[0] or joint_msg.velocity[1] == 0:
                joint_msg.velocity = [1, 1]
            joint_msg.effort = -ref_control[-2:]
        elif self.arm_control_mode == 'effort':
            joint_msg.position = [0.0, 0.0]
            joint_msg.velocity = [0.0, 0.0]
            
            # Calculate time step for filter
            current_time = rospy.Time.now()
            dt = (current_time - self.last_filter_time).to_sec()
            self.last_filter_time = current_time
            
            # Apply low pass filter to effort command
            alpha = dt / (self.filter_time_constant_arm_control + dt)  # Filter coefficient
            raw_effort = np.array([-self.control_command[-2], -self.control_command[-1]])
            self.filtered_effort = alpha * raw_effort + (1 - alpha) * self.filtered_effort
            
            joint_msg.effort = self.filtered_effort.tolist()
            
        # print debug info
        print('current state: ', self.state)
        print('control command: ', self.control_command)
            
        # add constrain to joint position
        joint_msg.position[0] = np.clip(joint_msg.position[0], -1.57, 1.57)
        joint_msg.position[1] = np.clip(joint_msg.position[1], -1.57, 1.57)
        
        joint_msg.velocity[0] = np.clip(joint_msg.velocity[0], -1.0, 1.0)
        joint_msg.velocity[1] = np.clip(joint_msg.velocity[1], -1.0, 1.0)
        
        effort_limit = 0.2
        joint_msg.effort[0] = np.clip(joint_msg.effort[0], -effort_limit, effort_limit)
        joint_msg.effort[1] = np.clip(joint_msg.effort[1], -effort_limit, effort_limit)
        
        self.arm_control_pub.publish(joint_msg)
        
    def publish_l1_control_command(self, u_mpc, u_ad, u_tracking):
        '''
        发布L1的控制指令
            u_ad: L1控制器得到的控制指令，包括推力、力矩和机械臂控制力矩， 如何转换成 PX4 控制指令 推力+角速度？
            如何得到角速度？
            可以用当前的控制量在名义模型上进行仿真，得到下一时刻的角速度，设置为期望角速度
            
            【solved】:将L1得到的力矩作为期望角速度的增量
        '''
        
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        # get body rate command from MPC
        self.roll_rate_ref_old = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref_old = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref_old = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        # get additional body rate control command
        model = self.mpc_controller.robot_model
        data = model.createData()
        q = self.state[:model.nq]    # state
        v = self.state[model.nq:]    # velocity
        u = u_mpc + u_ad                      # using l1 control command
        dt = 1 / self.control_rate
        v_hat = pin.aba(model, data, q, v, u_ad)   # 使用u_ad来得到角速度的增量
        
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3] + v_hat[3] * dt
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4] + v_hat[4] * dt
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5] + v_hat[5] * dt
        
        # get thrust command
        self.total_thrust = np.sum(self.thrust_command) + u_ad[2] + u_tracking[2] # add u_ad[2] to total thrust +10
        
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 设置 type_mask，忽略姿态，仅使用角速度 + 推力
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE 
        
        # 机体系角速度 (rad/s)
        att_msg.body_rate = Vector3(self.roll_rate_ref_old, self.pitch_rate_ref_old, self.yaw_rate_ref_old)  # 仅绕 Z 轴旋转 0.1 rad/s
        
        # 推力值 (范围 0 ~ 1)
        att_msg.thrust = self.total_thrust / self.max_thrust  # 60% 油门
        
        # 对推力进行限幅
        att_msg.thrust = np.clip(att_msg.thrust, 0, 1)

        self.body_rate_thrust_pub.publish(att_msg)
            
    def mpc_status_time_callback(self, event):
        # check if the controller is started
        if self.controller_started:
            rospy.loginfo("MPC controller is started")
        else:
            rospy.loginfo("MPC controller is not started")
            
        # check if model is offboard
        if self.current_state.mode == "OFFBOARD":
            rospy.loginfo("Model is offboard")
        else:
            rospy.loginfo("Model is not offboard")
            self.controller_started = False
        
        # check if model is armed
        if self.current_state.armed:
            rospy.loginfo("Model is armed")
        else:
            rospy.loginfo("Model is not armed")
            self.controller_started = False
        
    def publish_body_rate_thrust(self, yaw_rate, thrust):
        msg = AttitudeTarget()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map" 

        msg.type_mask = AttitudeTarget.IGNORE_PX | AttitudeTarget.IGNORE_PY | AttitudeTarget.IGNORE_PZ
        
        msg.body_rate.x = yaw_rate
        msg.body_rate.y = 0
        msg.body_rate.z = 0
        
        msg.thrust = thrust
        self.body_rate_thrust_pub.publish(msg)
        
    def publish_mpc_l1_debug_data(self):
        '''
        # 状态向量
        float64[] state           # 完整状态向量
        float64[] state_ref      # 参考状态向量
        float64[] state_error    # 状态误差

        int32 mpc_time_step      # 迭代位置
        float64 solving_time     # 求解时间
        
        # MPC控制指令
        float64[] u_mpc          # MPC控制指令
        
        # L1控制器调试信息
        # L1 debug information
        float64[] u_ad           # L1控制指令

        float64[] z_ref          # 参考状态
        float64[] z_hat          # 估计状态
        float64[] z_real         # 实际状态

        float64[] sig_hat        # 估计扰动
        float64[] z_tilde        # 状态误差
        float64[] z_tilde_ref        # 与参考状态的误差
        float64[] z_tilde_tracking   # for tracking error

        # tracking controller
        float64[] u_tracking     # 轨迹跟踪控制指令
        '''
        # 发布MPC状态
        debug_msg = MpcState()
        debug_msg.header.stamp = rospy.Time.now()
        
        current_state = self.state.tolist()
        state_ref = self.traj_state_ref[self.traj_ref_index]
        
        # transfer quaternion to euler        
        quat = current_state[3:7]
        euler = euler_from_quaternion(quat)
        state_array_new = np.hstack((current_state[0:3], euler, current_state[7:]))
        
        
        quat_ref = state_ref[3:7]
        euler_ref = euler_from_quaternion(quat_ref)
        state_array_ref_new = np.hstack((state_ref[0:3], euler_ref, state_ref[7:]))
        
        state_ref_next = self.state_ref.copy()
        euler_ref_next = euler_from_quaternion(state_ref_next[3:7])
        state_array_ref_next = np.hstack((state_ref_next[0:3], euler_ref_next, state_ref_next[7:]))
        
        debug_msg.state = state_array_new
        debug_msg.state_ref = state_array_ref_new
        debug_msg.state_error = state_array_ref_new - state_array_new
        debug_msg.state_ref_next = state_array_ref_next # state next reference
        
        # MPC performance info
        debug_msg.mpc_time_step = self.mpc_ref_index
        debug_msg.mpc_iter_num = self.mpc_iter_num
        debug_msg.solving_time = self.solving_time
        debug_msg.mpc_start_cost = self.mpc_start_cost
        debug_msg.mpc_final_cost = self.mpc_final_cost
        
        # u_mpc need to transform to force/torque
        u_mpc_motor_thrust = self.control_command[:self.mpc_controller.platform_params.n_rotors]
        baseline_control_ft = thrustToForceTorqueAll(
                u_mpc_motor_thrust,
                self.mpc_controller.platform_params.tau_f
            )
        
        debug_msg.u_mpc = self.control_command.tolist()
        
        # if self.enable_l1_control:
        debug_msg.u_ad = self.l1_controller.u_ad.tolist()
        
        debug_msg.z_ref = self.l1_controller.z_ref.tolist()
        debug_msg.z_hat = self.l1_controller.z_hat.tolist()
        debug_msg.z_real = self.l1_controller.z_real.tolist()
        
        debug_msg.sig_hat = self.l1_controller.sig_hat.tolist()
        
        debug_msg.z_tilde = self.l1_controller.z_tilde.tolist()
        debug_msg.z_tilde_ref = self.l1_controller.z_tilde_ref.tolist()
        debug_msg.z_tilde_tracking = self.l1_controller.z_tilde_tracking.tolist()
        debug_msg.u_tracking = self.l1_controller.u_tracking.tolist()
        
        # Calculate gripper position and orientation using forward kinematics
        model = self.trajectory_obj.robot_model
        data = model.createData()
        
        # Get current state
        q = self.state[:model.nq]    # position state
        v = self.state[model.nq:]    # velocity state
        
        # Update forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get gripper frame ID
        gripper_frame_id = model.getFrameId("gripper_link")
        if gripper_frame_id < model.nframes:
            # Get gripper position and orientation
            gripper_pose = data.oMf[gripper_frame_id]
            gripper_position = gripper_pose.translation
            gripper_orientation = pin.Quaternion(gripper_pose.rotation)
            
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            gripper_euler = pin.rpy.matrixToRpy(gripper_pose.rotation)
            
            # Add gripper pose to debug message
            debug_msg.gripper_position = gripper_position.tolist()
            debug_msg.gripper_orientation = [gripper_orientation.x, gripper_orientation.y, 
                                          gripper_orientation.z, gripper_orientation.w]
            debug_msg.gripper_euler = gripper_euler.tolist()  # [roll, pitch, yaw] in radians
        else:
            rospy.logwarn("Gripper frame not found in robot model")
            debug_msg.gripper_position = [0.0, 0.0, 0.0]
            debug_msg.gripper_orientation = [0.0, 0.0, 0.0, 1.0]
            debug_msg.gripper_euler = [0.0, 0.0, 0.0]
        
        # 发布消息
        self.mpc_state_pub.publish(debug_msg)
        
    def publish_mavros_setpoint_raw(self, pos_world, vel_world, acc_world, yaw, yaw_rate):
        """Publish setpoint using mavros setpoint raw for PX4 controller"""

        setpoint_msg = PositionTarget()
        setpoint_msg.header.stamp = rospy.Time.now()
        setpoint_msg.header.frame_id = "map"
        setpoint_msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        setpoint_msg.type_mask = setpoint_msg.IGNORE_YAW_RATE  # Don't ignore acceleration
        
        # setpoint_msg.type_mask = (
        #     PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
        #     PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ 
        # )
        
        setpoint_msg.position = Point(pos_world[0], pos_world[1], pos_world[2])
        
        setpoint_msg.velocity.x = vel_world[0]
        setpoint_msg.velocity.y = vel_world[1]
        setpoint_msg.velocity.z = vel_world[2]
        
        setpoint_msg.acceleration_or_force.x = acc_world[0]
        setpoint_msg.acceleration_or_force.y = acc_world[1]
        setpoint_msg.acceleration_or_force.z = acc_world[2]
        
        # yaw and yaw rate
        setpoint_msg.yaw = yaw
        setpoint_msg.yaw_rate = yaw_rate
        
        self.mavros_setpoint_raw_pub.publish(setpoint_msg)
        
    def publish_reference_yaw(self, yaw):
        yaw_msg = Float32()
        yaw_msg.data = yaw
        self.yaw_pub.publish(yaw_msg)
        
    # def publish_flat_target(self, pos_world, vel_world, acc_world):
        
    #     # Publish flat reference
    #     flat_target_msg = FlatTarget()
    #     flat_target_msg.header.stamp = rospy.Time.now()
    #     flat_target_msg.header.frame_id = "world"
    #     flat_target_msg.type_mask = flat_target_msg.IGNORE_SNAP_JERK  # Don't ignore acceleration
        
    #     # Position
    #     flat_target_msg.position.x = pos_world[0]
    #     flat_target_msg.position.y = pos_world[1]
    #     flat_target_msg.position.z = pos_world[2] + 3
        
    #     # Velocity (world frame)
    #     flat_target_msg.velocity.x = vel_world[0]
    #     flat_target_msg.velocity.y = vel_world[1]
    #     flat_target_msg.velocity.z = vel_world[2]
        
    #     # Acceleration (world frame)
    #     flat_target_msg.acceleration.x = acc_world[0]
    #     flat_target_msg.acceleration.y = acc_world[1]
    #     flat_target_msg.acceleration.z = acc_world[2]
        
    #     # Publish messages
    #     self.flat_target_pub.publish(flat_target_msg)
        
    def mav_state_callback(self, msg):
        self.current_state = msg
        
    def arm_state_callback(self, msg):
        self.arm_state = msg
        
        # update self.state
        self.state[7:7+self.arm_joint_number] = [-msg.position[-1], -msg.position[-2]]
        self.state[-2:] = [-msg.velocity[-1], -msg.velocity[-2]]
        
    def callback_model_state_gazebo(self, msg):
        model_states = msg
        try:
            index = model_states.name.index(self.robot_name)
            pose = model_states.pose[index]
            twist = model_states.twist[index]
            
            # Update the state with the pose and twist
            self.state[0:3] = [pose.position.x,
                            pose.position.y,
                            pose.position.z - 0.224]
            
            self.state[3:7] = [pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w]
            self.state[7+self.arm_joint_number:10+self.arm_joint_number] = [twist.linear.x,
                                twist.linear.y,
                                twist.linear.z]
            self.state[10+self.arm_joint_number:13+self.arm_joint_number] = [twist.angular.x,
                                twist.angular.y,
                                twist.angular.z]
        except ValueError:
            rospy.logerr("Robot model not found in Gazebo model states")
            return  
    
    def callback_model_local_position(self, msg):
        """处理来自MAVROS的本地位置信息"""
        pose = msg.pose.pose
        twist = msg.twist.twist
        
        
        state_new = np.copy(self.state)
        # 更新位置
        state_new[0:3] = [pose.position.x,
                        pose.position.y,
                        pose.position.z]
        # 更新姿态四元数
        state_new[3:7] = [pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w]
        
        nq = self.mpc_controller.state.nq
        # 更新线速度
        state_new[nq:nq+3] = [twist.linear.x,
                            twist.linear.y,
                            twist.linear.z]
        # 更新角速度
        state_new[nq+3:nq+6] = [twist.angular.x,
                                twist.angular.y,
                                twist.angular.z]
        
        self.state = state_new
        
    def start_arm_test(self, req):
        # self.controller_started = True
        self.current_state.mode = "OFFBOARD"
        self.current_state.armed = True
        
        return TriggerResponse(success=True, message="Trajectory started.")
    
    def initialize_trajectory(self, req):
        # 初始化轨迹数据
        self.controller_started = False
        self.current_state.mode = "POSCTL"
        self.current_state.armed = False
        self.traj_finished = False
        
        return TriggerResponse(success=True, message="Trajectory initialized.")
    
    def start_l1_control(self, req):
        self.l1_controller.init_controller()
        self.enable_l1_control = True
        return TriggerResponse(success=True, message="L1 control started.")
    
    def stop_l1_control(self, req):
        self.enable_l1_control = False
        # self.l1_controller.init_controller()
        return TriggerResponse(success=True, message="L1 control stopped.")
    
    def start_trajectory(self, req):
        # Check if we're in offboard mode and armed
        if self.current_state.mode != "OFFBOARD":
            return TriggerResponse(success=False, message="Must be in OFFBOARD mode to start trajectory")
            
        if not self.current_state.armed:
            return TriggerResponse(success=False, message="Must be armed to start trajectory")
            
        # Start trajectory
        self.controller_started = True
        self.traj_finished = False
        
        # Set the start time for the trajectory
        self.controller_start_time = rospy.Time.now()
        
        return TriggerResponse(success=True, message="Trajectory started")

    def publish_wholebody_state_target(self):
        x = self.traj_state_ref[self.traj_ref_index]
        nq = self.mpc_controller.robot_model.nq
        nRotors = self.mpc_controller.platform_params.n_rotors
        
        u = self.traj_solver.us_squash[self.traj_ref_index-1]
        
        # publish  t, q, v, thrusts, tau
        self.statePub_target.publish(0.123, x[:nq], x[nq:], u[:nRotors], u[nRotors:])

        # qs, vs, ts = [], [], []
        # for x in self.mpc_controller.xss[self.traj_ref_index]:
        #     qs.append(x[:nq])
        #     vs.append(x[nq:])
        #     ts.append(0.1)
        # if self.horizon_enabled:
        #     self.partialTrajectoryPub.publish(ts[0::2], qs[0::2], vs[0::2])

    def publish_wholebody_state_current(self):
        x = self.state
        nq = self.mpc_controller.robot_model.nq
        nRotors = self.mpc_controller.platform_params.n_rotors
        
        u = self.mpc_controller.solver.us_squash[0]
        
        # publish  t, q, v, thrusts, tau
        self.statePub_current.publish(0.123, x[:nq], x[nq:], u[:nRotors], u[nRotors:])
        
        # publish plan horizon
        qs, vs, ts = [], [], []
        for x in self.mpc_controller.solver.xs:
            qs.append(x[:nq])
            vs.append(x[nq:])
            ts.append(0.1)
        self.partialTrajectoryPub.publish(ts[0::2], qs[0::2], vs[0::2])
        
        # Add current position to the path
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = self.state[0]
        pose_msg.pose.position.y = self.state[1]
        pose_msg.pose.position.z = self.state[2]
        self.path_msg.poses.append(pose_msg)
        
        # 限制路径长度
        if len(self.path_msg.poses) > 1000:
            self.path_msg.poses.pop(0)  # 移除最早的点
        
        # Publish the path
        self.path_pub.publish(self.path_msg)
        
if __name__ == '__main__':
    try:
        trajectory_publisher = TrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
