#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Trajectory publisher for geometric controller, using MPC trajectory
'''

import rospy
import numpy as np
import time
import math
import threading
import queue
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from eagle_mpc_msgs.msg import MpcState
from utils.create_problem import get_opt_traj, create_mpc_controller
from utils.u_convert import thrustToForceTorqueAll

# from controller_msgs.msg import FlatTarget
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from mavros_msgs.srv import SetMode, SetModeRequest
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion, Twist
from std_msgs.msg import Float32, Header, Float64
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from eagle_mpc_msgs.msg import SolverPerformance, MpcState, MpcControl
from l1_control.L1AdaptiveController_v1 import L1AdaptiveController_V1
from l1_control.L1AdaptiveController_v2 import L1AdaptiveControllerAll
from l1_control.L1AdaptiveController_v3 import L1AdaptiveControllerRefactored
from typing import Dict, Any, Tuple
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, GetModelStateResponse

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
        self.trajectory_name = rospy.get_param('~trajectory_name', 'catch_vicon_real')   # displacement, catch_vicon
        self.dt_traj_opt = rospy.get_param('~dt_traj_opt', 50)  # ms
        self.use_squash = rospy.get_param('~use_squash', True)
        self.use_simulation = rospy.get_param('~use_simulation', False)   #  if true, publish arm control command for ros_control
        self.yaml_path = rospy.get_param('~yaml_path', 'config/yaml')
        self.control_rate = rospy.get_param('~control_rate', 50.0)  # Hz
        
        self.odom_source = rospy.get_param('~odom_source', 'mavros')  # mavros, gazebo  
        
        self.control_mode = rospy.get_param('~control_mode', 'MPC')  # MPC, Geometric, PX4, MPC_L1
        self.arm_enabled = rospy.get_param('~arm_enabled', True)
        self.arm_control_mode = rospy.get_param('~arm_control_mode', 'position')  # position, position_velocity, position_velocity_effort, effort
        
        self.max_thrust = rospy.get_param('~max_thrust', 8.0664 * 4)
        
        # Control limits for L1 controller
        self.max_angular_velocity = rospy.get_param('~max_angular_velocity', math.radians(120))  # rad/s
        self.min_thrust_cmd = rospy.get_param('~min_thrust_cmd', 0.0)  # Minimum thrust command pass to px4
        self.max_thrust_cmd = rospy.get_param('~max_thrust_cmd', 0.75)  # Maximum thrust command pass to px4
        
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
        
        # Load trajectory
        self.load_trajectory()
        
        # Initialize MPC and L1 controller
        if self.control_mode == 'MPC':
            self.init_mpc_controller()
            self.init_l1_controller()
        
        # Subscriber
        self.current_state = State()
        self.arm_state = JointState()
        self.mav_state_sub = rospy.Subscriber('/mavros/state', State, self.mav_state_callback)
        if self.use_simulation:
            self.arm_state_sub = rospy.Subscriber('/arm_controller/joint_states', JointState, self.arm_state_sim_callback)
        else:
            self.arm_state_sub = rospy.Subscriber('/joint_states', JointState, self.arm_state_callback)
        
        if self.odom_source == 'mavros':
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback_model_local_position)
        else:
            self.odom_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state_gazebo)
        
        # Publishers
        self.pose_pub = rospy.Publisher('/reference/pose', PoseStamped, queue_size=10)
        # self.flat_target_pub = rospy.Publisher('/reference/flatsetpoint', FlatTarget, queue_size=10)
        self.mavros_setpoint_raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.yaw_pub = rospy.Publisher('/reference/yaw', Float32, queue_size=10)
        self.body_rate_thrust_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.mpc_state_pub = rospy.Publisher("/mpc/state", MpcState, queue_size=10)
        self.arm_control_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)
        
        # Create publishers for each joint position controller
        self.joint1_pub = rospy.Publisher('/arm_controller/joint_1_position_controller/command', Float64, queue_size=10)
        self.joint2_pub = rospy.Publisher('/arm_controller/joint_2_position_controller/command', Float64, queue_size=10)
        self.joint3_pub = rospy.Publisher('/arm_controller/joint_3_position_controller/command', Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher('/arm_controller/joint_4_position_controller/command', Float64, queue_size=10)
        
        # arm control publisher
        
        
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
        
        # Add gripper control services
        self.open_gripper_service = rospy.Service('open_gripper', Trigger, self.open_gripper_callback)
        self.close_gripper_service = rospy.Service('close_gripper', Trigger, self.close_gripper_callback)
        self.reset_beer_service = rospy.Service('reset_beer', Trigger, self.reset_beer_callback)
        
        # Wait for Gazebo services
        if self.use_simulation:
            rospy.wait_for_service('/gazebo/get_model_state')
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Initial beer position
        self.beer_initial_euler = np.array([-1.5708, -1.50229, -1.4e-05])
        # transfer euler to quaternion
        self.beer_initial_quat = quaternion_from_euler(self.beer_initial_euler[0], self.beer_initial_euler[1], self.beer_initial_euler[2])
        self.beer_initial_pose = Pose(
            position=Point(x=0.0, y=-0.12, z=0.83),
            orientation=Quaternion(x=self.beer_initial_quat[0], y=self.beer_initial_quat[1], z=self.beer_initial_quat[2], w=self.beer_initial_quat[3])
        )
        self.beer_initial_twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0)
        )
        
        # --------------------------------------timer--------------------------------------
        # Timer 1: for publishing trajectory
        self.trajectory_started = False
        self.trajectory_started_last = False
        self.traj_finished = False
        self.timer = rospy.Timer(rospy.Duration(1.0/self.control_rate), self.controller_callback)
        
        # timer 2: 1 Hz state check to start MPC controller
        self.mpc_status_timer = rospy.Timer(rospy.Duration(1), self.mpc_status_time_callback)
        
        # Low pass filter parameters for joint effort
        self.filter_time_constant_arm_control = 1  # seconds
        self.filtered_effort = np.zeros(2)  # Store filtered effort values
        self.last_filter_time = rospy.Time.now()
        
        # Grasping related variables
        self.grasp_position = np.array([0.0, 0.0, 0.85])  # Fixed target position for grasping
        self.grasp_time = rospy.Duration(3.0)  # Fixed grasp time (2 seconds)
        self.is_grasping = False    # Whether the gripper is grasping
        self.gripper_open_position = -0.7  # Position when gripper is open
        self.gripper_close_position = 0.0  # Position when gripper is closed
        self.grasp_start_time = None  # Will be set when trajectory starts
        
        # gripper settings for real flight test
        self.gripper_open_position_real = 0.6
        self.gripper_close_velocity_real = 1
        self.gripper_close_position_real = 0
        
        self.gripper_position_cmd = self.gripper_close_position_real  # start from closed position
        
        # Add MPC computation mode parameter
        self.use_multi_thread = rospy.get_param('~use_multi_thread', False)
        self.mpc_timeout = rospy.get_param('~mpc_timeout', 0.5)  # Timeout for MPC computation in seconds
        
        # Initialize thread-related attributes
        self.mpc_result_queue = queue.Queue()
        self.mpc_solver_thread = None
        self.last_valid_control = None  # Store last valid control command
        self.mpc_solving = False  # Flag to track if MPC is currently solving
        self.mpc_solve_start_time = None  # Track when MPC solving started
        
        # Add debugging statistics
        self.solve_times = deque(maxlen=100)  # Store last 100 solve times
        self.timeout_count = 0  # Count of timeouts
        self.error_count = 0  # Count of solver errors
        self.last_solve_status = None  # Last solver status
        self.last_error_msg = None  # Last error message
        
        # Add thread control
        self.thread_lock = threading.Lock()
        self.stop_thread = False
        
        # Initialize debug-related attributes
        self.solving_time = 0.0
        
        # Initialize control command
        self.control_command_mpc = None
        self.mpc_control_command_ft = None
        
        # State limit parameters
        self.position_limits = np.array([
            rospy.get_param('~max_position_xy', 3.0),  # max position in x, y direction
            rospy.get_param('~max_position_xy', 3.0),
            rospy.get_param('~max_position_z', 2.5)     # max position in z direction
        ])
        
        self.velocity_limits = np.array([
            rospy.get_param('~max_velocity_xy', 2.0),   # max velocity in x, y direction
            rospy.get_param('~max_velocity_xy', 2.0),
            rospy.get_param('~max_velocity_z', 2.0)     # max velocity in z direction
        ])
        
        self.angular_velocity_limits = np.array([
            rospy.get_param('~max_angular_velocity', 2.0),  # max angular velocity
            rospy.get_param('~max_angular_velocity', 2.0),
            rospy.get_param('~max_angular_velocity', 2.0)
        ])
        
        # Arm joint limit parameters
        self.arm_joint_limits = np.array([
            rospy.get_param('~max_arm_joint_angle', 1.57),  # max joint angle (±90 degrees)
            rospy.get_param('~max_arm_joint_angle', 1.57)
        ])
        
        self.arm_joint_velocity_limits = np.array([
            rospy.get_param('~max_arm_joint_velocity', 2.0),  # max joint angular velocity
            rospy.get_param('~max_arm_joint_velocity', 2.0)
        ])
        
        # Print parameter information
        rospy.loginfo("State limits parameters:")
        rospy.loginfo(f"Position limits (x,y,z): {self.position_limits}")
        rospy.loginfo(f"Velocity limits (x,y,z): {self.velocity_limits}")
        rospy.loginfo(f"Angular velocity limits: {self.angular_velocity_limits}")
        rospy.loginfo(f"Arm joint limits: {self.arm_joint_limits}")
        rospy.loginfo(f"Arm joint velocity limits: {self.arm_joint_velocity_limits}")
        
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
        
        self.hover_state_ref = [initial_state] * len(self.traj_state_ref)
        
        # Initialize hover MPC
        self.hover_mpc = create_mpc_controller(
            mpc_name,
            self.trajectory_obj,
            self.hover_state_ref,  # Use hover reference state
            self.dt_traj_opt,
            mpc_yaml
        )
        
        # Set initial MPC controller
        self.mpc_controller = self.hover_mpc
        
        self.state = initial_state.copy()
        print(f"initial state: {self.state}")
        
        self.speed_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.total_thrust = 0.0
        
        self.mpc_control_command_ft = np.zeros(6)
        
        self.arm_joint_number = self.mpc_controller.robot_model.nq - 7
        
    def init_l1_controller(self) -> None:
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
        """Timer callback to publish control command
        MPC:
            self.control_command_mpc: MPC control command for single rotor
        L1:
            self.l1_controller.u_mpc: L1 control command
            self.l1_controller.u_ad: L1 adaptive control command
            self.l1_controller.u_tracking: L1 tracking control command
        """
        
        now = time.time()
        self.last_times.append(now)

        # 计算频率
        if len(self.last_times) >= 2:
            dt_list = [t2 - t1 for t1, t2 in zip(self.last_times, list(self.last_times)[1:])]
            avg_dt = sum(dt_list) / len(dt_list)
            freq = 1.0 / avg_dt if avg_dt > 0 else 0
            rospy.loginfo_throttle(1.0, f"controller running freq: {freq:.2f} Hz")
        
        # 1. Get current reference state from planned trajectory
        # Three conditions: not_started, started, finished       
        if self.trajectory_started and not self.traj_finished:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0)
            self.traj_ref_index = int(self.mpc_ref_index / self.dt_traj_opt)
            
            if self.traj_ref_index >= len(self.traj_state_ref):
                self.traj_finished = True
                self.trajectory_started_last = False
                self.traj_ref_index = len(self.traj_state_ref)-1
                rospy.loginfo("Trajectory finished")
                
            rospy.loginfo(f"Trajectory index: {self.traj_ref_index}/ {len(self.traj_state_ref)}")
                
        elif self.traj_finished:
            self.traj_ref_index = len(self.traj_state_ref)-1
        else:
            self.traj_ref_index = 0
            
        ref_state = self.traj_state_ref[self.traj_ref_index]
        
        if self.control_mode == 'PX4' or self.control_mode == 'Geometric':
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
            t0 = time.time()
            # 1. Get MPC control command
            self.get_mpc_command()
            t1 = time.time()
            
            # 2. Get L1 control command
            if self.enable_l1_control:
                self.get_l1_control(self.state, self.mpc_ref_index)
            else:
                self.init_l1_controller()

            t2 = time.time()
            
            # 3. Publish control command
            self.publish_mpc_control_command(self.l1_controller.u_mpc, self.l1_controller.u_ad, self.l1_controller.u_tracking)
            
            if self.arm_enabled:
                self.publish_arm_control_command()
                # self.publish_gripper_control_command()
                
            # 4. Publish debug info
            self.publish_mpc_l1_debug_data()
            
            t3 = time.time()
            
            rospy.loginfo_throttle(1.0, f"mpc time: {(t1-t0)*1000:.2f} ms l1 time: {(t2-t1)*1000:.2f} publish time: {(t3-t2)*1000:.2f} ms")
        else:
            rospy.logwarn("Invalid control mode")
            
        # Publish whole body state
        # self.publish_wholebody_state_target()
        # self.publish_wholebody_state_current()

    def get_mpc_command(self):
        """Get MPC control command."""
        if self.use_multi_thread:
            self.get_mpc_command_multi_thread()
        else:
            self.get_mpc_command_single_thread()

    def get_mpc_command_single_thread(self):
        """Get MPC control command using single thread."""
        self.mpc_controller.problem.x0 = self.state
            
        # select MPC controller, trajectory tracking or hover
        if self.trajectory_started:
            # Switch to trajectory tracking MPC
            self.mpc_controller = self.trajectory_mpc
        else:
            # Use hover MPC
            self.mpc_controller = self.hover_mpc
            
        # 检查并限制输入状态
        # self.state = self.limit_state_input(self.state)
        
        # update problem
        self.mpc_controller.problem.x0 = self.state
        self.mpc_controller.updateProblem(self.mpc_ref_index)   # update problem using current time in ms
        
        time_start = rospy.Time.now()
        try:
            success = self.mpc_controller.solver.solve(
                self.mpc_controller.solver.xs,
                self.mpc_controller.solver.us,
                self.mpc_controller.iters
            )
            if not success or self.mpc_controller.safe_cb.cost > 20000:
                rospy.logerr("MPC solver failed, cost: {}".format(self.mpc_controller.safe_cb.cost))
                self.trajectory_started = False
                # switch to auto land mode
                # try:
                #     rospy.wait_for_service('/mavros/set_mode')
                #     set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                #     response = set_mode(custom_mode='AUTO.LAND')
                #     if response.mode_sent:
                #         rospy.loginfo("Successfully switched to PX4 auto land mode")
                #     else:
                #         rospy.logerr("Failed to switch to auto land mode")
                # except rospy.ServiceException as e:
                #     rospy.logerr(f"Service call failed: {e}")
                # return
        except Exception as e:
            rospy.logerr(f"Error in MPC solver: {str(e)}")
            return
        
        time_end = rospy.Time.now()
        
        # get mpc control command
        self.control_command_mpc = self.mpc_controller.solver.us_squash[0]
        
        # Convert to force/torque
        self.mpc_control_command_ft = thrustToForceTorqueAll(
            self.control_command_mpc,
            self.mpc_controller.platform_params.tau_f
        )
        
        # get MPC debug info
        self.mpc_iter_num = self.mpc_controller.solver.iter
        self.solving_time = (time_end - time_start).to_sec()
        self.mpc_start_cost = self.mpc_controller.logger.costs[0]
        self.mpc_final_cost = self.mpc_controller.logger.costs[-1]
        
        rospy.logdebug("MPC index: {} Solving time: {}".format(self.mpc_ref_index, self.solving_time))
        rospy.logdebug('MPC state          : {}'.format(self.state[7:]))
        rospy.logdebug('MPC reference state: {}'.format(self.traj_state_ref[self.traj_ref_index][7:]))
        rospy.logdebug('MPC control command: {}'.format(self.control_command_mpc))

    def solve_mpc_thread(self, q):
        """Solve MPC problem in a separate thread."""
        try:
            with self.thread_lock:
                if self.stop_thread:
                    return
                
                # Create a copy of the current state
                state_copy = self.state.copy()
                
                # select MPC controller, trajectory tracking or hover
                if self.trajectory_started:
                    # Switch to trajectory tracking MPC
                    mpc_controller = self.trajectory_mpc
                else:
                    # Use hover MPC
                    mpc_controller = self.hover_mpc
                    
                mpc_ref_index = self.mpc_ref_index
                
                # Update problem
                mpc_controller.problem.x0 = state_copy
                mpc_controller.updateProblem(mpc_ref_index)
            
            # Solve MPC problem
            time_start = rospy.Time.now()
            mpc_controller.solver.solve(
                mpc_controller.solver.xs,
                mpc_controller.solver.us,
                mpc_controller.iters
            )
            time_end = rospy.Time.now()
            solving_time = (time_end - time_start).to_sec()
            
            # Get control command
            control_command = mpc_controller.solver.us_squash[0]
            control_command_ft = thrustToForceTorqueAll(
                control_command,
                mpc_controller.platform_params.tau_f
            )
            
            # Put result in queue
            q.put({
                'success': True,
                'control_command': control_command,
                'control_command_ft': control_command_ft,
                'solving_time': solving_time,
                'mpc_iter_num': mpc_controller.iters,
                'mpc_start_cost': mpc_controller.solver.cost,
                'mpc_final_cost': mpc_controller.solver.cost
            })
            
        except Exception as e:
            rospy.logerr(f"Error in MPC solver thread: {str(e)}")
            q.put({
                'success': False,
                'error': str(e)
            })

    def get_mpc_command_multi_thread(self):
        """Get MPC control command using multi-threading with timeout."""
        current_time = rospy.Time.now()
        
        # Check if we need to start a new solve
        with self.thread_lock:
            if not self.mpc_solving:
                # Stop any existing thread
                self.stop_thread = True
                if self.mpc_solver_thread and self.mpc_solver_thread.is_alive():
                    self.mpc_solver_thread.join(timeout=0.1)
                
                # Start new solve
                self.stop_thread = False
                self.mpc_solving = True
                self.mpc_solve_start_time = current_time
                self.mpc_solver_thread = threading.Thread(target=self.solve_mpc_thread, args=(self.mpc_result_queue,))
                self.mpc_solver_thread.daemon = True
                self.mpc_solver_thread.start()
                rospy.logdebug("Started new MPC solve")
        
        # Check if current solve has timed out
        if (current_time - self.mpc_solve_start_time).to_sec() > self.mpc_timeout:
            with self.thread_lock:
                self.timeout_count += 1
                rospy.logwarn(f"MPC solver timeout after {self.mpc_timeout} seconds")
                self.mpc_solving = False
                self.stop_thread = True
                if self.last_valid_control is not None:
                    self.control_command_mpc = self.last_valid_control
                    rospy.logdebug("Using last valid control command")
                else:
                    self.control_command_mpc = np.zeros_like(self.control_command_mpc)
                    rospy.logwarn("No valid control command available, using zero control")
            return
        
        # Try to get result from queue
        try:
            result = self.mpc_result_queue.get_nowait()
            with self.thread_lock:
                self.mpc_solving = False
                self.stop_thread = True
                
                if result['success']:
                    # Update control commands and debug info
                    self.control_command_mpc = result['control_command']
                    self.last_valid_control = self.control_command_mpc.copy()
                    self.mpc_control_command_ft = result['control_command_ft']
                    self.mpc_iter_num = result['mpc_iter_num']
                    self.solving_time = result['solving_time']
                    self.mpc_start_cost = result['mpc_start_cost']
                    self.mpc_final_cost = result['mpc_final_cost']
                    
                    # Log detailed debug info
                    rospy.logdebug(f"MPC solve successful:")
                    rospy.logdebug(f"  - Index: {self.mpc_ref_index}")
                    rospy.logdebug(f"  - Solving time: {self.solving_time:.3f}s")
                    rospy.logdebug(f"  - Iterations: {self.mpc_iter_num}")
                    rospy.logdebug(f"  - Start cost: {self.mpc_start_cost:.3f}")
                    rospy.logdebug(f"  - Final cost: {self.mpc_final_cost:.3f}")
                    rospy.logdebug(f"  - Control command: {self.control_command_mpc}")
                else:
                    rospy.logwarn(f"MPC solver failed: {result['error']}")
                    if self.last_valid_control is not None:
                        self.control_command_mpc = self.last_valid_control
                        rospy.logdebug("Using last valid control command after failure")
                    else:
                        self.control_command_mpc = np.zeros_like(self.control_command_mpc)
                        rospy.logwarn("No valid control command available after failure")
                        
                    self.mpc_control_command_ft = thrustToForceTorqueAll(
                        self.control_command_mpc,
                        self.mpc_controller.platform_params.tau_f
                    )
                
        except queue.Empty:
            # No result yet, continue using last valid control
            if self.last_valid_control is not None:
                self.control_command_mpc = self.last_valid_control
            else:
                rospy.logwarn("No valid control command available, using zero control")
                self.control_command_mpc = np.zeros_like(self.mpc_controller.solver.us_squash[0])
        
        # Log periodic statistics
        if len(self.solve_times) > 0 and len(self.solve_times) % 10 == 0:
            avg_solve_time = sum(self.solve_times) / len(self.solve_times)
            rospy.loginfo(f"MPC Statistics:")
            rospy.loginfo(f"  - Average solve time: {avg_solve_time:.3f}s")
            rospy.loginfo(f"  - Timeout count: {self.timeout_count}")
            rospy.loginfo(f"  - Error count: {self.error_count}")
            rospy.loginfo(f"  - Last status: {self.last_solve_status}")
            if self.last_error_msg:
                rospy.loginfo(f"  - Last error: {self.last_error_msg}")

    def get_l1_control(self, current_state: np.ndarray, time_step: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get control command from L1 controller."""
        
        # Get the baseline control from mpc_controller_l1
        index_plan = self.traj_ref_index
        self.l1_controller.u_mpc = self.mpc_control_command_ft.copy()
        
        # Update current state and reference
        t1 = time.time()
        
        self.l1_controller.current_state = current_state.copy()
        self.l1_controller.z_ref_all = self.traj_state_ref[index_plan].copy()
        self.l1_controller.z_ref = self.l1_controller.get_state_angle_single_rad(self.l1_controller.z_ref_all)
        self.l1_controller.z_real = self.l1_controller.get_state_angle_single_rad(self.l1_controller.current_state)
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
    
    def publish_arm_control_command(self):
        '''
        description: publish arm control command
        return {*}
        '''        
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint_1', 'joint_2', 'joint_3']
        
        # get reference state
        ref_state = self.traj_state_ref[self.traj_ref_index]
        ref_control = self.traj_solver.us[min(self.traj_ref_index, len(self.traj_solver.us)-1)]
        
        # mpc_planned_state = self.mpc_controller.solver.xs[0]
        # mpc_planned_state_next = self.mpc_controller.solver.xs[1]
        # ref_state = mpc_planned_state_next
        
        current_time = rospy.Time.now()
        
        gripper_position = self.gripper_close_position_real
        gripper_velocity = 0
        
        # select gripper state
        if self.trajectory_started and not self.traj_finished:
            if current_time >= self.grasp_start_time and not self.is_grasping:
                # Close gripper at grasp time
                self.gripper_position_cmd = self.gripper_close_position_real
                gripper_velocity = self.gripper_close_velocity_real
                self.is_grasping = True
                rospy.loginfo("Gripper closed at grasp time")
            elif current_time < self.grasp_start_time:
                # Keep gripper open before grasp time
                self.gripper_position_cmd = self.gripper_open_position_real
                gripper_velocity = 0
                self.is_grasping = False
                rospy.loginfo_throttle(1.0, "Gripper open, waiting for grasp time")
        
        if self.arm_control_mode == 'position':
            joint_msg.position = [ref_state[7], ref_state[8], self.gripper_position_cmd]
            joint_msg.velocity = [0.0, 0.0, gripper_velocity]
            joint_msg.effort = [0.0, 0.0, 0.0]
        elif self.arm_control_mode == 'position_velocity':
            joint_msg.position = [ref_state[7], ref_state[8], self.gripper_position_cmd]
            joint_msg.velocity = [ref_state[-2], ref_state[-1], gripper_velocity]
            joint_msg.effort = [0.0, 0.0, 0.0]
        elif self.arm_control_mode == 'effort':
            joint_msg.position = [0.0, 0.0, 0.0]
            joint_msg.velocity = [0.0, 0.0, 0.0]
            
            # Calculate time step for filter
            current_time = rospy.Time.now()
            dt = (current_time - self.last_filter_time).to_sec()
            self.last_filter_time = current_time
            
            # Apply low pass filter to effort command
            alpha = dt / (self.filter_time_constant_arm_control + dt)  # Filter coefficient
            raw_effort = np.array([self.control_command[-2], self.control_command[-1]])
            self.filtered_effort = alpha * raw_effort + (1 - alpha) * self.filtered_effort
            
            joint_msg.effort = self.filtered_effort.tolist()
            
        # add constrain to joint position
        joint_msg.position[0] = np.clip(joint_msg.position[0], -1.3, 1.3)
        joint_msg.position[1] = np.clip(joint_msg.position[1], -0.8, 0.8)
        
        joint_msg.velocity[0] = np.clip(joint_msg.velocity[0], -1.0, 1.0)
        joint_msg.velocity[1] = np.clip(joint_msg.velocity[1], -1.0, 1.0)
        
        effort_limit = 0.2
        joint_msg.effort[0] = np.clip(joint_msg.effort[0], -effort_limit, effort_limit)
        joint_msg.effort[1] = np.clip(joint_msg.effort[1], -effort_limit, effort_limit)
        
        self.arm_control_pub.publish(joint_msg)
        
        if self.use_simulation:
            # Control arm joints
            self.joint1_pub.publish(Float64(joint_msg.position[0]))
            self.joint2_pub.publish(Float64(joint_msg.position[1]))

    def publish_gripper_control_command(self):
        '''
        description: publish gripper control command
        return {*}
        '''
        current_time = rospy.Time.now()
        
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint_3']
        
        if self.trajectory_started and not self.traj_finished:  # only publish gripper control command when trajectory is running
            if current_time >= self.grasp_start_time and not self.is_grasping:
                # Close gripper at grasp time
                if self.use_simulation:
                    self.joint3_pub.publish(Float64(self.gripper_close_position))
                    self.joint4_pub.publish(Float64(self.gripper_close_position))
                else:
                    joint_msg.position = [self.gripper_close_position_real]
                    joint_msg.velocity = [0.0]
                    joint_msg.effort = [0.0]
                    self.arm_control_pub.publish(joint_msg)
                self.is_grasping = True
                rospy.loginfo("Gripper closed at grasp time")
            elif current_time < self.grasp_start_time:
                # Keep gripper open before grasp time
                if self.use_simulation:
                    self.joint3_pub.publish(Float64(self.gripper_open_position))
                    self.joint4_pub.publish(Float64(self.gripper_open_position))
                else:
                    joint_msg.position = [self.gripper_open_position_real]
                    joint_msg.velocity = [0.0]
                    joint_msg.effort = [0.0]
                    self.arm_control_pub.publish(joint_msg)
                self.is_grasping = False
                rospy.loginfo_throttle(1.0, "Gripper open, waiting for grasp time")
              
    def publish_mpc_control_command(self, u_mpc, u_ad, u_tracking):
        '''
        发布L1的控制指令
            u_ad: L1控制器得到的控制指令，包括推力、力矩和机械臂控制力矩， 如何转换成 PX4 控制指令 推力+角速度？
            如何得到角速度？
            可以用当前的控制量在名义模型上进行仿真，得到下一时刻的角速度，设置为期望角速度
            
            【solved】:将L1得到的力矩作为期望角速度的增量
        '''
        # get collected thrust command
        if self.control_command_mpc is not None:
            self.control_command = self.control_command_mpc.copy()
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        # get body rate command from MPC
        self.roll_rate_ref_next_step = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref_next_step = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref_next_step = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        # get additional body rate control command
        model = self.mpc_controller.robot_model
        data = model.createData()
        q = self.state[:model.nq]    # state
        v = self.state[model.nq:]    # velocity
        u = u_mpc + u_ad                      # using l1 control command
        dt = 1 / self.control_rate
        rate_ad = pin.aba(model, data, q, v, u_ad)   # TODO: using u or u_ad?
        
        # Calculate angular velocity with limits
        self.roll_rate_ref = np.clip(
            self.roll_rate_ref_next_step + rate_ad[3] * dt,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )
        self.pitch_rate_ref = np.clip(
            self.pitch_rate_ref_next_step + rate_ad[4] * dt,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )
        self.yaw_rate_ref = np.clip(
            self.yaw_rate_ref_next_step + rate_ad[5] * dt,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )
        
        # Limit total thrust
        self.total_thrust = self.mpc_control_command_ft[2] + u_ad[2] + u_tracking[2]
        
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 设置 type_mask，忽略姿态，仅使用角速度 + 推力
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE 
        
        # 机体系角速度 (rad/s)
        att_msg.body_rate = Vector3(self.roll_rate_ref, self.pitch_rate_ref, self.yaw_rate_ref)
        
        # 推力值 (范围 0 ~ 1)
        att_msg.thrust = self.total_thrust / self.max_thrust
        
        # 对推力进行限幅
        att_msg.thrust = np.clip(att_msg.thrust, self.min_thrust_cmd, self.max_thrust_cmd)

        self.body_rate_thrust_pub.publish(att_msg)
            
    def mpc_status_time_callback(self, event):
        # check if the controller is started
        if self.trajectory_started:
            rospy.loginfo("MPC controller is started")
        else:
            rospy.loginfo("MPC controller is not started")
            
        # check if model is offboard
        if self.current_state.mode == "OFFBOARD":
            rospy.loginfo("Model is offboard")
        else:
            rospy.loginfo("Model is not offboard")
            self.trajectory_started = False
        
        # check if model is armed
        if self.current_state.armed:
            rospy.loginfo("Model is armed")
        else:
            rospy.loginfo("Model is not armed")
            self.trajectory_started = False
        
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
        
        if self.trajectory_started:
            state_ref = self.traj_state_ref[self.traj_ref_index]
        else:
            state_ref = self.hover_state_ref[0]
        
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
        
        # u_mpc
        debug_msg.u_mpc = self.mpc_control_command_ft.tolist()
        
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
        self.state[7:7+self.arm_joint_number] = [msg.position[-1], msg.position[-2]]
        self.state[-2:] = [msg.velocity[-1], msg.velocity[-2]]
        
    def arm_state_sim_callback(self, msg):
        self.arm_state = msg
        
        # update self.state
        self.state[7:7+self.arm_joint_number] = [msg.position[0], msg.position[1]]
        self.state[-2:] = [msg.velocity[0], msg.velocity[1]]
        
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
        # self.trajectory_started = True
        self.current_state.mode = "OFFBOARD"
        self.current_state.armed = True
        
        return TriggerResponse(success=True, message="Trajectory started.")
    
    def initialize_trajectory(self, req):
        # 初始化轨迹数据
        self.trajectory_started = False
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
        self.trajectory_started = True
        self.traj_finished = False
        
        # Set the start time for the trajectory
        self.controller_start_time = rospy.Time.now()
        # Set the grasp start time
        self.grasp_start_time = self.controller_start_time + self.grasp_time
        
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
        
    def set_grasp_target(self, position, grasp_time):
        """
        Set the target position and time for grasping
        
        Args:
            position: Target position for grasping
            grasp_time: Time to start grasping (rospy.Time)
        """
        self.grasp_position = position
        self.grasp_time = grasp_time
        rospy.loginfo(f"Set grasp target at position {position} and time {grasp_time}")
        
    def open_gripper_callback(self, req):
        """
        Service callback to open the gripper
        """
        if self.use_simulation:
            self.joint3_pub.publish(Float64(self.gripper_open_position))
            self.joint4_pub.publish(Float64(self.gripper_open_position))
            self.is_grasping = False
            rospy.loginfo("Gripper opened")
            return TriggerResponse(success=True, message="Gripper opened (Simulation mode)")
        else:
            self.gripper_position_cmd = self.gripper_open_position_real
            self.is_grasping = False
            rospy.loginfo("Gripper opened")
            return TriggerResponse(success=True, message="Gripper opened")

    def close_gripper_callback(self, req):
        """
        Service callback to close the gripper
        """
        if self.use_simulation:
            self.joint3_pub.publish(Float64(self.gripper_close_position))
            self.joint4_pub.publish(Float64(self.gripper_close_position))
            self.is_grasping = True
            rospy.loginfo("Gripper closed")
            return TriggerResponse(success=True, message="Gripper closed (Simulation mode)")
        else:
            self.gripper_position_cmd = self.gripper_close_position_real
            self.is_grasping = False
            rospy.loginfo("Gripper closed")
            return TriggerResponse(success=True, message="Gripper closed")

    def reset_beer_callback(self, req):
        """
        Service callback to reset the beer model position
        """
        try:
            # Create model state message
            model_state = ModelState()
            model_state.model_name = "beer"
            model_state.pose = self.beer_initial_pose
            model_state.twist = self.beer_initial_twist
            model_state.reference_frame = "world"

            # Call Gazebo service to set model state
            response = self.set_model_state(model_state)
            
            if response.success:
                rospy.loginfo("Beer model position reset successfully")
                return TriggerResponse(success=True, message="Beer model position reset successfully")
            else:
                rospy.logerr("Failed to reset beer model position")
                return TriggerResponse(success=False, message="Failed to reset beer model position")
                
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return TriggerResponse(success=False, message=f"Service call failed: {e}")

    def limit_state_input(self, state):
        """Limit state input ranges, especially for velocities"""
        try:
            state_limited = state.copy()
            
            # Position limits (x, y, z)
            state_limited[0:3] = np.clip(state[0:3], -self.position_limits, self.position_limits)
            
            # Quaternion normalization
            quat = state[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                state_limited[3:7] = quat / quat_norm
            else:
                state_limited[3:7] = np.array([0.0, 0.0, 0.0, 1.0])  # default orientation
                
            # Get nq from robot model
            nq = self.mpc_controller.robot_model.nq
            
            # Velocity limits (vx, vy, vz)
            state_limited[nq:nq+3] = np.clip(state[nq:nq+3], -self.velocity_limits, self.velocity_limits)
            
            # Angular velocity limits (wx, wy, wz)
            state_limited[nq+3:nq+6] = np.clip(state[nq+3:nq+6], -self.angular_velocity_limits, self.angular_velocity_limits)
            
            # Arm joint angle limits (start after quaternion)
            arm_joint_start = 7  # start index of arm joint angles (after quaternion)
            arm_joint_end = arm_joint_start + self.arm_joint_number
            state_limited[arm_joint_start:arm_joint_end] = np.clip(
                state[arm_joint_start:arm_joint_end], 
                -self.arm_joint_limits, 
                self.arm_joint_limits
            )
            
            # Arm joint angular velocity limits (at the end of state vector)
            arm_vel_start = -self.arm_joint_number  # start index from the end
            arm_vel_end = None  # until the end
            state_limited[arm_vel_start:arm_vel_end] = np.clip(
                state[arm_vel_start:arm_vel_end], 
                -self.arm_joint_velocity_limits, 
                self.arm_joint_velocity_limits
            )
            
            # Log if state was limited
            tolerance = 1e-2  # 设置一个合适的容差
            
            # Check position limits
            pos_diff = np.abs(state[0:3] - state_limited[0:3])
            if np.any(pos_diff > tolerance):
                rospy.logwarn("Position limits exceeded:")
                for i, (orig, lim) in enumerate(zip(state[0:3], state_limited[0:3])):
                    if abs(orig - lim) > tolerance:
                        rospy.logwarn(f"  {['x', 'y', 'z'][i]}: {orig:.3f} -> {lim:.3f} (limit: ±{self.position_limits[i]:.3f})")
            
            # Check velocity limits
            vel_diff = np.abs(state[nq:nq+3] - state_limited[nq:nq+3])
            if np.any(vel_diff > tolerance):
                rospy.logwarn("Velocity limits exceeded:")
                for i, (orig, lim) in enumerate(zip(state[nq:nq+3], state_limited[nq:nq+3])):
                    if abs(orig - lim) > tolerance:
                        rospy.logwarn(f"  {['vx', 'vy', 'vz'][i]}: {orig:.3f} -> {lim:.3f} (limit: ±{self.velocity_limits[i]:.3f})")
            
            # Check angular velocity limits
            ang_vel_diff = np.abs(state[nq+3:nq+6] - state_limited[nq+3:nq+6])
            if np.any(ang_vel_diff > tolerance):
                rospy.logwarn("Angular velocity limits exceeded:")
                for i, (orig, lim) in enumerate(zip(state[nq+3:nq+6], state_limited[nq+3:nq+6])):
                    if abs(orig - lim) > tolerance:
                        rospy.logwarn(f"  {['wx', 'wy', 'wz'][i]}: {orig:.3f} -> {lim:.3f} (limit: ±{self.angular_velocity_limits[i]:.3f})")
            
            # Check arm joint angle limits
            arm_angle_diff = np.abs(state[arm_joint_start:arm_joint_end] - state_limited[arm_joint_start:arm_joint_end])
            if np.any(arm_angle_diff > tolerance):
                rospy.logwarn("Arm joint angle limits exceeded:")
                for i, (orig, lim) in enumerate(zip(state[arm_joint_start:arm_joint_end], state_limited[arm_joint_start:arm_joint_end])):
                    if abs(orig - lim) > tolerance:
                        rospy.logwarn(f"  Joint {i+1} angle: {orig:.3f} -> {lim:.3f} (limit: ±{self.arm_joint_limits[i]:.3f})")
            
            # Check arm joint velocity limits
            arm_vel_diff = np.abs(state[arm_vel_start:arm_vel_end] - state_limited[arm_vel_start:arm_vel_end])
            if np.any(arm_vel_diff > tolerance):
                rospy.logwarn("Arm joint velocity limits exceeded:")
                for i, (orig, lim) in enumerate(zip(state[arm_vel_start:arm_vel_end], state_limited[arm_vel_start:arm_vel_end])):
                    if abs(orig - lim) > tolerance:
                        rospy.logwarn(f"  Joint {i+1} velocity: {orig:.3f} -> {lim:.3f} (limit: ±{self.arm_joint_velocity_limits[i]:.3f})")
            
            return state_limited
            
        except Exception as e:
            rospy.logerr(f"Error in state limiting: {str(e)}")
            return state  # return original state if error occurs

if __name__ == '__main__':
    try:
        trajectory_publisher = TrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
