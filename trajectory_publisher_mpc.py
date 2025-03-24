#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Trajectory publisher for geometric controller, using MPC trajectory
'''

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from eagle_mpc_msgs.msg import MpcState
from utils.create_problem import get_opt_traj, create_mpc_controller
from controller_msgs.msg import FlatTarget
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float32, Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from eagle_mpc_msgs.msg import SolverPerformance, MpcState, MpcControl


class TrajectoryPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('trajectory_publisher_mpc', anonymous=True)

        # Get parameters
        self.robot_name = rospy.get_param('~robot_name', 's500')
        self.trajectory_name = rospy.get_param('~trajectory_name', 'displacement')
        self.dt_traj_opt = rospy.get_param('~dt_traj_opt', 5)  # ms
        self.use_squash = rospy.get_param('~use_squash', True)
        self.yaml_path = rospy.get_param('~yaml_path', '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml')
        self.publish_rate = rospy.get_param('~publish_rate', 100.0)  # Hz
        
        self.control_mode = rospy.get_param('~control_mode', 'MPC')  # MPC, Geometric, PX4
        
        self.max_thrust = rospy.get_param('~max_thrust', 10.0664 * 4)
        
        self.mpc_iter_num = 0
        
        # Load trajectory
        self.load_trajectory()
        if self.control_mode == 'MPC':
            self.init_mpc_controller()
        
        # Subscriber
        self.mav_state_sub = rospy.Subscriber('/mavros/state', State, self.mav_state_callback)
        print("Waiting for MAVROS state...")
        rospy.wait_for_message("/mavros/state", State, timeout=5)
        print("MAVROS state received")
        
        if self.control_mode == 'MPC':
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback_model_local_position)
            print("Waiting for MAVROS local position...")
            rospy.wait_for_message("/mavros/local_position/odom", Odometry, timeout=5)
            print("MAVROS local position received")
        
        # Publishers
        self.pose_pub = rospy.Publisher('/reference/pose', PoseStamped, queue_size=10)
        self.flat_target_pub = rospy.Publisher('/reference/flatsetpoint', FlatTarget, queue_size=10)
        self.mavros_setpoint_raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        
        self.yaw_pub = rospy.Publisher('/reference/yaw', Float32, queue_size=10)
        
        self.body_rate_thrust_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.mpc_state_pub = rospy.Publisher("/mpc/state", MpcState, queue_size=10)
        
        # --------------------------------------timer--------------------------------------
        # Timer 1: for publishing trajectory
        self.controller_started = False
        self.traj_finished = False
        self.timer = rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.timer_callback)
        
        # timer 2: 1 Hz state check to start MPC controller
        self.mpc_status_timer = rospy.Timer(rospy.Duration(1), self.mpc_status_time_callback)
        
        rospy.loginfo("Trajectory publisher initialized")
        
    def init_mpc_controller(self):
        # create mpc controller to get tau_f
        mpc_name = "rail"
        mpc_yaml = '{}/mpc/{}_mpc.yaml'.format(self.yaml_path, self.robot_name)
        self.mpc_controller = create_mpc_controller(
            mpc_name,
            self.trajectory_obj,
            self.traj_state_ref,
            self.dt_traj_opt,
            mpc_yaml
        )
        
        self.state = self.mpc_controller.state.zero()
        
        self.thrust_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.speed_command = np.zeros(self.mpc_controller.platform_params.n_rotors)
        self.total_thrust = 0.0

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
            
            # Pre-calculate accelerations using finite differences
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

    def timer_callback(self, event):
        """Timer callback to publish trajectory setpoints"""
        
        # Three conditions: not_started, started, finished       
        if self.controller_started and not self.traj_finished:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0)
            self.traj_ref_index = int(self.mpc_ref_index / self.dt_traj_opt)
            
            if self.traj_ref_index >= len(self.traj_state_ref):
                self.traj_finished = True
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
        
        # Publish reference
        if self.control_mode == 'PX4':
            # using default PX4 controller, recieve p, v, a
            self.publish_mavros_setpoint_raw(ref_state[0:3], vel_world, acc_world, yaw, 0)  
        elif self.control_mode == 'Geometric':
            # using geometric controller
            self.publish_flat_target(ref_state[0:3], vel_world, acc_world)
            self.publish_reference_yaw(yaw)
        elif self.control_mode == 'MPC':
            # using MPC controller
            self.get_mpc_command()
            self.publish_mpc_control_command()
            self.publish_mpc_debug_data()
        else:
            rospy.logwarn("Invalid control mode")  
          
    def get_mpc_command(self):
        self.mpc_controller.problem.x0 = self.state
            
        if self.controller_started:
            self.controller_time = rospy.Time.now() - self.controller_start_time
            self.mpc_ref_index = int(self.controller_time.to_sec() * 1000.0)
        else:
            self.mpc_ref_index = 0
            
        # update problem
        self.mpc_controller.updateProblem(self.mpc_ref_index)   # update problem using current time in ms
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()
        self.solving_time = (time_end - time_start).to_sec()
        # 获取迭代次数
        self.mpc_iter_num = self.mpc_controller.solver.iter
        
    def publish_mpc_control_command(self):
        # using mavros setpoint to achieve rate control
        
        self.control_command = self.mpc_controller.solver.us_squash[0]
        self.thrust_command = self.control_command[:len(self.thrust_command)]
        
        # get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        # get body rate command
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        # get thrust command
        self.total_thrust = np.sum(self.thrust_command)
        
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
            
        if not self.controller_started and self.current_state.mode == "OFFBOARD" and self.current_state.armed:
            rospy.loginfo("All conditions met for MPC start")
            self.controller_started = True
            self.traj_finished = False
            self.controller_start_time = rospy.Time.now()
        else:
            # self.controller_started = False
            rospy.loginfo("Not all conditions met for MPC start")
        
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
        
    def publish_mpc_debug_data(self):
        '''
        # 状态向量
        float64[] state           # 完整状态向量
        float64[] state_ref      # 参考状态向量
        float64[] state_error    # 状态误差

        int32 mpc_time_step      # 迭代位置
        float64 solving_time     # 求解时间
        
        # 位置和姿态
        geometry_msgs/Point position
        geometry_msgs/Quaternion orientation
        geometry_msgs/Vector3 velocity
        geometry_msgs/Vector3 angular_velocity 
        '''
        # 发布MPC状态
        state_msg = MpcState()
        state_msg.header.stamp = rospy.Time.now()
        
        current_state = self.state.tolist()
        state_ref = self.traj_state_ref[self.traj_ref_index]
        
        # transfer quaternion to euler
        state_euler = np.zeros(len(current_state) - 1)
        state_euler_ref = np.zeros(len(current_state) - 1)
        
        quat = current_state[3:7]
        euler = euler_from_quaternion(quat)
        state_array_new = np.hstack((current_state[0:3], euler, current_state[7:]))
        
        
        quat_ref = state_ref[3:7]
        euler_ref = euler_from_quaternion(quat_ref)
        state_array_ref_new = np.hstack((state_ref[0:3], euler_ref, state_ref[7:]))
        
        
        state_msg.state = state_array_new
        state_msg.state_ref = state_array_ref_new
        state_msg.state_error = state_array_ref_new - state_array_new
        
        # 填充求解器信息
        # state_msg.mpc_time_step = self.mpc_ref_index
        state_msg.mpc_time_step = self.mpc_iter_num
        state_msg.solving_time = self.solving_time
        
        # 填充位置和姿态信息
        state_msg.position.x = self.state[0]
        state_msg.position.y = self.state[1]
        state_msg.position.z = self.state[2]
        state_msg.orientation.x = self.state[3]
        state_msg.orientation.y = self.state[4]
        state_msg.orientation.z = self.state[5]
        state_msg.orientation.w = self.state[6]
        
        
        # 发布消息
        self.mpc_state_pub.publish(state_msg)
        
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
        
    def publish_flat_target(self, pos_world, vel_world, acc_world):
        
        # Publish flat reference
        flat_target_msg = FlatTarget()
        flat_target_msg.header.stamp = rospy.Time.now()
        flat_target_msg.header.frame_id = "world"
        flat_target_msg.type_mask = flat_target_msg.IGNORE_SNAP_JERK  # Don't ignore acceleration
        
        # Position
        flat_target_msg.position.x = pos_world[0]
        flat_target_msg.position.y = pos_world[1]
        flat_target_msg.position.z = pos_world[2] + 3
        
        # Velocity (world frame)
        flat_target_msg.velocity.x = vel_world[0]
        flat_target_msg.velocity.y = vel_world[1]
        flat_target_msg.velocity.z = vel_world[2]
        
        # Acceleration (world frame)
        flat_target_msg.acceleration.x = acc_world[0]
        flat_target_msg.acceleration.y = acc_world[1]
        flat_target_msg.acceleration.z = acc_world[2]
        
        # Publish messages
        self.flat_target_pub.publish(flat_target_msg)
        
    def mav_state_callback(self, msg):
        self.current_state = msg
        
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
        

if __name__ == '__main__':
    try:
        trajectory_publisher = TrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
