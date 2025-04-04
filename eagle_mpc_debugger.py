'''
Author: Lei He
Date: 2025-02-19 11:40:31
LastEditTime: 2025-04-02 22:01:50
Description: MPC Debug Interface, useful for debugging your MPC controller before deploying it to the real robot
Github: https://github.com/heleidsn
'''
#!/usr/bin/env python3

import rospy
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from eagle_mpc_msgs.msg import MpcState, MpcControl
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from std_msgs.msg import Float64
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_matrix


from utils.create_problem import get_opt_traj, create_mpc_controller

from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import Vector3
from scipy.spatial.transform import Rotation as R
from utils.u_convert import thrustToForceTorqueAll_array


class MpcDebugInterface(QWidget):
    def __init__(self, using_ros=False, mpc_name='rail', mpc_yaml_path=None, robot_name='iris', trajectory_name='hover', dt_traj_opt=20, useSquash=True):
        super(MpcDebugInterface, self).__init__()
        self.setWindowTitle('MPC Debug Interface')
        
        self.using_ros = using_ros
        self.mpc_name = mpc_name
        self.mpc_yaml_path = mpc_yaml_path
        self.robot_name = robot_name
        self.trajectory_name = trajectory_name
        self.use_squash = useSquash
        self.dt_traj_opt = dt_traj_opt  # ms
        self.yaml_path = rospy.get_param('~yaml_path', '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml')
        self.arm_enabled = rospy.get_param('~arm_enabled', True)
        
        # numpy print options
        np.set_printoptions(formatter={'float': lambda x: f"{x:>4.2f}"})  # 固定 6 位小数
        
        # setting
        self.plot_thrust_torque = True
        
        # Create layout
        self.layout = QVBoxLayout()
        
        # Time control
        time_layout = QHBoxLayout()
        self.time_slider = QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(3000)  # 3 seconds
        self.time_slider.valueChanged.connect(self.time_changed)
        self.time_label = QLabel('0 ms')
        time_layout.addWidget(QLabel('Time (ms):'))
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_label)
        
        # Reference state selection
        ref_layout = QHBoxLayout()
        self.ref_selector = QtWidgets.QComboBox()
        self.ref_selector.addItems(['Current Reference', 'Current offset', 'Initial State', 'Final State'])
        self.ref_selector.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                min-width: 150px;
            }
            QComboBox:hover {
                border: 1px solid #4a90e2;
            }
        """)
        
        self.set_ref_button = QPushButton('Set to Reference')
        self.set_ref_button.clicked.connect(self.set_to_reference_button_clicked)
        self.set_ref_button.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5d8c;
            }
        """)
        
        ref_layout.addWidget(QLabel('Reference:'))
        ref_layout.addWidget(self.ref_selector)
        ref_layout.addWidget(self.set_ref_button)
        ref_layout.addStretch()

        # Add to main layout after time_layout
        self.layout.addLayout(ref_layout)
        
        # Mass modification layout
        mass_layout = QHBoxLayout()
        
        # Create mass input field
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setPlaceholderText("Enter new mass (kg)")
        self.mass_input.setFixedWidth(150)
        self.mass_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
            }
        """)
        
        # Add validator for mass input (only positive numbers)
        validator = QtGui.QDoubleValidator(0.0, 100.0, 3)  # min, max, decimals
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.mass_input.setValidator(validator)
        
        # Create set mass button
        self.set_mass_button = QPushButton('Set Mass')
        self.set_mass_button.clicked.connect(self.update_mass)
        self.set_mass_button.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5d8c;
            }
        """)
        
        # Current mass display
        self.current_mass_label = QLabel("Current Mass: 1.5 kg")  # 默认质量
        self.current_mass_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px;
            }
        """)
        
        # Add widgets to mass layout
        mass_layout.addWidget(QLabel("New Mass:"))
        mass_layout.addWidget(self.mass_input)
        mass_layout.addWidget(self.set_mass_button)
        mass_layout.addWidget(self.current_mass_label)
        mass_layout.addStretch()
        
        # Add mass layout to main layout after time layout
        self.layout.addLayout(mass_layout)
        
        # State modification
        state_layout = QHBoxLayout()
        self.state_sliders = {}
        self.state_labels = {}
        
        # Create state group layouts
        state_groups = {
            'Position (m)': ['X', 'Y', 'Z'],
            'Orientation (deg)': ['roll', 'pitch', 'yaw'],
            'Linear Velocity (m/s)': ['vx', 'vy', 'vz'],
            'Angular Velocity (deg/s)': ['wx', 'wy', 'wz']
        }
        
        # Create sliders and labels
        slider_configs = [
            # Position control (x, y, z)
            ('X', -2, 2),
            ('Y', -2, 2),
            ('Z', -2, 2),
            # Attitude Euler angles control (roll, pitch, yaw)
            ('roll', -90, 90),
            ('pitch', -90, 90),
            ('yaw', -90, 90),
            # Linear velocity control (vx, vy, vz)
            ('vx', -2, 2),
            ('vy', -2, 2),
            ('vz', -2, 2),
            # Angular velocity control (wx, wy, wz)
            ('wx', -90, 90),
            ('wy', -90, 90),
            ('wz', -90, 90)
        ]
        
        # Create a vertical layout for each group
        for group_name, states in state_groups.items():
            # Create a QGroupBox as container
            group_box = QtWidgets.QGroupBox()
            group_box.setTitle(group_name)
            
            # Create vertical layout as GroupBox's internal layout
            group_layout = QVBoxLayout(group_box)
            group_layout.setSpacing(10)
            group_layout.setContentsMargins(10, 15, 10, 10)  # Left, Top, Right, Bottom margins
            
            # Set GroupBox style
            group_box.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid gray;
                    border-radius: 5px;
                    margin-top: 1ex;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 5px;
                    color: darkblue;
                }
            """)
            
            # Create horizontal layout to place all states in this group
            states_layout = QHBoxLayout()
            states_layout.setSpacing(15)  # Increase spacing between states
            
            # Create sliders for each state in this group
            for state_name in states:
                # Find corresponding configuration
                config = next(cfg for cfg in slider_configs if cfg[0] == state_name)
                state_layout_single = self.create_state_slider(*config)
                states_layout.addLayout(state_layout_single)
            
            group_layout.addLayout(states_layout)
            state_layout.addWidget(group_box)
        
        # Set main state layout spacing
        state_layout.setSpacing(20)
        state_layout.setContentsMargins(10, 10, 10, 10)
        
        # Plot display: position, attitude, linear velocity, angular velocity
        self.figure = Figure(figsize=(15, 20), dpi=100)  # 添加dpi参数
        # Set spacing between subplots
        self.figure.subplots_adjust(
            left=0.08,    # Left margin
            right=0.95,   # Right margin
            bottom=0.08,  # Bottom margin
            top=0.95,     # Top margin
            wspace=0.25,  # Horizontal spacing between subplots
            hspace=0.35   # Vertical spacing between subplots，增加垂直间距
        )
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(1000)  # 设置最小高度
        self.canvas.setMinimumWidth(1000)   # 设置最小宽度
        
        if self.plot_thrust_torque:
            plot_row_num = 4
        else:
            plot_row_num = 3
            
        if self.arm_enabled:
            plot_col_num = 3
            self.ax_state = self.figure.add_subplot(plot_row_num, plot_col_num, 1)  # State
            self.ax_attitude = self.figure.add_subplot(plot_row_num, plot_col_num, 2)  # Attitude
            
            self.ax_linear_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 4)  # Linear velocity
            self.ax_angular_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 5)  # Angular velocity
            
            self.ax_control = self.figure.add_subplot(plot_row_num, plot_col_num, 7)  # Control
            self.ax_time = self.figure.add_subplot(plot_row_num, plot_col_num, 8)   # Solving time
            
            if self.plot_thrust_torque: 
                self.ax_thrust = self.figure.add_subplot(plot_row_num, plot_col_num, 10)  # Thrust
                self.ax_torque = self.figure.add_subplot(plot_row_num, plot_col_num, 11)  # Torque
            
            # for joint 
            self.ax_joint_position = self.figure.add_subplot(plot_row_num, plot_col_num, 3)  # Joint position
            self.ax_joint_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 6)  # Joint velocity
            self.ax_joint_effort = self.figure.add_subplot(plot_row_num, plot_col_num, 9)  # Joint effort
            self.ax_joint_control = self.figure.add_subplot(plot_row_num, plot_col_num, 12)  # Joint control
        else:
            plot_col_num = 2
            self.ax_state = self.figure.add_subplot(plot_row_num, plot_col_num, 1)  # State
            self.ax_attitude = self.figure.add_subplot(plot_row_num, plot_col_num, 2)  # Attitude
            
            self.ax_linear_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 3)  # Linear velocity
            self.ax_angular_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 4)  # Angular velocity
            
            self.ax_control = self.figure.add_subplot(plot_row_num, plot_col_num, 5)  # Control
            self.ax_time = self.figure.add_subplot(plot_row_num, plot_col_num, 6)   # Solving time
            
            if self.plot_thrust_torque: 
                self.ax_thrust = self.figure.add_subplot(plot_row_num, plot_col_num, 7)  # Thrust
                self.ax_torque = self.figure.add_subplot(plot_row_num, plot_col_num, 8)  # Torque
        
        
        
        # Add to main layout
        self.layout.addLayout(time_layout, stretch=1)
        self.layout.addLayout(state_layout, stretch=2)
        self.layout.addWidget(self.canvas, stretch=8)  # 给canvas更大的比例
        self.setLayout(self.layout)
      
        # Data storage
        self.state_history = []
        self.control_history = []
        self.mpc_solve_time_history = []  # Store solving time history
        
        self.joint_position_buffer = []
        self.joint_velocity_buffer = []
        self.joint_effort_buffer = []
        self.joint_control_buffer = []
        
        # Timer for updating plots
        self.timer_plot = QtCore.QTimer()
        self.timer_plot.timeout.connect(self.update_plot)
        self.timer_plot.start(1)  # 10Hz update
        
        # Load trajectory and initialize MPC controller
        self.load_trajectory()
        self.init_mpc_controller()
        
        self.state = self.mpc_controller.state.zero()
        self.state_ref = np.copy(self.mpc_controller.state_ref)
        self.mpc_ref_time = 0
        self.solving_time = 0.0
        
        # Set time slider maximum based on trajectory length
        traj_duration_ms = (len(self.state_ref) - 1) * self.dt_traj_opt
        self.time_slider.setMaximum(traj_duration_ms+2000)
        
        # Store MPC time step (convert to milliseconds)
        self.dt_mpc = self.mpc_controller.dt  # Convert to milliseconds
        print(f"Trajectory dt: {self.dt_traj_opt}ms, MPC dt: {self.dt_mpc}ms")
        
        # Create MPC controller with different timer
        if self.using_ros:
            rospy.loginfo("MPC controller initialized")
            
            # ROS subscribers and publishers
            self.pose_pub = rospy.Publisher('/debug/pose', PoseStamped, queue_size=1)
            # self.time_pub = rospy.Publisher('/debug/time', Float64, queue_size=1)
            
            # Subscriber
            self.current_state = State()
            self.arm_state = JointState()
            self.mav_state_sub = rospy.Subscriber('/mavros/state', State, self.mav_state_callback)
            self.arm_state_sub = rospy.Subscriber('/joint_states', JointState, self.arm_state_callback)
            
            self.state_sub = rospy.Subscriber('/debug/mpc_state', MpcState, self.state_callback)
            self.control_sub = rospy.Subscriber('/debug/mpc_control', MpcControl, self.control_callback)
            
            self.mpc_state_pub = rospy.Publisher('/mpc/state', MpcState, queue_size=10)
            self.mpc_control_pub = rospy.Publisher('/mpc/control', MpcControl, queue_size=10)
            
            self.solving_time_pub = rospy.Publisher('/mpc/solving_time', Float64, queue_size=1)
            
            self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
            
            # Start MPC controller and wait for it to start
            self.mpc_rate = 100.0  # Hz
            self.mpc_ref_index = 0
            self.mpc_timer = rospy.Timer(rospy.Duration(1.0/self.mpc_rate), self.mpc_timer_callback_ros)
            
            rospy.loginfo(f"MPC started at {self.mpc_rate}Hz")
        
        else:
            self.mpc_rate = 100.0  # Hz
            self.mpc_timer = QtCore.QTimer()
            self.mpc_timer.timeout.connect(self.mpc_running_callback)
            self.mpc_timer.start(int(1000/self.mpc_rate))
        
        # 在类的__init__方法中添加一个字典来存储偏移量的基准值
        # 初始化偏移量为零数组
        self.state_offset = np.zeros_like(self.state)
        
        # 创建一个映射字典，用于快速查找状态量的索引
        self.state_indices = {
            'X': 0, 'Y': 1, 'Z': 2,                    # 位置
            'roll': 3, 'pitch': 4, 'yaw': 5,          # 姿态(用于临时存储欧拉角)
            'vx': 7, 'vy': 8, 'vz': 9,                # 线速度
            'wx': 10, 'wy': 11, 'wz': 12              # 角速度
        }
        
        # 在__init__方法的最后
        # self.setMinimumSize(1200, 1500)  # 设置窗口最小大小
    
    #region --------------MPC related-------------------------------    
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
    
    #endregion
    
    #region --------------Ros interface--------------------------------
    def mpc_timer_callback_ros(self, event):
        
        self.mpc_controller.problem.x0 = self.state
        
        # print(self.mpc_ref_index)
        self.mpc_controller.updateProblem(self.mpc_ref_index)
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()

        self.solving_time = (time_end - time_start).to_sec()
        
        # 记录求解时间
        self.mpc_solve_time_history.append(self.solving_time)
        if len(self.mpc_solve_time_history) > 100:
            self.mpc_solve_time_history.pop(0)
        
        self.traj_ref_index = int(self.mpc_ref_index / self.dt_traj_opt)
        self.control_command = self.mpc_controller.solver.us_squash[0]
        
        rospy.logdebug('MPC ref index      : {}'.format(self.traj_ref_index))
        rospy.logdebug('MPC state          : {}'.format(self.state[7:]))
        rospy.logdebug('MPC reference state: {}'.format(self.traj_state_ref[self.traj_ref_index][7:]))
        rospy.logdebug('MPC control command: {}'.format(self.control_command))
        
        
        # Publish MPC data
        # self.publish_mpc_data()
        # self.publish_mavros_rate_command()
        
    def publish_mavros_rate_command(self):
        # Using MAVROS setpoint to achieve rate control
        
        
        
        # Get planned state
        self.state_ref = self.mpc_controller.solver.xs[1]
        
        self.roll_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 3]
        self.pitch_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 4]
        self.yaw_rate_ref = self.state_ref[self.mpc_controller.robot_model.nq + 5]
        
        self.total_thrust = np.sum(self.control_command)
        
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # Set type_mask to ignore attitude, only use angular velocity + thrust
        att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE 
        
        # Body frame angular velocity (rad/s)
        att_msg.body_rate = Vector3(self.roll_rate_ref, self.pitch_rate_ref, self.yaw_rate_ref)
        
        # Thrust value (range 0 ~ 1)
        max_thrust = 7.0664 * 4
        att_msg.thrust = self.total_thrust / max_thrust
        
        # Clamp thrust value
        att_msg.thrust = np.clip(att_msg.thrust, 0, 1)

        self.attitude_pub.publish(att_msg)
        
    def publish_mpc_data(self):
        # Publish MPC state
        state_msg = MpcState()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.state = self.state.tolist()
        state_msg.state_ref = self.state_ref.tolist()
        state_msg.state_error = (self.state_ref - self.state).tolist()
        
        # Fill position and attitude information
        state_msg.position.x = self.state[0]
        state_msg.position.y = self.state[1]
        state_msg.position.z = self.state[2]
        state_msg.orientation.x = self.state[3]
        state_msg.orientation.y = self.state[4]
        state_msg.orientation.z = self.state[5]
        state_msg.orientation.w = self.state[6]
        
        # Publish control input
        control_msg = MpcControl()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.control_raw = self.mpc_controller.solver.us[0].tolist()
        control_msg.control_squash = self.mpc_controller.solver.us_squash[0].tolist()
        control_msg.thrust_command = self.mpc_controller.solver.xs[0].tolist()
        control_msg.speed_command = self.mpc_controller.solver.xs[1].tolist()
        
        # Publish messages
        self.mpc_state_pub.publish(state_msg)
        self.mpc_control_pub.publish(control_msg)
        self.solving_time_pub.publish(Float64(self.solving_time))
        
    def state_callback(self, msg):
        self.state_history.append(msg)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
            
        # Update slider positions to match current state
        if self.state_history:
            current_state = self.state_history[-1].state
            for i, axis in enumerate(['X', 'Y', 'Z']):
                self.state_sliders[axis].blockSignals(True)  # Prevent callback triggering
                self.state_sliders[axis].setValue(int(current_state[i] * 100))
                self.state_labels[axis].setText(f'{current_state[i]:.2f}')
                self.state_sliders[axis].blockSignals(False)
            
    def control_callback(self, msg):
        self.control_history.append(msg)
        if len(self.control_history) > 100:
            self.control_history.pop(0)

    def state_changed_ros(self, axis, value):
        # 发布新的位置
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"
        if axis == 'X':
            pose.pose.position.x = value
            self.state[0] = value
        elif axis == 'Y':
            pose.pose.position.y = value
            self.state[1] = value
        elif axis == 'Z':
            pose.pose.position.z = value
            self.state[2] = value
        
        self.pose_pub.publish(pose)
        
        # 更新显示的值
        self.state_labels[axis].setText(f'{value:.2f}')

    def mav_state_callback(self, msg):
        self.current_state = msg
        
    def arm_state_callback(self, msg):
        self.arm_state = msg
        
        # update self.state
        self.state[7:9] = [msg.position[1], msg.position[0]] # TODO: don't know why it is reversed
        self.state[-2:] = [msg.velocity[1], msg.velocity[0]]
        
        max_buffer_size = 100
        self.joint_position_buffer.append(self.state[7:9])
        if len(self.joint_position_buffer) > max_buffer_size:
            self.joint_position_buffer.pop(0)  # 删除最旧的值
            
        self.joint_velocity_buffer.append(self.state[-2:])
        if len(self.joint_velocity_buffer) > max_buffer_size:
            self.joint_velocity_buffer.pop(0)  # 删除最旧的值
            
        self.joint_effort_buffer.append([msg.effort[1], msg.effort[0]])
        if len(self.joint_effort_buffer) > max_buffer_size:
            self.joint_effort_buffer.pop(0)  # 删除最旧的值
            
        self.joint_control_buffer.append(self.control_command[-2:])
        if len(self.joint_control_buffer) > max_buffer_size:
            self.joint_control_buffer.pop(0)
    #endregion

    #region --------------QT without ROS--------------------------------
    def create_state_slider(self, name, min_val, max_val):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        
        # 标题标签
        title_label = QLabel(name)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFixedWidth(80)
        title_label.setStyleSheet("""
            QLabel {
                color: darkblue;
                font-weight: bold;
                padding: 2px;
            }
        """)
        layout.addWidget(title_label, 0, QtCore.Qt.AlignCenter)
        
        # 滑块
        slider = QSlider(QtCore.Qt.Vertical)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(0)
        slider.setFixedHeight(150)
        slider.setFixedWidth(30)
        slider.setStyleSheet("""
            QSlider::groove:vertical {
                background: #e0e0e0;
                width: 10px;
                border-radius: 5px;
            }
            QSlider::handle:vertical {
                background: #4a90e2;
                border: 1px solid #5c5c5c;
                height: 18px;
                margin: 0 -4px;
                border-radius: 9px;
            }
        """)
        slider.valueChanged.connect(lambda v: self.slider_value_changed(name, v))
        layout.addWidget(slider, 0, QtCore.Qt.AlignCenter)
        
        # 值输入框
        value_input = QtWidgets.QLineEdit('0.00')
        value_input.setAlignment(QtCore.Qt.AlignCenter)
        value_input.setFixedWidth(100)
        value_input.setStyleSheet("""
            QLineEdit {
                color: #333;
                font-family: monospace;
                padding: 3px;
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
                background: white;
            }
        """)
        # 添加输入验证器
        validator = QtGui.QDoubleValidator(min_val, max_val, 2)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        value_input.setValidator(validator)
        # 连接输入框的信号
        value_input.editingFinished.connect(lambda: self.value_input_changed(name))
        layout.addWidget(value_input, 0, QtCore.Qt.AlignCenter)
        
        # 存储滑块和标签的引用
        self.state_sliders[name] = slider
        self.state_labels[name] = value_input
        
        return layout
    
    def slider_value_changed(self, name, value):
        """处理滑块值变化"""
        actual_value = value / 100.0
        
        # 获取当前时间对应的参考状态
        time_index = int(self.mpc_ref_time / self.dt_traj_opt)
        time_index = min(time_index, len(self.state_ref) - 1)
        ref_state = self.state_ref[time_index]
        
        # 更新偏移量和状态
        if name in ['X', 'Y', 'Z']:
            idx = self.state_indices[name]
            self.state_offset[idx] = actual_value
            self.state[idx] = ref_state[idx] + actual_value
            
        elif name in ['roll', 'pitch', 'yaw']:
            idx = self.state_indices[name]
            # 存储欧拉角偏移量
            self.state_offset[idx] = actual_value
            # 计算新的姿态
            euler_ref = R.from_quat(ref_state[3:7]).as_euler('xyz', degrees=True)
            euler_new = euler_ref.copy()
            euler_new[idx-3] = euler_ref[idx-3] + actual_value
            self.state[3:7] = R.from_euler('xyz', euler_new, degrees=True).as_quat()
            
        elif name in ['vx', 'vy', 'vz']:
            idx = self.state_indices[name]
            self.state_offset[idx] = actual_value
            self.state[idx] = ref_state[idx] + actual_value
            
        elif name in ['wx', 'wy', 'wz']:
            idx = self.state_indices[name]
            self.state_offset[idx] = np.radians(actual_value)
            self.state[idx] = ref_state[idx] + np.radians(actual_value)
        
        # 更新显示值
        self.state_labels[name].blockSignals(True)
        self.state_labels[name].setText(f'{actual_value:.2f}')
        self.state_labels[name].blockSignals(False)

    def value_input_changed(self, name):
        """处理输入框值变化"""
        try:
            # 获取输入框的值
            text_value = self.state_labels[name].text()
            value = float(text_value)
            # 更新滑块，但不触发滑块的信号
            self.state_sliders[name].blockSignals(True)
            self.state_sliders[name].setValue(int(value * 100))
            self.state_sliders[name].blockSignals(False)
            # 更新状态
            self.state_changed_input(name, value)
        except ValueError:
            # 如果输入无效，恢复到滑块的值
            slider_value = self.state_sliders[name].value() / 100.0
            self.state_labels[name].setText(f'{slider_value:.2f}')

    def state_changed_input(self, axis, value, is_offset=False):
        print("state changed: ", axis, value)
        
        # 获取当前四元数
        current_quat = self.state[3:7]
        # 转换为欧拉角
        euler = R.from_quat(current_quat).as_euler('xyz', degrees=True)
        
        if is_offset:
            # 使用偏移量更新状态
            if 'X' in axis:
                self.state[0] += value
            elif 'Y' in axis:
                self.state[1] += value
            elif 'Z' in axis:
                self.state[2] += value
            elif 'roll' in axis:
                euler[0] += value
                quat = R.from_euler('xyz', euler, degrees=True).as_quat()
                self.state[3:7] = quat
            elif 'pitch' in axis:
                euler[1] += value
                quat = R.from_euler('xyz', euler, degrees=True).as_quat()
                self.state[3:7] = quat
            elif 'yaw' in axis:
                euler[2] += value
                quat = R.from_euler('xyz', euler, degrees=True).as_quat()
                self.state[3:7] = quat
            elif 'vx' in axis:
                self.state[7] += value
            elif 'vy' in axis:
                self.state[8] += value
            elif 'vz' in axis:
                self.state[9] += value
            elif 'wx' in axis:
                self.state[10] += np.radians(value)
            elif 'wy' in axis:
                self.state[11] += np.radians(value)
            elif 'wz' in axis:
                self.state[12] += np.radians(value)
        else:
            # 原来的绝对值更新逻辑
            if 'X' in axis:
                self.state[0] = value
            elif 'Y' in axis:
                self.state[1] = value
            elif 'Z' in axis:
                self.state[2] = value
            elif 'roll' in axis:
                euler[0] = value  # value已经是度数
                quat = R.from_euler('xyz', euler, degrees=True).as_quat()
                self.state[3:7] = quat
            elif 'pitch' in axis:
                euler[1] = value  # value已经是度数
                quat = R.from_euler('xyz', euler, degrees=True).as_quat()
                self.state[3:7] = quat
            elif 'yaw' in axis:
                euler[2] = value  # value已经是度数
                quat = R.from_euler('xyz', euler, degrees=True).as_quat()
                self.state[3:7] = quat
            elif 'vx' in axis:
                self.state[7] = value
            elif 'vy' in axis:
                self.state[8] = value
            elif 'vz' in axis:
                self.state[9] = value
            elif 'wx' in axis:
                self.state[10] = np.radians(value)  # 转换为弧度
            elif 'wy' in axis:
                self.state[11] = np.radians(value)  # 转换为弧度
            elif 'wz' in axis:
                self.state[12] = np.radians(value)  # 转换为弧度
        
    def time_changed(self, value):
        
        self.time_label.setText(f'{value} ms')
        self.mpc_ref_time = int(value)   # 单位为ms
        
        # 重置所有滑块到零位置，这样偏移量会重置
        # self.reset_sliders()
        self.set_to_reference()

    def mpc_running_callback(self):
        # calculate the mpc output
        self.mpc_controller.problem.x0 = self.state
        # print(self.mpc_ref_index, len(self.state_ref))
        self.mpc_controller.updateProblem(self.mpc_ref_time)
        
        time_start = time.time()

        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
         
        time_end = time.time()
        self.solving_time = time_end - time_start
        
        # 记录求解时间
        self.mpc_solve_time_history.append(self.solving_time)
        if len(self.mpc_solve_time_history) > 100:
            self.mpc_solve_time_history.pop(0)
        
        print("solving time: {:.2f} ms".format(self.solving_time * 1000))
        print("state: ", self.state)
        
        state_predict = np.array(self.mpc_controller.solver.xs)   # 50, 13
        print('body rate current0: ', state_predict[0][10:13])
        print('body rate command1: ', state_predict[1][10:13])
        print('body rate command2: ', state_predict[2][10:13])

    #endregion

    #region --------------plot--------------------------------  
    def update_plot(self):
        # 更新状态图
        state_predict = np.array(self.mpc_controller.solver.xs)
        state_ref = np.array(self.mpc_controller.state_ref)
        self.update_state_plot(state_predict, state_ref)
        self.update_attitude_plot(state_predict, state_ref)
        self.update_linear_velocity_plot(state_predict, state_ref)
        self.update_angular_velocity_plot(state_predict, state_ref)
        
        # update control plot
        control_predict = np.array(self.mpc_controller.solver.us_squash)
        control_ref = np.array(self.traj_solver.us_squash)
        self.update_control_plot(control_predict, control_ref)
        
        if self.plot_thrust_torque:
            self.update_thrust_torque_plot(control_predict, control_ref)
        
        # update solving time plot
        self.update_solving_time_plot()
        
        # update arm plot
        self.update_arm_plot(state_predict, state_ref, control_predict, control_ref)
        
    def update_arm_plot(self, state_predict, state_ref, control_predict=None, control_ref=None):
        if self.arm_enabled:
            joint_position = np.array(self.joint_position_buffer)
            joint_velocity = np.array(self.joint_velocity_buffer)
            joint_effort = np.array(self.joint_effort_buffer)
            joint_control = np.array(self.joint_control_buffer)
            
            self.update_joint_history_plot(self.ax_joint_position, 'Joint Position', joint_position, 1.5, 'Position (rad)', state_predict[:, 7:9], state_ref[:, 7:9])
            self.update_joint_history_plot(self.ax_joint_velocity, 'Joint Velocity', joint_velocity, 4.0, 'Velocity (rad/s)', state_predict[:, -2:], state_ref[:, -2: ])
            self.update_joint_history_plot(self.ax_joint_effort,   'Joint Effort',   joint_effort,   2.0, 'Effort (Nm)')
            
            self.update_joint_history_plot(self.ax_joint_control,  'Joint Control',  None,  0.3, 'Control (Nm)', control_predict[:, -2:], control_ref[:, -2:])
            
        
    def update_joint_history_plot(self, ax, title, data, y_lim, y_label, state_predict=None, state_ref=None):
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(y_label) 
        
        if data is not None:
            freq_state_update = 62 # 62Hz
            time_history = np.arange(-len(data), 0) / freq_state_update 
            ax.plot(time_history, data[:,0], label='Joint_1', color='black')
            ax.plot(time_history, data[:,1], label='Joint_2', color='yellow')
        
        # plot predicted joint position
        if state_predict is not None:
            predict_start_time = self.mpc_ref_time / 1000.0
            predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
            ax.plot(predict_time, state_predict[:, 0], label='Joint_1_predict', color='red', linewidth=2)
            ax.plot(predict_time, state_predict[:, 1], label='Joint_2_predict', color='green', linewidth=2)
        
        # plot reference joint position
        if state_ref is not None:
            ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
            ax.plot(ref_time, state_ref[:, 0], label='Joint_1_ref', linestyle='--', color='red', linewidth=1)
            ax.plot(ref_time, state_ref[:, 1], label='Joint_2_ref', linestyle='--', color='green', linewidth=1)
        
        
        ax.legend()
        ax.set_ylim(-y_lim, y_lim)
        
    def update_state_plot(self, state_predict, state_ref):
        self.ax_state.clear()
        
        self.ax_state.set_title('Position')
        self.ax_state.set_xlabel('Time (s)')
        self.ax_state.set_ylabel('Position (m)')
        
        # 计算参考轨迹时间轴 (使用轨迹优化的dt)
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        
        # 计算预测轨迹时间轴 (使用MPC的dt)
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        # 预测状态线条更粗 (linewidth=2)
        self.ax_state.plot(predict_time, state_predict[:, 0], label='X', color='red', linewidth=2)
        self.ax_state.plot(predict_time, state_predict[:, 1], label='Y', color='green', linewidth=2)
        self.ax_state.plot(predict_time, state_predict[:, 2], label='Z', color='blue', linewidth=2)
        
        # 参考状态线条较细 (linewidth=1)
        self.ax_state.plot(ref_time, state_ref[:, 0], label='X_ref', linestyle='--', color='red', linewidth=1)
        self.ax_state.plot(ref_time, state_ref[:, 1], label='Y_ref', linestyle='--', color='green', linewidth=1)
        self.ax_state.plot(ref_time, state_ref[:, 2], label='Z_ref', linestyle='--', color='blue', linewidth=1)
        
        self.ax_state.legend()

    def update_attitude_plot(self, state_predict, state_ref):
        self.ax_attitude.clear()
        self.ax_attitude.set_title('Attitude')
        self.ax_attitude.set_xlabel('Time (s)')
        self.ax_attitude.set_ylabel('Angle (deg)')
        
        euler_predict = np.array([R.from_quat(q).as_euler('xyz', degrees=True) 
                                 for q in state_predict[:, 3:7]])
        euler_ref = np.array([R.from_quat(q).as_euler('xyz', degrees=True) 
                             for q in state_ref[:, 3:7]])
        
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        # 预测状态线条更粗
        self.ax_attitude.plot(predict_time, euler_predict[:, 0], label='Roll', color='red', linewidth=2)
        self.ax_attitude.plot(predict_time, euler_predict[:, 1], label='Pitch', color='green', linewidth=2)
        self.ax_attitude.plot(predict_time, euler_predict[:, 2], label='Yaw', color='blue', linewidth=2)
        
        # 参考状态线条较细
        self.ax_attitude.plot(ref_time, euler_ref[:, 0], label='Roll SP', linestyle='--', color='red', linewidth=1)
        self.ax_attitude.plot(ref_time, euler_ref[:, 1], label='Pitch SP', linestyle='--', color='green', linewidth=1)
        self.ax_attitude.plot(ref_time, euler_ref[:, 2], label='Yaw SP', linestyle='--', color='blue', linewidth=1)
        
        self.ax_attitude.legend()

    def update_linear_velocity_plot(self, state_predict, state_ref):
        self.ax_linear_velocity.clear()
        self.ax_linear_velocity.set_title('Linear Velocity (body frame)')
        self.ax_linear_velocity.set_xlabel('Time (s)')
        self.ax_linear_velocity.set_ylabel('Velocity (m/s)')
        
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        index_plot = 7
        
        # 预测状态线条更粗
        self.ax_linear_velocity.plot(predict_time, state_predict[:, index_plot], label='X', color='red', linewidth=2)
        self.ax_linear_velocity.plot(predict_time, state_predict[:, index_plot+1], label='Y', color='green', linewidth=2)
        self.ax_linear_velocity.plot(predict_time, state_predict[:, index_plot+2], label='Z', color='blue', linewidth=2)
        
        # 参考状态线条较细
        self.ax_linear_velocity.plot(ref_time, state_ref[:, index_plot], label='X_ref', linestyle='--', color='red', linewidth=1)
        self.ax_linear_velocity.plot(ref_time, state_ref[:, index_plot+1], label='Y_ref', linestyle='--', color='green', linewidth=1)
        self.ax_linear_velocity.plot(ref_time, state_ref[:, index_plot+2], label='Z_ref', linestyle='--', color='blue', linewidth=1)
        
        self.ax_linear_velocity.legend()

    def update_angular_velocity_plot(self, state_predict, state_ref):
        self.ax_angular_velocity.clear()
        self.ax_angular_velocity.set_title('Angular Velocity (body frame)')
        self.ax_angular_velocity.set_xlabel('Time (s)')
        self.ax_angular_velocity.set_ylabel('Velocity (rad/s)')
        
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        index_plot = 10
        
        # 预测状态线条更粗
        self.ax_angular_velocity.plot(predict_time, state_predict[:, index_plot], label='X', color='red', linewidth=2)
        self.ax_angular_velocity.plot(predict_time, state_predict[:, index_plot+1], label='Y', color='green', linewidth=2)
        self.ax_angular_velocity.plot(predict_time, state_predict[:, index_plot+2], label='Z', color='blue', linewidth=2)
        
        # 参考状态线条较细
        self.ax_angular_velocity.plot(ref_time, state_ref[:, index_plot], label='X_ref', linestyle='--', color='red', linewidth=1)
        self.ax_angular_velocity.plot(ref_time, state_ref[:, index_plot+1], label='Y_ref', linestyle='--', color='green', linewidth=1)
        self.ax_angular_velocity.plot(ref_time, state_ref[:, index_plot+2], label='Z_ref', linestyle='--', color='blue', linewidth=1)
        
        self.ax_angular_velocity.legend()

    def update_control_plot(self, control_predict, control_ref):
        self.ax_control.clear()
        self.ax_control.set_title('Control')
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.set_ylabel('Control')
        
        ref_time = np.arange(len(control_ref)) * self.dt_traj_opt / 1000.0
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(control_predict)) * self.dt_mpc / 1000.0
        
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        control_num = 4
        for i in range(control_num):
            # 预测控制线条更粗
            self.ax_control.plot(predict_time, control_predict[:, i], 
                               label='Control_{}'.format(i), color=colors[i], linewidth=2)
            # 参考控制线条较细
            self.ax_control.plot(ref_time, control_ref[:, i],
                               label='Control_ref_{}'.format(i), linestyle='--', 
                               color=colors[i], linewidth=1)
        
        self.ax_control.legend()
        
    def update_solving_time_plot(self):
        self.ax_time.clear()
        self.ax_time.set_title('MPC Solve Time')
        self.ax_time.set_xlabel('Iteration')
        self.ax_time.set_ylabel('Time (ms)')
        
        if self.mpc_solve_time_history:
            times = range(len(self.mpc_solve_time_history))
            self.ax_time.plot(times, [t*1000 for t in self.mpc_solve_time_history], 'b-', label='Solve Time')
            self.ax_time.axhline(y=np.mean([t*1000 for t in self.mpc_solve_time_history]), color='r', linestyle='--', label='Mean')
            self.ax_time.legend()
        
        # 自动调整子图布局
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_thrust_torque_plot(self, control_predict, control_ref):
        self.ax_thrust.clear()
        self.ax_thrust.set_title('Thrust')
        self.ax_thrust.set_xlabel('Time (s)')
        self.ax_thrust.set_ylabel('Thrust (N)')
        
        self.ax_torque.clear()
        self.ax_torque.set_title('Torque')
        self.ax_torque.set_xlabel('Time (s)')
        self.ax_torque.set_ylabel('Torque (Nm)')
        
        # 计算参考轨迹时间轴 (使用轨迹优化的dt)
        ref_time = np.arange(len(control_ref)) * self.dt_traj_opt / 1000.0
        
        # 计算预测轨迹时间轴 (使用MPC的dt)
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(control_predict)) * self.dt_mpc / 1000.0
        
        # get tau_f
        tau_f = self.mpc_controller.platform_params.tau_f
        
        # 计算predict的总推力
        total_thrust_torque_predict = thrustToForceTorqueAll_array(control_predict, tau_f)
        # 计算ref的总推力
        total_thrust_torque_ref = thrustToForceTorqueAll_array(control_ref, tau_f)
        
        # 绘制推力
        self.ax_thrust.plot(predict_time, total_thrust_torque_predict[:, 2], label='Thrust', color='red')
        self.ax_thrust.plot(ref_time, total_thrust_torque_ref[:, 2], label='Thrust_ref', linestyle='--', color='red')
        
        # 绘制扭矩
        colors = ['red', 'green', 'blue']
        for i in range(3):
            self.ax_torque.plot(predict_time, total_thrust_torque_predict[:, i+3], label='Torque_{}'.format(i), color=colors[i])
            self.ax_torque.plot(ref_time, total_thrust_torque_ref[:, i+3], label='Torque_ref_{}'.format(i), linestyle='--', color=colors[i])
        
        self.ax_thrust.legend()
        self.ax_torque.legend()
        
        
        
        
        # 计算predict的总扭矩
        
    
    #endregion
    
    def set_to_reference_button_clicked(self):
        
        time_index = int(self.mpc_ref_time / self.dt_traj_opt)
        time_index = min(time_index, len(self.state_ref) - 1)
        ref_state = self.state_ref[time_index]
        # Update state
        self.state = np.copy(ref_state)
        
        # Update all sliders and labels to match reference state
        # # Position
        # for i, axis in enumerate(['X', 'Y', 'Z']):
        #     self.state_sliders[axis].blockSignals(True)
        #     self.state_sliders[axis].setValue(int(ref_state[i] * 100))
        #     self.state_labels[axis].setText(f'{ref_state[i]:.2f}')
        #     self.state_sliders[axis].blockSignals(False)
            
        # # Orientation (convert quaternion to euler angles)
        # euler = R.from_quat(ref_state[3:7]).as_euler('xyz', degrees=True)
        # for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        #     self.state_sliders[axis].blockSignals(True)
        #     self.state_sliders[axis].setValue(int(euler[i]))
        #     self.state_labels[axis].setText(f'{euler[i]:.2f}')
        #     self.state_sliders[axis].blockSignals(False)
            
        # # Linear velocity
        # for i, axis in enumerate(['vx', 'vy', 'vz']):
        #     self.state_sliders[axis].blockSignals(True)
        #     self.state_sliders[axis].setValue(int(ref_state[i + 7] * 100))
        #     self.state_labels[axis].setText(f'{ref_state[i + 7]:.2f}')
        #     self.state_sliders[axis].blockSignals(False)
            
        # # Angular velocity (convert to degrees)
        # for i, axis in enumerate(['wx', 'wy', 'wz']):
        #     ang_vel_deg = np.degrees(ref_state[i + 10])
        #     self.state_sliders[axis].blockSignals(True)
        #     self.state_sliders[axis].setValue(int(ang_vel_deg))
        #     self.state_labels[axis].setText(f'{ang_vel_deg:.2f}')
        #     self.state_sliders[axis].blockSignals(False)
        
        self.reset_sliders()
        
    def set_to_reference(self):
        selected = self.ref_selector.currentText()
        
        if selected == 'Initial State':
            ref_state = self.state_ref[0]
        elif selected == 'Final State':
            ref_state = self.state_ref[-1]
        elif selected == 'Current offset':
            # 获取当前参考状态
            time_index = int(self.mpc_ref_time / self.dt_traj_opt)
            time_index = min(time_index, len(self.state_ref) - 1)
            ref_state = self.state_ref[time_index]
            
            # 应用偏移量
            new_state = ref_state.copy()
            # 位置和速度直接相加
            for i in [0,1,2,7,8,9,10,11,12]:
                new_state[i] = ref_state[i] + self.state_offset[i]
            # 姿态需要特殊处理
            euler_ref = R.from_quat(ref_state[3:7]).as_euler('xyz', degrees=True)
            euler_new = euler_ref + self.state_offset[3:6]  # 3:6存储的是欧拉角偏移量
            new_state[3:7] = R.from_euler('xyz', euler_new, degrees=True).as_quat()
            
            ref_state = new_state
            
        elif selected == 'Current Reference':
            time_index = int(self.mpc_ref_time / self.dt_traj_opt)
            time_index = min(time_index, len(self.state_ref) - 1)
            ref_state = self.state_ref[time_index]
            
            self.reset_sliders()
        
        # 更新状态
        self.state = np.copy(ref_state)

    def reset_sliders(self):
        """重置所有滑块和偏移量到零位置"""
        # 重置偏移量数组
        self.state_offset = np.zeros_like(self.state)
        
        # 重置所有滑块到零位置
        for axis in self.state_sliders:
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(0)
            self.state_labels[axis].setText('0.00')
            self.state_sliders[axis].blockSignals(False)

    # 添加更新质量的方法
    def update_mass(self):
        """更新baselink的质量"""
        try:
            new_mass = float(self.mass_input.text())
            if new_mass <= 0:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", "Mass must be positive!")
                return
                
            # 更新MPC控制器中的质量参数
            self.mpc_controller.platform_params.mass = new_mass
            
            # 更新显示的当前质量
            self.current_mass_label.setText(f"Current Mass: {new_mass:.3f} kg")
            
            # 清空输入框
            self.mass_input.clear()
            
            # 可选：显示成功消息
            QtWidgets.QMessageBox.information(self, "Success", f"Mass updated to {new_mass:.3f} kg")
            
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid number!")


if __name__ == '__main__':
    import sys
    
    # Settings
    using_ros = False
    mpc_name = 'rail'
    mpc_yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml'
    
    robot_name = 's500_uam'
    trajectory_name = 'arm_test'
    dt_traj_opt = 50  # ms
    useSquash = True
    
    if using_ros:
        rospy.init_node('mpc_debug_interface', anonymous=False, log_level=rospy.DEBUG)
    
    app = QtWidgets.QApplication(sys.argv)
    window = MpcDebugInterface(using_ros=using_ros, mpc_name=mpc_name, mpc_yaml_path=mpc_yaml_path, robot_name=robot_name, trajectory_name=trajectory_name, dt_traj_opt=dt_traj_opt, useSquash=useSquash)
    window.show()
    
    sys.exit(app.exec_()) 