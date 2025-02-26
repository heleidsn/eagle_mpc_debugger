'''
Author: Lei He
Date: 2025-02-19 11:40:31
LastEditTime: 2025-02-26 14:44:10
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

from utils.create_problem import get_opt_traj, create_mpc_controller

from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import Vector3
from scipy.spatial.transform import Rotation as R

class MpcDebugInterface(QWidget):
    def __init__(self, using_ros=False, mpc_name='rail', mpc_yaml_path=None, robot_name='iris', trajectory_name='hover', dt_traj_opt=20, useSquash=True):
        super(MpcDebugInterface, self).__init__()
        self.setWindowTitle('MPC Debug Interface')
        
        self.using_ros = using_ros
        self.mpc_name = mpc_name
        self.mpc_yaml_path = mpc_yaml_path
        self.robot_name = robot_name
        self.trajectory_name = trajectory_name
        self.useSquash = useSquash
        self.dt_traj_opt = dt_traj_opt  # ms
        
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
        self.ref_selector.addItems(['Current State', 'Initial State', 'Final State', 'Current Reference'])
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
            ('Z', 0, 4),
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
        self.figure = Figure(figsize=(20, 10))
        # Set spacing between subplots
        self.figure.subplots_adjust(
            left=0.08,    # Left margin
            right=0.95,   # Right margin
            bottom=0.08,  # Bottom margin
            top=0.95,     # Top margin
            wspace=0.25,  # Horizontal spacing between subplots
            hspace=0.35   # Vertical spacing between subplots
        )
        self.canvas = FigureCanvas(self.figure)
        plot_row_num = 3
        plot_col_num = 2
        self.ax_state = self.figure.add_subplot(plot_row_num, plot_col_num, 1)  # State
        self.ax_attitude = self.figure.add_subplot(plot_row_num, plot_col_num, 2)  # Attitude
        self.ax_linear_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 3)  # Linear velocity
        self.ax_angular_velocity = self.figure.add_subplot(plot_row_num, plot_col_num, 4)  # Angular velocity
        
        self.ax_control = self.figure.add_subplot(plot_row_num, plot_col_num, 5)  # Control
        self.ax_time = self.figure.add_subplot(plot_row_num, plot_col_num, 6)   # Solving time
        
        # Add to main layout
        self.layout.addLayout(time_layout)
        self.layout.addLayout(state_layout)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
      
        # Data storage
        self.state_history = []
        self.control_history = []
        self.mpc_solve_time_history = []  # Store solving time history
        
        # Timer for updating plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # 10Hz update
        
        # Initialize MPC
        traj_solver, traj_state_ref, traj_problem, trajectory_obj = get_opt_traj(
            self.robot_name, 
            self.trajectory_name, 
            self.dt_traj_opt, 
            self.useSquash,
            self.mpc_yaml_path)
        
        self.traj_solver = traj_solver
        
        # Create MPC controller to get tau_f
        mpc_name = self.mpc_name
        mpc_yaml = '{}/mpc/{}_mpc.yaml'.format(self.mpc_yaml_path, self.robot_name)
        self.mpc_controller = create_mpc_controller(
            mpc_name,
            trajectory_obj,
            traj_state_ref,
            self.dt_traj_opt,
            mpc_yaml
        )
        
        self.mpc_controller.solver.setCallbacks([])  # Disable callback outputs
        
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
            self.time_pub = rospy.Publisher('/debug/time', Float64, queue_size=1)
            
            self.state_sub = rospy.Subscriber('/debug/mpc_state', MpcState, self.state_callback)
            self.control_sub = rospy.Subscriber('/debug/mpc_control', MpcControl, self.control_callback)
            
            self.mpc_state_pub = rospy.Publisher('/mpc/state', MpcState, queue_size=10)
            self.mpc_control_pub = rospy.Publisher('/mpc/control', MpcControl, queue_size=10)
            
            self.solving_time_pub = rospy.Publisher('/mpc/solving_time', Float64, queue_size=1)
            
            self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
            
            # Start MPC controller and wait for it to start
            self.mpc_rate = 10.0  # Hz
            self.mpc_timer = rospy.Timer(rospy.Duration(1.0/self.mpc_rate), self.mpc_timer_callback_ros)
            
            rospy.loginfo(f"MPC started at {self.mpc_rate}Hz")
        
        else:
            self.mpc_rate = 10.0  # Hz
            self.mpc_timer = QtCore.QTimer()
            self.mpc_timer.timeout.connect(self.mpc_running_callback)
            self.mpc_timer.start(int(1000/self.mpc_rate))
            
    #region --------------Ros interface--------------------------------
    def mpc_timer_callback_ros(self, event):
        
        self.mpc_controller.problem.x0 = self.state
        
        print(self.mpc_ref_index)
        self.mpc_controller.updateProblem(self.mpc_ref_index)
        
        time_start = rospy.Time.now()
        self.mpc_controller.solver.solve(
            self.mpc_controller.solver.xs,
            self.mpc_controller.solver.us,
            self.mpc_controller.iters
        )
        time_end = rospy.Time.now()

        self.solving_time = (time_end - time_start).to_sec()
        
        # Publish MPC data
        self.publish_mpc_data()
        self.publish_mavros_rate_command()
        
    def publish_mavros_rate_command(self):
        # Using MAVROS setpoint to achieve rate control
        
        self.control_command = self.mpc_controller.solver.us_squash[0]
        
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
        # 更新输入框，但不触发输入框的信号
        self.state_labels[name].blockSignals(True)
        self.state_labels[name].setText(f'{actual_value:.2f}')
        self.state_labels[name].blockSignals(False)
        # 更新状态
        self.state_changed(name, actual_value)
        
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
            self.state_changed(name, value)
        except ValueError:
            # 如果输入无效，恢复到滑块的值
            slider_value = self.state_sliders[name].value() / 100.0
            self.state_labels[name].setText(f'{slider_value:.2f}')

    def state_changed(self, axis, value):
        print("state changed: ", axis, value)
        
        # 获取当前四元数
        current_quat = self.state[3:7]
        # 转换为欧拉角
        euler = R.from_quat(current_quat).as_euler('xyz', degrees=True)
        
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
        # 发布新的时间戳
        
        if self.using_ros:
            self.time_pub.publish(Float64(value))
        
        self.time_label.setText(f'{value} ms')
        self.mpc_ref_time = int(value)   # 单位为ms
        
        # 发布新的参考轨迹
        self.set_to_reference()
        
        # self.mpc_ref_index = int(value / self.dt_traj_opt)
        
        # # contstrain the time to be within the range of the trajectory
        # print(self.mpc_ref_index, len(self.state_ref))
        
        # if self.mpc_ref_index < 0:
        #     self.mpc_ref_index = 0
        # elif self.mpc_ref_index > len(self.state_ref):
        #     self.mpc_ref_index = len(self.state_ref)
            
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
        
        # update solving time plot
        self.update_solving_time_plot()
        
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
        
        self.ax_state.plot(predict_time,
                         state_predict[:, 0], label='X', color='red')
        self.ax_state.plot(predict_time,
                         state_predict[:, 1], label='Y', color='green')
        self.ax_state.plot(predict_time,
                         state_predict[:, 2], label='Z', color='blue')
        
        self.ax_state.plot(ref_time, state_ref[:, 0], label='X_ref', linestyle='--', color='red')
        self.ax_state.plot(ref_time, state_ref[:, 1], label='Y_ref', linestyle='--', color='green')
        self.ax_state.plot(ref_time, state_ref[:, 2], label='Z_ref', linestyle='--', color='blue')
        
        self.ax_state.legend()

    def update_attitude_plot(self, state_predict, state_ref):
        # 更新姿态图
        self.ax_attitude.clear()
        self.ax_attitude.set_title('Attitude')
        self.ax_attitude.set_xlabel('Time (s)')
        self.ax_attitude.set_ylabel('Angle (deg)')
        
        # 将预测状态和参考状态的四元数转换为欧拉角
        euler_predict = np.array([R.from_quat(q).as_euler('xyz', degrees=True) 
                                 for q in state_predict[:, 3:7]])
        euler_ref = np.array([R.from_quat(q).as_euler('xyz', degrees=True) 
                             for q in state_ref[:, 3:7]])
        
        # 计算参考轨迹时间轴 (使用轨迹优化的dt)
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        
        # 计算预测轨迹时间轴 (使用MPC的dt)
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        # 绘制欧拉角
        self.ax_attitude.plot(predict_time,
                            euler_predict[:, 0], label='Roll', color='red')
        self.ax_attitude.plot(predict_time,
                            euler_predict[:, 1], label='Pitch', color='green')
        self.ax_attitude.plot(predict_time,
                            euler_predict[:, 2], label='Yaw', color='blue')
        
        self.ax_attitude.plot(ref_time, euler_ref[:, 0], label='Roll SP', linestyle='--', color='red')
        self.ax_attitude.plot(ref_time, euler_ref[:, 1], label='Pitch SP', linestyle='--', color='green')
        self.ax_attitude.plot(ref_time, euler_ref[:, 2], label='Yaw SP', linestyle='--', color='blue')
        
        self.ax_attitude.legend()
        
    def update_linear_velocity_plot(self, state_predict, state_ref):
        self.ax_linear_velocity.clear()
        self.ax_linear_velocity.set_title('Linear Velocity (body frame)')
        self.ax_linear_velocity.set_xlabel('Time (s)')
        self.ax_linear_velocity.set_ylabel('Velocity (m/s)')
        
        # 计算参考轨迹时间轴 (使用轨迹优化的dt)
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        
        # 计算预测轨迹时间轴 (使用MPC的dt)
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        index_plot = 7
        
        self.ax_linear_velocity.plot(predict_time, state_predict[:, index_plot], label='X', color='red')
        self.ax_linear_velocity.plot(predict_time, state_predict[:, index_plot+1], label='Y', color='green')
        self.ax_linear_velocity.plot(predict_time, state_predict[:, index_plot+2], label='Z', color='blue')
        
        self.ax_linear_velocity.plot(ref_time, state_ref[:, index_plot], label='X_ref', linestyle='--', color='red')
        self.ax_linear_velocity.plot(ref_time, state_ref[:, index_plot+1], label='Y_ref', linestyle='--', color='green')
        self.ax_linear_velocity.plot(ref_time, state_ref[:, index_plot+2], label='Z_ref', linestyle='--', color='blue')
        
        self.ax_linear_velocity.legend()
        
    def update_angular_velocity_plot(self, state_predict, state_ref):
        self.ax_angular_velocity.clear()
        self.ax_angular_velocity.set_title('Angular Velocity (body frame)')
        self.ax_angular_velocity.set_xlabel('Time (s)')
        self.ax_angular_velocity.set_ylabel('Velocity (rad/s)')
        
        # 计算参考轨迹时间轴 (使用轨迹优化的dt)
        ref_time = np.arange(len(state_ref)) * self.dt_traj_opt / 1000.0
        
        # 计算预测轨迹时间轴 (使用MPC的dt)
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(state_predict)) * self.dt_mpc / 1000.0
        
        index_plot = 10
        
        self.ax_angular_velocity.plot(predict_time, state_predict[:, index_plot], label='X', color='red')
        self.ax_angular_velocity.plot(predict_time, state_predict[:, index_plot+1], label='Y', color='green')
        self.ax_angular_velocity.plot(predict_time, state_predict[:, index_plot+2], label='Z', color='blue')
        
        self.ax_angular_velocity.plot(ref_time, state_ref[:, index_plot], label='X_ref', linestyle='--', color='red')
        self.ax_angular_velocity.plot(ref_time, state_ref[:, index_plot+1], label='Y_ref', linestyle='--', color='green')
        self.ax_angular_velocity.plot(ref_time, state_ref[:, index_plot+2], label='Z_ref', linestyle='--', color='blue')
        
        self.ax_angular_velocity.legend()
        
    def update_control_plot(self, control_predict, control_ref):
        self.ax_control.clear()
        self.ax_control.set_title('Control')
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.set_ylabel('Control')
        
        # 计算参考轨迹时间轴 (使用轨迹优化的dt)
        ref_time = np.arange(len(control_ref)) * self.dt_traj_opt / 1000.0
        
        # 计算预测轨迹时间轴 (使用MPC的dt)
        predict_start_time = self.mpc_ref_time / 1000.0
        predict_time = predict_start_time + np.arange(len(control_predict)) * self.dt_mpc / 1000.0
        
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        control_num = control_predict.shape[1]
        for i in range(control_num):
            self.ax_control.plot(predict_time,
                                 control_predict[:, i], label='Control_{}'.format(i), color=colors[i])
            self.ax_control.plot(ref_time, control_ref[:, i],
                                 label='Control_ref_{}'.format(i), linestyle='--', color=colors[i])
        
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
    
    #endregion
    
    def set_to_reference_button_clicked(self):
        
        time_index = int(self.mpc_ref_time / self.dt_traj_opt)
        time_index = min(time_index, len(self.state_ref) - 1)
        ref_state = self.state_ref[time_index]
        # Update state
        self.state = np.copy(ref_state)
        
        # Update all sliders and labels to match reference state
        # Position
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(ref_state[i] * 100))
            self.state_labels[axis].setText(f'{ref_state[i]:.2f}')
            self.state_sliders[axis].blockSignals(False)
            
        # Orientation (convert quaternion to euler angles)
        euler = R.from_quat(ref_state[3:7]).as_euler('xyz', degrees=True)
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(euler[i]))
            self.state_labels[axis].setText(f'{euler[i]:.2f}')
            self.state_sliders[axis].blockSignals(False)
            
        # Linear velocity
        for i, axis in enumerate(['vx', 'vy', 'vz']):
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(ref_state[i + 7] * 100))
            self.state_labels[axis].setText(f'{ref_state[i + 7]:.2f}')
            self.state_sliders[axis].blockSignals(False)
            
        # Angular velocity (convert to degrees)
        for i, axis in enumerate(['wx', 'wy', 'wz']):
            ang_vel_deg = np.degrees(ref_state[i + 10])
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(ang_vel_deg))
            self.state_labels[axis].setText(f'{ang_vel_deg:.2f}')
            self.state_sliders[axis].blockSignals(False)

    def set_to_reference(self):
        selected = self.ref_selector.currentText()
        
        if selected == 'Initial State':
            ref_state = self.state_ref[0]
        elif selected == 'Final State':
            ref_state = self.state_ref[-1]
        elif selected == 'Current State':
            ref_state = self.state
        elif selected == 'Current Reference':  # Current Reference
            time_index = int(self.mpc_ref_time / self.dt_traj_opt)
            time_index = min(time_index, len(self.state_ref) - 1)
            ref_state = self.state_ref[time_index]
            
        # Update state
        self.state = np.copy(ref_state)
        
        # Update all sliders and labels to match reference state
        # Position
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(ref_state[i] * 100))
            self.state_labels[axis].setText(f'{ref_state[i]:.2f}')
            self.state_sliders[axis].blockSignals(False)
            
        # Orientation (convert quaternion to euler angles)
        euler = R.from_quat(ref_state[3:7]).as_euler('xyz', degrees=True)
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(euler[i]))
            self.state_labels[axis].setText(f'{euler[i]:.2f}')
            self.state_sliders[axis].blockSignals(False)
            
        # Linear velocity
        for i, axis in enumerate(['vx', 'vy', 'vz']):
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(ref_state[i + 7] * 100))
            self.state_labels[axis].setText(f'{ref_state[i + 7]:.2f}')
            self.state_sliders[axis].blockSignals(False)
            
        # Angular velocity (convert to degrees)
        for i, axis in enumerate(['wx', 'wy', 'wz']):
            ang_vel_deg = np.degrees(ref_state[i + 10])
            self.state_sliders[axis].blockSignals(True)
            self.state_sliders[axis].setValue(int(ang_vel_deg))
            self.state_labels[axis].setText(f'{ang_vel_deg:.2f}')
            self.state_sliders[axis].blockSignals(False)


if __name__ == '__main__':
    import sys
    
    # Settings
    using_ros = False
    mpc_name = 'rail'
    mpc_yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml'
    
    robot_name = 'iris'
    trajectory_name = 'hover'
    dt_traj_opt = 10  # ms
    useSquash = True
    
    if using_ros:
        rospy.init_node('mpc_debug_interface')
    
    app = QtWidgets.QApplication(sys.argv)
    window = MpcDebugInterface(using_ros=using_ros, mpc_name=mpc_name, mpc_yaml_path=mpc_yaml_path, robot_name=robot_name, trajectory_name=trajectory_name, dt_traj_opt=dt_traj_opt, useSquash=useSquash)
    window.show()
    
    sys.exit(app.exec_()) 