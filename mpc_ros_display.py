#!/usr/bin/env python3
'''
MPC ROS Data Display GUI
Author: Lei He
Date: 2024-12-19
Description: Independent PyQtGraph GUI for displaying MPC data from ROS topics
'''

import sys
import os
import numpy as np
import time
from datetime import datetime

# PyQt5 imports
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                 QWidget, QLabel, QGridLayout, QGroupBox, QPushButton,
                                 QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox)
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QFont
    import pyqtgraph as pg
    PYQT_AVAILABLE = True
except ImportError as e:
    print(f"PyQt5/pyqtgraph not available: {e}")
    PYQT_AVAILABLE = False

# ROS imports
try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray
    from geometry_msgs.msg import PoseStamped
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. GUI will run in simulation mode.")

class ROSDataSubscriber(QThread):
    """ROS data subscriber running in separate thread"""
    
    data_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.joint_states = None
        self.control_input = None
        self.target_state = None
        self.mpc_cost = 0.0
        self.solve_time = 0.0
        self.iterations = 0
        self.predicted_states = None
        self.predicted_controls = None
        
        # Initialize ROS node if available
        if ROS_AVAILABLE:
            rospy.init_node('mpc_display_gui', anonymous=True)
            
            # Subscribers
            self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
            self.control_sub = rospy.Subscriber('/mpc/control_input', Float64MultiArray, self.control_callback)
            self.target_sub = rospy.Subscriber('/mpc/target_state', Float64MultiArray, self.target_callback)
            
            # MPC solver information subscribers
            self.solver_info_sub = rospy.Subscriber('/mpc/solver_info', Float64MultiArray, self.solver_info_callback)
            self.predicted_states_sub = rospy.Subscriber('/mpc/predicted_states', Float64MultiArray, self.predicted_states_callback)
            self.predicted_controls_sub = rospy.Subscriber('/mpc/predicted_controls', Float64MultiArray, self.predicted_controls_callback)
            self.cost_sub = rospy.Subscriber('/mpc/cost', Float64, self.cost_callback)
            self.solve_time_sub = rospy.Subscriber('/mpc/solve_time', Float64, self.solve_time_callback)
            self.iterations_sub = rospy.Subscriber('/mpc/iterations', Float64, self.iterations_callback)
            
            # For simulation mode, also subscribe to simulation topics
            self.sim_joint_sub = rospy.Subscriber('/arm_controller/joint_states', JointState, self.joint_callback)
            self.sim_control_sub = rospy.Subscriber('/arm_controller/joint_1_position_controller/command', Float64, self.sim_control_callback)
    
    def joint_callback(self, msg):
        """Callback for joint state messages"""
        try:
            # Extract joint positions and velocities by name
            joint_data = {
                name: {
                    'position': position,
                    'velocity': velocity
                }
                for name, position, velocity in zip(msg.name, msg.position, msg.velocity)
            }
            
            # Extract joint_1 and joint_2 data by name
            if 'joint_1' in joint_data and 'joint_2' in joint_data:
                self.joint_states = {
                    'position': np.array([joint_data['joint_1']['position'], joint_data['joint_2']['position']]),
                    'velocity': np.array([joint_data['joint_1']['velocity'], joint_data['joint_2']['velocity']]),
                    'timestamp': msg.header.stamp.to_sec() if msg.header.stamp else time.time()
                }
                self.emit_data()
        except Exception as e:
            print(f"Error in joint callback: {e}")
    
    def control_callback(self, msg):
        """Callback for control input messages"""
        try:
            if len(msg.data) >= 2:
                self.control_input = np.array(msg.data[:2])
                self.emit_data()
        except Exception as e:
            print(f"Error in control callback: {e}")
    
    def target_callback(self, msg):
        """Callback for target state messages"""
        try:
            if len(msg.data) >= 2:
                self.target_state = np.array(msg.data[:2])
                self.emit_data()
        except Exception as e:
            print(f"Error in target callback: {e}")
    
    def sim_control_callback(self, msg):
        """Callback for simulation control messages"""
        try:
            # In simulation, we might get individual joint commands
            # This topic is specifically for joint_1, so we set it as the first control input
            if self.control_input is None:
                self.control_input = np.array([msg.data, 0.0])
            else:
                self.control_input[0] = msg.data  # joint_1 control
            self.emit_data()
        except Exception as e:
            print(f"Error in sim control callback: {e}")
    
    def solver_info_callback(self, msg):
        """Callback for solver information messages"""
        try:
            if len(msg.data) >= 3:
                self.mpc_cost = msg.data[0]
                self.solve_time = msg.data[1]
                self.iterations = int(msg.data[2])
                self.emit_data()
        except Exception as e:
            print(f"Error in solver info callback: {e}")
    
    def predicted_states_callback(self, msg):
        """Callback for predicted states messages"""
        try:
            # Store predicted states for display
            if len(msg.data) > 0:
                # Reshape: [horizon * state_dim] -> [horizon, state_dim]
                horizon = len(msg.data) // 4  # Assuming 4D state (2 pos + 2 vel)
                if horizon > 0:
                    states = np.array(msg.data).reshape(horizon, 4)
                    # Store all predicted states for display
                    self.predicted_states = states
        except Exception as e:
            print(f"Error in predicted states callback: {e}")
    
    def predicted_controls_callback(self, msg):
        """Callback for predicted controls messages"""
        try:
            # Store predicted controls for display
            if len(msg.data) > 0:
                # Reshape: [horizon * control_dim] -> [horizon, control_dim]
                horizon = len(msg.data) // 2  # Assuming 2D control
                if horizon > 0:
                    controls = np.array(msg.data).reshape(horizon, 2)
                    # Store all predicted controls for display
                    self.predicted_controls = controls
        except Exception as e:
            print(f"Error in predicted controls callback: {e}")
    
    def cost_callback(self, msg):
        """Callback for cost messages"""
        try:
            self.mpc_cost = msg.data
            self.emit_data()
        except Exception as e:
            print(f"Error in cost callback: {e}")
    
    def solve_time_callback(self, msg):
        """Callback for solve time messages"""
        try:
            self.solve_time = msg.data
            self.emit_data()
        except Exception as e:
            print(f"Error in solve time callback: {e}")
    
    def iterations_callback(self, msg):
        """Callback for iterations messages"""
        try:
            self.iterations = int(msg.data)
            self.emit_data()
        except Exception as e:
            print(f"Error in iterations callback: {e}")
    
    def emit_data(self):
        """Emit collected data"""
        if self.joint_states is not None:
            data = {
                'joint_states': self.joint_states,
                'control_input': self.control_input,
                'target_state': self.target_state,
                'mpc_cost': self.mpc_cost,
                'solve_time': self.solve_time,
                'iterations': self.iterations,
                'timestamp': time.time()
            }
            self.data_received.emit(data)
    
    def run(self):
        """Main thread loop"""
        self.is_running = True
        
        if ROS_AVAILABLE:
            # ROS spin loop
            rate = rospy.Rate(100)  # 100 Hz
            while self.is_running and not rospy.is_shutdown():
                rate.sleep()
        else:
            # Simulation mode - generate fake data
            while self.is_running:
                # Generate fake data for demonstration
                t = time.time()
                self.joint_states = {
                    'position': np.array([0.5 * np.sin(t), 0.3 * np.cos(t)]),
                    'velocity': np.array([0.5 * np.cos(t), -0.3 * np.sin(t)]),
                    'timestamp': t
                }
                self.control_input = np.array([0.1 * np.sin(t), 0.05 * np.cos(t)])
                self.target_state = np.array([0.4, 0.5])
                self.mpc_cost = 0.1 + 0.05 * np.sin(t)
                self.solve_time = 0.001 + 0.0005 * np.sin(t)
                self.iterations = 10 + int(5 * np.sin(t))
                
                self.emit_data()
                time.sleep(0.01)  # 100 Hz
    
    def stop(self):
        """Stop the thread"""
        self.is_running = False

class MPCDisplayGUI(QMainWindow):
    """Main GUI window for MPC data display"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data storage
        self.time_data = []
        self.position_data = []
        self.velocity_data = []
        self.control_data = []
        self.target_data = []
        self.cost_data = []
        self.solve_time_data = []
        self.iterations_data = []
        
        # Data retention settings
        self.max_data_points = 1000
        self.update_rate = 10  # Hz
        
        # Initialize GUI
        self.init_ui()
        
        # Initialize ROS subscriber
        self.ros_subscriber = ROSDataSubscriber()
        self.ros_subscriber.data_received.connect(self.update_display)
        self.ros_subscriber.start()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(1000 // self.update_rate)  # Convert Hz to milliseconds
        
        print("MPC Display GUI initialized")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('MPC ROS Data Display')
        self.setGeometry(100, 100, 1800, 1400)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create plots widget
        self.create_plots_widget(main_layout)
        
        # Create control panel
        self.create_control_panel(main_layout)
        
        # Create info panel
        self.create_info_panel(main_layout)
    
    def create_plots_widget(self, parent_layout):
        """Create the plots widget"""
        plots_widget = QWidget()
        plots_layout = QGridLayout(plots_widget)
        
        # Configure PyQtGraph with unified style
        pg.setConfigOptions(antialias=True)
        
        # Set unified style for all plots
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')  # Black text
        
        # Set plot style
        pg.setConfigOption('useOpenGL', False)
        
        # Create plots
        self.plots = {}
        
        # Helper function to set plot style
        def setup_plot_style(plot):
            plot.getAxis('left').setPen(pg.mkPen('k'))
            plot.getAxis('bottom').setPen(pg.mkPen('k'))
        
        # Position tracking plot
        self.plots['position'] = pg.PlotWidget(title='Joint Position Tracking')
        self.plots['position'].setLabel('left', 'Position (rad)')
        self.plots['position'].setLabel('bottom', 'Time (s)')
        self.plots['position'].addLegend()
        self.plots['position'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['position'])
        self.position_curves = {
            'joint1_actual': self.plots['position'].plot(pen=pg.mkPen('b', width=2), name='Joint 1 Actual'),
            'joint2_actual': self.plots['position'].plot(pen=pg.mkPen('r', width=2), name='Joint 2 Actual'),
            'joint1_target': self.plots['position'].plot(pen=pg.mkPen('b', width=1, style=pg.QtCore.Qt.DashLine), name='Joint 1 Target'),
            'joint2_target': self.plots['position'].plot(pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine), name='Joint 2 Target')
        }
        
        # Velocity plot
        self.plots['velocity'] = pg.PlotWidget(title='Joint Velocity')
        self.plots['velocity'].setLabel('left', 'Velocity (rad/s)')
        self.plots['velocity'].setLabel('bottom', 'Time (s)')
        self.plots['velocity'].addLegend()
        self.plots['velocity'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['velocity'])
        self.velocity_curves = {
            'joint1': self.plots['velocity'].plot(pen=pg.mkPen('b', width=2), name='Joint 1'),
            'joint2': self.plots['velocity'].plot(pen=pg.mkPen('r', width=2), name='Joint 2'),
            'joint1_target': self.plots['velocity'].plot(pen=pg.mkPen('b', width=1, style=pg.QtCore.Qt.DashLine), name='Joint 1 Target'),
            'joint2_target': self.plots['velocity'].plot(pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine), name='Joint 2 Target')
        }
        
        # Control input plot
        self.plots['control'] = pg.PlotWidget(title='Control Input')
        self.plots['control'].setLabel('left', 'Control (Nm)')
        self.plots['control'].setLabel('bottom', 'Time (s)')
        self.plots['control'].addLegend()
        self.plots['control'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['control'])
        self.control_curves = {
            'joint1': self.plots['control'].plot(pen=pg.mkPen('b', width=2), name='Joint 1'),
            'joint2': self.plots['control'].plot(pen=pg.mkPen('r', width=2), name='Joint 2')
        }
        
        # Cost function plot
        self.plots['cost'] = pg.PlotWidget(title='MPC Optimization Cost')
        self.plots['cost'].setLabel('left', 'Cost')
        self.plots['cost'].setLabel('bottom', 'Time (s)')
        self.plots['cost'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['cost'])
        self.cost_curve = self.plots['cost'].plot(pen=pg.mkPen('g', width=2))
        
        # Solve time plot
        self.plots['solve_time'] = pg.PlotWidget(title='MPC Solve Time')
        self.plots['solve_time'].setLabel('left', 'Time (ms)')
        self.plots['solve_time'].setLabel('bottom', 'Time (s)')
        self.plots['solve_time'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['solve_time'])
        self.solve_time_curve = self.plots['solve_time'].plot(pen=pg.mkPen('m', width=2))
        
        # Iterations plot
        self.plots['iterations'] = pg.PlotWidget(title='MPC Solver Iterations')
        self.plots['iterations'].setLabel('left', 'Iterations')
        self.plots['iterations'].setLabel('bottom', 'Time (s)')
        self.plots['iterations'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['iterations'])
        self.iterations_curve = self.plots['iterations'].plot(pen=pg.mkPen('c', width=2))
        
        # Predicted positions plot
        self.plots['predicted_positions'] = pg.PlotWidget(title='MPC Predicted Positions (All Steps)')
        self.plots['predicted_positions'].setLabel('left', 'Position (rad)')
        self.plots['predicted_positions'].setLabel('bottom', 'Time Steps')
        self.plots['predicted_positions'].addLegend()
        self.plots['predicted_positions'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['predicted_positions'])
        self.predicted_positions_curves = {
            'joint1': self.plots['predicted_positions'].plot(pen=pg.mkPen('b', width=2), name='Joint 1'),
            'joint2': self.plots['predicted_positions'].plot(pen=pg.mkPen('r', width=2), name='Joint 2'),
            'joint1_target': self.plots['predicted_positions'].plot(pen=pg.mkPen('b', width=1, style=pg.QtCore.Qt.DashLine), name='Joint 1 Target'),
            'joint2_target': self.plots['predicted_positions'].plot(pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine), name='Joint 2 Target')
        }
        
        # Predicted velocities plot
        self.plots['predicted_velocities'] = pg.PlotWidget(title='MPC Predicted Velocities (All Steps)')
        self.plots['predicted_velocities'].setLabel('left', 'Velocity (rad/s)')
        self.plots['predicted_velocities'].setLabel('bottom', 'Time Steps')
        self.plots['predicted_velocities'].addLegend()
        self.plots['predicted_velocities'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['predicted_velocities'])
        self.predicted_velocities_curves = {
            'joint1': self.plots['predicted_velocities'].plot(pen=pg.mkPen('b', width=2), name='Joint 1'),
            'joint2': self.plots['predicted_velocities'].plot(pen=pg.mkPen('r', width=2), name='Joint 2')
        }
        
        # Predicted controls plot
        self.plots['predicted_controls'] = pg.PlotWidget(title='MPC Predicted Controls (All Steps)')
        self.plots['predicted_controls'].setLabel('left', 'Control (Nm)')
        self.plots['predicted_controls'].setLabel('bottom', 'Time Steps')
        self.plots['predicted_controls'].addLegend()
        self.plots['predicted_controls'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['predicted_controls'])
        self.predicted_controls_curves = {
            'joint1': self.plots['predicted_controls'].plot(pen=pg.mkPen('b', width=2), name='Joint 1'),
            'joint2': self.plots['predicted_controls'].plot(pen=pg.mkPen('r', width=2), name='Joint 2')
        }
        
        # Add plots to grid (3x4 layout)
        plots_layout.addWidget(self.plots['position'], 0, 0)
        plots_layout.addWidget(self.plots['velocity'], 0, 1)
        plots_layout.addWidget(self.plots['control'], 0, 2)
        plots_layout.addWidget(self.plots['cost'], 0, 3)
        plots_layout.addWidget(self.plots['solve_time'], 1, 0)
        plots_layout.addWidget(self.plots['iterations'], 1, 1)
        plots_layout.addWidget(self.plots['predicted_positions'], 1, 2)
        plots_layout.addWidget(self.plots['predicted_velocities'], 1, 3)
        plots_layout.addWidget(self.plots['predicted_controls'], 2, 0, 1, 2)  # Span 2 columns
        
        parent_layout.addWidget(plots_widget, 2)
    
    def create_control_panel(self, parent_layout):
        """Create the control panel"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Data retention settings
        retention_group = QGroupBox("Data Retention")
        retention_layout = QVBoxLayout(retention_group)
        
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(100, 10000)
        self.max_points_spin.setValue(self.max_data_points)
        self.max_points_spin.valueChanged.connect(self.update_max_points)
        retention_layout.addWidget(QLabel("Max Data Points:"))
        retention_layout.addWidget(self.max_points_spin)
        
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(1, 100)
        self.update_rate_spin.setValue(self.update_rate)
        self.update_rate_spin.valueChanged.connect(self.update_rate_changed)
        retention_layout.addWidget(QLabel("Update Rate (Hz):"))
        retention_layout.addWidget(self.update_rate_spin)
        
        control_layout.addWidget(retention_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        display_layout.addWidget(self.auto_scale_check)
        
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self.toggle_grid)
        display_layout.addWidget(self.show_grid_check)
        
        control_layout.addWidget(display_group)
        
        # Control buttons
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout(button_group)
        
        self.clear_button = QPushButton("Clear Data")
        self.clear_button.clicked.connect(self.clear_data)
        button_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_button)
        
        self.pause_button = QPushButton("Pause/Resume")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)
        
        control_layout.addWidget(button_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Running")
        status_layout.addWidget(self.status_label)
        
        self.data_count_label = QLabel("Data Points: 0")
        status_layout.addWidget(self.data_count_label)
        
        self.ros_status_label = QLabel("ROS: Connected" if ROS_AVAILABLE else "ROS: Not Available")
        status_layout.addWidget(self.ros_status_label)
        
        control_layout.addWidget(status_group)
        
        control_layout.addStretch()
        parent_layout.addWidget(control_widget, 1)
    
    def create_info_panel(self, parent_layout):
        """Create the information panel"""
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        # Current state information
        state_group = QGroupBox("Current State")
        state_layout = QVBoxLayout(state_group)
        
        self.joint1_pos_label = QLabel("Joint 1 Position: 0.000")
        self.joint2_pos_label = QLabel("Joint 2 Position: 0.000")
        self.joint1_vel_label = QLabel("Joint 1 Velocity: 0.000")
        self.joint2_vel_label = QLabel("Joint 2 Velocity: 0.000")
        
        for label in [self.joint1_pos_label, self.joint2_pos_label, 
                     self.joint1_vel_label, self.joint2_vel_label]:
            state_layout.addWidget(label)
        
        info_layout.addWidget(state_group)
        
        # Target information
        target_group = QGroupBox("Target Information")
        target_layout = QVBoxLayout(target_group)
        
        self.target1_label = QLabel("Target 1: 0.000")
        self.target2_label = QLabel("Target 2: 0.000")
        
        for label in [self.target1_label, self.target2_label]:
            target_layout.addWidget(label)
        
        info_layout.addWidget(target_group)
        
        # Control information
        control_group = QGroupBox("Control Information")
        control_layout = QVBoxLayout(control_group)
        
        self.control1_label = QLabel("Control 1: 0.000")
        self.control2_label = QLabel("Control 2: 0.000")
        self.control_mag_label = QLabel("Control Magnitude: 0.000")
        
        for label in [self.control1_label, self.control2_label, self.control_mag_label]:
            control_layout.addWidget(label)
        
        info_layout.addWidget(control_group)
        
        # MPC information
        mpc_group = QGroupBox("MPC Information")
        mpc_layout = QVBoxLayout(mpc_group)
        
        self.cost_label = QLabel("Current Cost: 0.000")
        self.solve_time_label = QLabel("Solve Time: 0.000 ms")
        self.iterations_label = QLabel("Iterations: 0")
        
        for label in [self.cost_label, self.solve_time_label, self.iterations_label]:
            mpc_layout.addWidget(label)
        
        info_layout.addWidget(mpc_group)
        
        # Error information
        error_group = QGroupBox("Tracking Error")
        error_layout = QVBoxLayout(error_group)
        
        self.pos_error1_label = QLabel("Position Error 1: 0.000")
        self.pos_error2_label = QLabel("Position Error 2: 0.000")
        self.vel_error1_label = QLabel("Velocity Error 1: 0.000")
        self.vel_error2_label = QLabel("Velocity Error 2: 0.000")
        
        for label in [self.pos_error1_label, self.pos_error2_label, 
                     self.vel_error1_label, self.vel_error2_label]:
            error_layout.addWidget(label)
        
        info_layout.addWidget(error_group)
        
        info_layout.addStretch()
        parent_layout.addWidget(info_widget, 1)
    
    def update_display(self, data):
        """Update display with new data"""
        try:
            # Add new data
            current_time = data.get('timestamp', time.time())
            self.time_data.append(current_time)
            
            if data.get('joint_states'):
                self.position_data.append(data['joint_states']['position'])
                self.velocity_data.append(data['joint_states']['velocity'])
            else:
                self.position_data.append(np.array([0.0, 0.0]))
                self.velocity_data.append(np.array([0.0, 0.0]))
            
            if data.get('control_input') is not None:
                self.control_data.append(data['control_input'])
            else:
                self.control_data.append(np.array([0.0, 0.0]))
            
            if data.get('target_state') is not None:
                self.target_data.append(data['target_state'])
            else:
                self.target_data.append(np.array([0.0, 0.0]))
            
            self.cost_data.append(data.get('mpc_cost', 0.0))
            self.solve_time_data.append(data.get('solve_time', 0.0))
            self.iterations_data.append(data.get('iterations', 0))
            
            # Update info labels
            self.update_info_labels(data)
            
            # Limit data points
            if len(self.time_data) > self.max_data_points:
                self.time_data = self.time_data[-self.max_data_points:]
                self.position_data = self.position_data[-self.max_data_points:]
                self.velocity_data = self.velocity_data[-self.max_data_points:]
                self.control_data = self.control_data[-self.max_data_points:]
                self.target_data = self.target_data[-self.max_data_points:]
                self.cost_data = self.cost_data[-self.max_data_points:]
                self.solve_time_data = self.solve_time_data[-self.max_data_points:]
                self.iterations_data = self.iterations_data[-self.max_data_points:]
            
            # Update status
            self.data_count_label.setText(f"Data Points: {len(self.time_data)}")
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def update_info_labels(self, data):
        """Update information labels"""
        try:
            if data.get('joint_states'):
                pos = data['joint_states']['position']
                vel = data['joint_states']['velocity']
                
                self.joint1_pos_label.setText(f"Joint 1 Position: {pos[0]:.3f}")
                self.joint2_pos_label.setText(f"Joint 2 Position: {pos[1]:.3f}")
                self.joint1_vel_label.setText(f"Joint 1 Velocity: {vel[0]:.3f}")
                self.joint2_vel_label.setText(f"Joint 2 Velocity: {vel[1]:.3f}")
            
            if data.get('target_state') is not None:
                target = data['target_state']
                self.target1_label.setText(f"Target 1: {target[0]:.3f}")
                self.target2_label.setText(f"Target 2: {target[1]:.3f}")
            
            if data.get('control_input') is not None:
                control = data['control_input']
                self.control1_label.setText(f"Control 1: {control[0]:.3f}")
                self.control2_label.setText(f"Control 2: {control[1]:.3f}")
                control_mag = np.linalg.norm(control)
                self.control_mag_label.setText(f"Control Magnitude: {control_mag:.3f}")
            
            self.cost_label.setText(f"Current Cost: {data.get('mpc_cost', 0.0):.6f}")
            self.solve_time_label.setText(f"Solve Time: {data.get('solve_time', 0.0)*1000:.1f} ms")
            self.iterations_label.setText(f"Iterations: {data.get('iterations', 0)}")
            
            # Calculate errors
            if data.get('joint_states') and data.get('target_state') is not None:
                pos = data['joint_states']['position']
                vel = data['joint_states']['velocity']
                target = data['target_state']
                
                pos_error1 = pos[0] - target[0]
                pos_error2 = pos[1] - target[1]
                vel_error1 = vel[0]  # Assuming target velocity is 0
                vel_error2 = vel[1]
                
                self.pos_error1_label.setText(f"Position Error 1: {pos_error1:.3f}")
                self.pos_error2_label.setText(f"Position Error 2: {pos_error2:.3f}")
                self.vel_error1_label.setText(f"Velocity Error 1: {vel_error1:.3f}")
                self.vel_error2_label.setText(f"Velocity Error 2: {vel_error2:.3f}")
            
        except Exception as e:
            print(f"Error updating info labels: {e}")
    
    def update_plots(self):
        """Update all plots"""
        try:
            if len(self.time_data) > 1:
                time_array = np.array(self.time_data)
                position_array = np.array(self.position_data)
                velocity_array = np.array(self.velocity_data)
                control_array = np.array(self.control_data)
                target_array = np.array(self.target_data)
                cost_array = np.array(self.cost_data)
                solve_time_array = np.array(self.solve_time_data)
                iterations_array = np.array(self.iterations_data)
                
                # Update position tracking plot
                self.position_curves['joint1_actual'].setData(time_array, position_array[:, 0])
                self.position_curves['joint2_actual'].setData(time_array, position_array[:, 1])
                self.position_curves['joint1_target'].setData(time_array, target_array[:, 0])
                self.position_curves['joint2_target'].setData(time_array, target_array[:, 1])
                
                # Update velocity plot
                self.velocity_curves['joint1'].setData(time_array, velocity_array[:, 0])
                self.velocity_curves['joint2'].setData(time_array, velocity_array[:, 1])
                # Add target velocities (typically 0 for position control)
                target_velocity = np.zeros_like(time_array)
                self.velocity_curves['joint1_target'].setData(time_array, target_velocity)
                self.velocity_curves['joint2_target'].setData(time_array, target_velocity)
                
                # Update control plot
                self.control_curves['joint1'].setData(time_array, control_array[:, 0])
                self.control_curves['joint2'].setData(time_array, control_array[:, 1])
                
                # Update cost plot
                self.cost_curve.setData(time_array, cost_array)
                
                # Update solve time plot
                self.solve_time_curve.setData(time_array, solve_time_array * 1000)  # Convert to ms
                
                # Update iterations plot
                self.iterations_curve.setData(time_array, iterations_array)
                
                # Update predicted positions plot (all steps)
                if hasattr(self.ros_subscriber, 'predicted_states') and self.ros_subscriber.predicted_states is not None:
                    predicted_states = self.ros_subscriber.predicted_states
                    if len(predicted_states) > 0:
                        steps = np.arange(len(predicted_states))
                        # States format: [pos1, pos2, vel1, vel2]
                        self.predicted_positions_curves['joint1'].setData(steps, predicted_states[:, 0])
                        self.predicted_positions_curves['joint2'].setData(steps, predicted_states[:, 1])
                        
                        # Add target positions (horizontal lines)
                        if len(self.target_data) > 0:
                            current_target = self.target_data[-1]  # Get latest target
                            target_steps = np.arange(len(predicted_states))
                            target_joint1 = np.full(len(predicted_states), current_target[0])
                            target_joint2 = np.full(len(predicted_states), current_target[1])
                            self.predicted_positions_curves['joint1_target'].setData(target_steps, target_joint1)
                            self.predicted_positions_curves['joint2_target'].setData(target_steps, target_joint2)
                
                # Update predicted velocities plot (all steps)
                if hasattr(self.ros_subscriber, 'predicted_states') and self.ros_subscriber.predicted_states is not None:
                    predicted_states = self.ros_subscriber.predicted_states
                    if len(predicted_states) > 0:
                        steps = np.arange(len(predicted_states))
                        # States format: [pos1, pos2, vel1, vel2]
                        self.predicted_velocities_curves['joint1'].setData(steps, predicted_states[:, 2])
                        self.predicted_velocities_curves['joint2'].setData(steps, predicted_states[:, 3])
                
                # Update predicted controls plot (all steps)
                if hasattr(self.ros_subscriber, 'predicted_controls') and self.ros_subscriber.predicted_controls is not None:
                    predicted_controls = self.ros_subscriber.predicted_controls
                    if len(predicted_controls) > 0:
                        steps = np.arange(len(predicted_controls))
                        self.predicted_controls_curves['joint1'].setData(steps, predicted_controls[:, 0])
                        self.predicted_controls_curves['joint2'].setData(steps, predicted_controls[:, 1])
                
                # Auto-scale if enabled
                if self.auto_scale_check.isChecked():
                    for plot in self.plots.values():
                        plot.autoRange()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def update_max_points(self, value):
        """Update maximum data points"""
        self.max_data_points = value
    
    def update_rate_changed(self, value):
        """Update display rate"""
        self.update_rate = value
        self.update_timer.setInterval(1000 // value)
    
    def toggle_grid(self, checked):
        """Toggle grid display"""
        for plot in self.plots.values():
            plot.showGrid(x=checked, y=checked)
    
    def clear_data(self):
        """Clear all data"""
        self.time_data.clear()
        self.position_data.clear()
        self.velocity_data.clear()
        self.control_data.clear()
        self.target_data.clear()
        self.cost_data.clear()
        self.solve_time_data.clear()
        self.iterations_data.clear()
        
        # Clear plots
        for plot in self.plots.values():
            plot.clear()
        
        self.data_count_label.setText("Data Points: 0")
    
    def save_data(self):
        """Save data to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mpc_data_{timestamp}.npz"
            
            data = {
                'time': np.array(self.time_data),
                'position': np.array(self.position_data),
                'velocity': np.array(self.velocity_data),
                'control': np.array(self.control_data),
                'target': np.array(self.target_data),
                'cost': np.array(self.cost_data),
                'solve_time': np.array(self.solve_time_data),
                'iterations': np.array(self.iterations_data)
            }
            
            np.savez(filename, **data)
            print(f"Data saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def toggle_pause(self, checked):
        """Toggle pause/resume"""
        if checked:
            self.update_timer.stop()
            self.status_label.setText("Status: Paused")
        else:
            self.update_timer.start()
            self.status_label.setText("Status: Running")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.ros_subscriber.stop()
        self.ros_subscriber.wait()
        event.accept()

def main():
    """Main function"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt5/pyqtgraph not available")
        return
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = MPCDisplayGUI()
    window.show()
    
    print("MPC ROS Display GUI started")
    print("Subscribing to ROS topics:")
    print("- /joint_states")
    print("- /mpc/control_input")
    print("- /mpc/target_state")
    print("- /mpc/solver_info")
    print("- /mpc/predicted_states")
    print("- /mpc/predicted_controls")
    print("- /mpc/cost")
    print("- /mpc/solve_time")
    print("- /mpc/iterations")
    print("- /arm_controller/joint_states (simulation)")
    print("- /arm_controller/joint_1_position_controller/command (simulation)")
    
    # Start the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 