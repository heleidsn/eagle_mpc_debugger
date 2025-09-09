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
    from eagle_mpc_msgs.msg import MpcState
    from std_srvs.srv import Trigger, TriggerResponse
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. GUI will run in simulation mode.")

class ROSServiceHandler:
    """ROS service handler for controlling system services"""
    
    def __init__(self):
        self.services = {}
        self.init_services()
        
    def init_services(self):
        """Initialize ROS service proxies"""
        try:
            self.services['open_gripper'] = rospy.ServiceProxy('/open_gripper', Trigger)
            self.services['close_gripper'] = rospy.ServiceProxy('/close_gripper', Trigger)
            self.services['reset_beer'] = rospy.ServiceProxy('/reset_beer', Trigger)
            self.services['start_trajectory'] = rospy.ServiceProxy('/start_trajectory', Trigger)
            self.services['initialize_trajectory'] = rospy.ServiceProxy('/initialize_trajectory', Trigger)
            self.services['start_l1_control'] = rospy.ServiceProxy('/start_l1_control', Trigger)
            self.services['stop_l1_control'] = rospy.ServiceProxy('/stop_l1_control', Trigger)
            self.services['start_arm_test'] = rospy.ServiceProxy('/start_arm_test', Trigger)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to initialize services: {str(e)}")
            
    def call_service(self, service_name, *args):
        """Call a ROS service with error handling"""
        try:
            if service_name in self.services:
                response = self.services[service_name]()
                return response.success, response.message
            else:
                return False, f"Service {service_name} not found"
        except rospy.ServiceException as e:
            return False, str(e)

class ROSDataSubscriber(QThread):
    """ROS data subscriber running in separate thread"""
    
    data_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.joint_states = None
        self.drone_states = None
        self.control_input = None
        self.target_state = None
        self.drone_target_state = None
        self.mpc_cost = 0.0
        self.solve_time = 0.0
        self.iterations = 0
        self.predicted_states = None
        self.predicted_controls = None
        
        # Robot model detection
        self.robot_model = None  # 's500' or 's500_uam'
        self.state_dim = None
        self.control_dim = None
        self.has_arm = False
        
        # Initialize ROS node if available
        if ROS_AVAILABLE:
            rospy.init_node('mpc_display_gui', anonymous=False)
            
            # Subscribers
            self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
            self.control_sub = rospy.Subscriber('/mpc/control_input', Float64MultiArray, self.control_callback)
            self.target_sub = rospy.Subscriber('/mpc/target_state', Float64MultiArray, self.target_callback)
            
            self.mpc_state_sub = rospy.Subscriber('/mpc/state', MpcState, self.mpc_state_callback)
            
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
            
    def mpc_state_callback(self, msg):
        """Callback for MPC state messages"""
        try:
            self.mpc_state = msg
            
            # Auto-detect robot model based on state dimension
            current_state_dim = len(self.mpc_state.state)
            if self.state_dim != current_state_dim:
                self.detect_robot_model(current_state_dim)
            
            # Extract drone states (common for both models)
            if len(self.mpc_state.state) >= 12:
                if self.robot_model == 's500':
                    # s500: 13D state [pos(3), quat(4), vel(3), angular_vel(3)]
                    self.drone_states = {
                        'position': np.array(self.mpc_state.state[0:3]),  # x, y, z
                        'velocity': np.array(self.mpc_state.state[6:9]), # vx, vy, vz
                        'orientation': np.array(self.mpc_state.state[3:6]), # roll, pitch, yaw
                        'angular_velocity': np.array(self.mpc_state.state[9:12]), # wx, wy, wz
                        'timestamp': time.time()
                    }
                    
                    # s500 target states
                    if len(self.mpc_state.state_ref) >= 12:
                        self.drone_target_state = {
                            'position': np.array(self.mpc_state.state_ref[0:3]),
                            'velocity': np.array(self.mpc_state.state_ref[6:9]),
                            'orientation': np.array(self.mpc_state.state_ref[3:6]),
                            'angular_velocity': np.array(self.mpc_state.state_ref[9:12])
                        }
                        
                elif self.robot_model == 's500_uam' and len(self.mpc_state.state) >= 16:
                    # s500_uam: 16D state [pos(3), quat(4), joint_pos(2), vel(3), angular_vel(3), joint_vel(2)]
                    self.drone_states = {
                        'position': np.array(self.mpc_state.state[0:3]),  # x, y, z
                        'velocity': np.array(self.mpc_state.state[6:9]), # vx, vy, vz
                        'orientation': np.array(self.mpc_state.state[3:6]), # roll, pitch, yaw
                        'angular_velocity': np.array(self.mpc_state.state[9:12]), # wx, wy, wz
                        'timestamp': time.time()
                    }
                    
                    # Extract joint states for s500_uam
                    self.joint_states = {
                        'position': np.array(self.mpc_state.state[7:9]),   # joint positions
                        'velocity': np.array(self.mpc_state.state[15:17]), # joint velocities
                        'timestamp': time.time()
                    }
                    
                    # s500_uam target states
                    if len(self.mpc_state.state_ref) >= 17:
                        self.drone_target_state = {
                            'position': np.array(self.mpc_state.state_ref[0:3]),
                            'velocity': np.array(self.mpc_state.state_ref[9:12]),
                            'orientation': np.array(self.mpc_state.state_ref[3:7]),
                            'angular_velocity': np.array(self.mpc_state.state_ref[12:15])
                        }
                        
                        # Joint target states
                        self.target_state = self.mpc_state.state_ref[7:9]
            
            self.mpc_cost = self.mpc_state.mpc_final_cost
            self.iterations = self.mpc_state.mpc_iter_num
            
            self.emit_data()
        except Exception as e:
            print(f"Error in MPC state callback: {e}")
    
    def control_callback(self, msg):
        """Callback for control input messages"""
        try:
            # Accept control input of different dimensions
            if len(msg.data) >= 2:
                self.control_input = np.array(msg.data)
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
        # print(f"Predicted states callback: {msg.data}")
        try:
            # Store predicted states for display
            if len(msg.data) > 0:
                # Reshape: [horizon * state_dim] -> [horizon, state_dim]
                horizon = len(msg.data) // 13  # Assuming 16D state (2 pos + 2 vel)
                if horizon > 0:
                    states = np.array(msg.data).reshape(horizon, 13)
                    # Store all predicted states for display
                    self.predicted_states = states
                    print(f"Predicted states received: shape {states.shape}, state_dim={self.state_dim}")
        except Exception as e:
            print(f"Error in predicted states callback: {e}")
    
    def predicted_controls_callback(self, msg):
        """Callback for predicted controls messages"""
        try:
            # Store predicted controls for display
            if len(msg.data) > 0 and self.control_dim is not None:
                # Reshape: [horizon * control_dim] -> [horizon, control_dim]
                horizon = len(msg.data) // self.control_dim
                if horizon > 0:
                    controls = np.array(msg.data).reshape(horizon, self.control_dim)
                    # Store all predicted controls for display
                    self.predicted_controls = controls
                    print(f"Predicted controls received: shape {controls.shape}, control_dim={self.control_dim}")
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
    
    def detect_robot_model(self, state_dim):
        """Detect robot model based on state dimension"""
        self.state_dim = state_dim
        if state_dim == 12:
            self.robot_model = 's500'
            self.control_dim = 4
            self.has_arm = False
            print(f"Detected robot model: s500 ({state_dim}D state, {self.control_dim}D control, no arm)")
        elif state_dim == 17:
            self.robot_model = 's500_uam'
            self.control_dim = 8
            self.has_arm = True
            print(f"Detected robot model: s500_uam ({state_dim}D state, {self.control_dim}D control, with 2DOF arm)")
        else:
            print(f"Warning: Unknown state dimension {state_dim}, defaulting to s500_uam")
            self.robot_model = 's500_uam'
            self.control_dim = 8
            self.has_arm = True
    
    def emit_data(self):
        """Emit collected data"""
        if self.joint_states is not None or self.drone_states is not None:
            data = {
                'joint_states': self.joint_states,
                'drone_states': self.drone_states,
                'control_input': self.control_input,
                'target_state': self.target_state,
                'drone_target_state': self.drone_target_state,
                'mpc_cost': self.mpc_cost,
                'solve_time': self.solve_time,
                'iterations': self.iterations,
                'robot_model': self.robot_model,
                'has_arm': self.has_arm,
                'state_dim': self.state_dim,
                'control_dim': self.control_dim,
                'predicted_states': getattr(self, 'predicted_states', None),
                'predicted_controls': getattr(self, 'predicted_controls', None),
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
            simulation_model = 's500_uam'  # Can be changed to 's500' for testing
            self.detect_robot_model(17 if simulation_model == 's500_uam' else 13)
            
            while self.is_running:
                # Generate fake data for demonstration
                t = time.time()
                
                # Generate fake drone data (common for both models)
                self.drone_states = {
                    'position': np.array([2.0 * np.sin(0.5*t), 1.5 * np.cos(0.5*t), 1.0 + 0.2 * np.sin(t)]),
                    'velocity': np.array([1.0 * np.cos(0.5*t), -0.75 * np.sin(0.5*t), 0.2 * np.cos(t)]),
                    'orientation': np.array([0.1 * np.sin(t), 0.1 * np.cos(t), 0.05 * np.sin(2*t), 0.95]),
                    'angular_velocity': np.array([0.2 * np.cos(t), 0.1 * np.sin(t), 0.05 * np.sin(2*t)]),
                    'timestamp': t
                }
                
                # Generate fake drone target data
                self.drone_target_state = {
                    'position': np.array([2.0, 1.5, 1.0]),
                    'velocity': np.array([0.0, 0.0, 0.0]),
                    'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                    'angular_velocity': np.array([0.0, 0.0, 0.0])
                }
                
                # Generate joint data only for s500_uam
                if simulation_model == 's500_uam':
                    self.joint_states = {
                        'position': np.array([0.5 * np.sin(t), 0.3 * np.cos(t)]),
                        'velocity': np.array([0.5 * np.cos(t), -0.3 * np.sin(t)]),
                        'timestamp': t
                    }
                    self.target_state = np.array([0.4, 0.5])
                    self.control_input = np.array([0.1 * np.sin(t), 0.05 * np.cos(t)])
                else:
                    self.joint_states = None
                    self.target_state = None
                    self.control_input = np.array([0.1 * np.sin(t), 0.05 * np.cos(t), 0.03 * np.sin(2*t), 0.02 * np.cos(2*t), 0.01 * np.sin(3*t), 0.01 * np.cos(3*t)])
                
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
        # Joint data
        self.position_data = []
        self.velocity_data = []
        self.control_data = []
        self.target_data = []
        # Drone data
        self.drone_position_data = []
        self.drone_velocity_data = []
        self.drone_target_position_data = []
        self.drone_target_velocity_data = []
        self.drone_orientation_data = []
        self.drone_angular_velocity_data = []
        # Common data
        self.cost_data = []
        self.solve_time_data = []
        self.iterations_data = []
        
        # Display mode: 'joint' or 'drone'
        self.display_mode = 'joint'
        
        # Robot model info
        self.current_robot_model = None
        self.current_has_arm = False
        
        # Data retention settings
        self.max_data_points = 5000
        self.update_rate = 10  # Hz
        
        # Initialize service handler
        self.service_handler = ROSServiceHandler()
        
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
        
        # Service status clear timer
        self.service_status_timer = QTimer()
        self.service_status_timer.timeout.connect(self.clear_service_status)
        self.service_status_timer.setSingleShot(True)
        
        print("MPC Display GUI initialized")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('MPC Plotter')
        self.setGeometry(100, 100, 2800, 1400)
        
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
        self.plots['position'] = pg.PlotWidget(title='Joint Position')
        self.plots['position'].setLabel('left', 'Position (rad)')
        self.plots['position'].setLabel('bottom', 'Time (s)')
        self.plots['position'].addLegend()
        self.plots['position'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['position'])
        self.position_curves = {
            'joint1_actual': self.plots['position'].plot(pen=pg.mkPen('b', width=3), name='Joint 1 Actual'),
            'joint2_actual': self.plots['position'].plot(pen=pg.mkPen('g', width=3), name='Joint 2 Actual'),
            'joint1_target': self.plots['position'].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Joint 1 Target'),
            'joint2_target': self.plots['position'].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Joint 2 Target')
        }
        
        # Velocity plot
        self.plots['velocity'] = pg.PlotWidget(title='Joint Velocity')
        self.plots['velocity'].setLabel('left', 'Velocity (rad/s)')
        self.plots['velocity'].setLabel('bottom', 'Time (s)')
        self.plots['velocity'].addLegend()
        self.plots['velocity'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['velocity'])
        self.velocity_curves = {
            'joint1': self.plots['velocity'].plot(pen=pg.mkPen('b', width=3), name='Joint 1'),
            'joint2': self.plots['velocity'].plot(pen=pg.mkPen('g', width=3), name='Joint 2'),
            'joint1_target': self.plots['velocity'].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Joint 1 Target'),
            'joint2_target': self.plots['velocity'].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Joint 2 Target')
        }
        
        # Joint effort plot
        self.plots['control'] = pg.PlotWidget(title='Joint Effort')
        self.plots['control'].setLabel('left', 'Control (Nm)')
        self.plots['control'].setLabel('bottom', 'Time (s)')
        self.plots['control'].addLegend()
        self.plots['control'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['control'])
        self.control_curves = {
            'joint1': self.plots['control'].plot(pen=pg.mkPen('b', width=3), name='Joint 1'),
            'joint2': self.plots['control'].plot(pen=pg.mkPen('g', width=3), name='Joint 2')
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
            'joint1': self.plots['predicted_positions'].plot(pen=pg.mkPen('b', width=3), name='Joint 1'),
            'joint2': self.plots['predicted_positions'].plot(pen=pg.mkPen('g', width=3), name='Joint 2'),
            'joint1_target': self.plots['predicted_positions'].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Joint 1 Target'),
            'joint2_target': self.plots['predicted_positions'].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Joint 2 Target')
        }
        
        # Predicted velocities plot
        self.plots['predicted_velocities'] = pg.PlotWidget(title='MPC Predicted Velocities (All Steps)')
        self.plots['predicted_velocities'].setLabel('left', 'Velocity (rad/s)')
        self.plots['predicted_velocities'].setLabel('bottom', 'Time Steps')
        self.plots['predicted_velocities'].addLegend()
        self.plots['predicted_velocities'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['predicted_velocities'])
        self.predicted_velocities_curves = {
            'joint1': self.plots['predicted_velocities'].plot(pen=pg.mkPen('b', width=3), name='Joint 1'),
            'joint2': self.plots['predicted_velocities'].plot(pen=pg.mkPen('g', width=3), name='Joint 2')
        }
        
        # Predicted controls plot
        self.plots['predicted_controls'] = pg.PlotWidget(title='MPC Predicted Controls (All Steps)')
        self.plots['predicted_controls'].setLabel('left', 'Control (Nm)')
        self.plots['predicted_controls'].setLabel('bottom', 'Time Steps')
        self.plots['predicted_controls'].addLegend()
        self.plots['predicted_controls'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['predicted_controls'])
        self.predicted_controls_curves = {
            'joint1': self.plots['predicted_controls'].plot(pen=pg.mkPen('b', width=3), name='Joint 1'),
            'joint2': self.plots['predicted_controls'].plot(pen=pg.mkPen('g', width=3), name='Joint 2')
        }
        
        # ========================= Drone Plots =========================
        # Drone position tracking plot
        self.plots['drone_position'] = pg.PlotWidget(title='Drone Position')
        self.plots['drone_position'].setLabel('left', 'Position (m)')
        self.plots['drone_position'].setLabel('bottom', 'Time (s)')
        self.plots['drone_position'].addLegend()
        self.plots['drone_position'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['drone_position'])
        self.drone_position_curves = {
            'x_actual': self.plots['drone_position'].plot(pen=pg.mkPen('r', width=3), name='X Actual'),
            'y_actual': self.plots['drone_position'].plot(pen=pg.mkPen('g', width=3), name='Y Actual'),
            'z_actual': self.plots['drone_position'].plot(pen=pg.mkPen('b', width=3), name='Z Actual'),
            'x_target': self.plots['drone_position'].plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine), name='X Target'),
            'y_target': self.plots['drone_position'].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Y Target'),
            'z_target': self.plots['drone_position'].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Z Target')
        }
        
        # Drone velocity plot
        self.plots['drone_velocity'] = pg.PlotWidget(title='Drone Velocity')
        self.plots['drone_velocity'].setLabel('left', 'Velocity (m/s)')
        self.plots['drone_velocity'].setLabel('bottom', 'Time (s)')
        self.plots['drone_velocity'].addLegend()
        self.plots['drone_velocity'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['drone_velocity'])
        self.drone_velocity_curves = {
            'vx_actual': self.plots['drone_velocity'].plot(pen=pg.mkPen('r', width=3), name='Vx Actual'),
            'vy_actual': self.plots['drone_velocity'].plot(pen=pg.mkPen('g', width=3), name='Vy Actual'),
            'vz_actual': self.plots['drone_velocity'].plot(pen=pg.mkPen('b', width=3), name='Vz Actual'),
            'vx_target': self.plots['drone_velocity'].plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine), name='Vx Target'),
            'vy_target': self.plots['drone_velocity'].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Vy Target'),
            'vz_target': self.plots['drone_velocity'].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Vz Target')
        }
        
        # Drone orientation plot (quaternion)
        self.plots['drone_orientation'] = pg.PlotWidget(title='Drone Orientation (Quaternion)')
        self.plots['drone_orientation'].setLabel('left', 'Quaternion')
        self.plots['drone_orientation'].setLabel('bottom', 'Time (s)')
        self.plots['drone_orientation'].addLegend()
        self.plots['drone_orientation'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['drone_orientation'])
        self.drone_orientation_curves = {
            'roll': self.plots['drone_orientation'].plot(pen=pg.mkPen('r', width=3), name='Roll'),
            'pitch': self.plots['drone_orientation'].plot(pen=pg.mkPen('g', width=3), name='Pitch'),
            'yaw': self.plots['drone_orientation'].plot(pen=pg.mkPen('b', width=3), name='Yaw')
        }
        
        self.plots['drone_predicted_positions'] = pg.PlotWidget(title='Drone Predicted Positions')
        self.plots['drone_predicted_positions'].setLabel('left', 'Position (m)')
        self.plots['drone_predicted_positions'].setLabel('bottom', 'Time (s)')
        self.plots['drone_predicted_positions'].addLegend()
        self.plots['drone_predicted_positions'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['drone_predicted_positions'])
        self.drone_predicted_positions_curves = {
            'x_predicted': self.plots['drone_predicted_positions'].plot(pen=pg.mkPen('r', width=3), name='X Predicted'),
            'y_predicted': self.plots['drone_predicted_positions'].plot(pen=pg.mkPen('g', width=3), name='Y Predicted'),
            'z_predicted': self.plots['drone_predicted_positions'].plot(pen=pg.mkPen('b', width=3), name='Z Predicted'),
            'x_target': self.plots['drone_predicted_positions'].plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine), name='X Target'),
            'y_target': self.plots['drone_predicted_positions'].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Y Target'),
            'z_target': self.plots['drone_predicted_positions'].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Z Target')
        }
        
        self.plots['drone_predicted_velocities'] = pg.PlotWidget(title='Drone Predicted Velocities')
        
        self.plots['drone_predicted_velocities'].setLabel('left', 'Velocity (m/s)')
        self.plots['drone_predicted_velocities'].setLabel('bottom', 'Time (s)')
        self.plots['drone_predicted_velocities'].addLegend()
        self.plots['drone_predicted_velocities'].showGrid(x=True, y=True)
        setup_plot_style(self.plots['drone_predicted_velocities'])
        self.drone_predicted_velocities_curves = {
            'vx_predicted': self.plots['drone_predicted_velocities'].plot(pen=pg.mkPen('r', width=3), name='Vx Predicted'),
            'vy_predicted': self.plots['drone_predicted_velocities'].plot(pen=pg.mkPen('g', width=3), name='Vy Predicted'),
            'vz_predicted': self.plots['drone_predicted_velocities'].plot(pen=pg.mkPen('b', width=3), name='Vz Predicted')
        }
        
        # Store plot layout widget for dynamic updates
        self.plots_layout = plots_layout
        self.plots_widget = plots_widget
        
        # Initially show joint plots
        self.show_joint_plots()
        
        parent_layout.addWidget(plots_widget, 4)  # 增加图表区域占比
    
    def show_joint_plots(self):
        """Show joint-related plots"""
        # Clear all widgets from layout
        self.clear_plot_layout()
        
        # Add joint plots to grid (3x3 layout)
        self.plots_layout.addWidget(self.plots['position'], 0, 0)
        self.plots_layout.addWidget(self.plots['velocity'], 0, 1)
        self.plots_layout.addWidget(self.plots['control'], 0, 2)
    
        self.plots_layout.addWidget(self.plots['solve_time'], 1, 0)
        self.plots_layout.addWidget(self.plots['iterations'], 1, 1)
        self.plots_layout.addWidget(self.plots['cost'], 1, 2)
        
        self.plots_layout.addWidget(self.plots['predicted_positions'], 2, 0)
        self.plots_layout.addWidget(self.plots['predicted_velocities'], 2, 1)
        self.plots_layout.addWidget(self.plots['predicted_controls'], 2, 2)
    
    def show_drone_plots(self):
        """Show drone-related plots"""
        # Clear all widgets from layout
        self.clear_plot_layout()
        
        # Add drone plots to grid (3x3 layout)
        self.plots_layout.addWidget(self.plots['drone_position'], 0, 0)
        self.plots_layout.addWidget(self.plots['drone_velocity'], 0, 1)
        self.plots_layout.addWidget(self.plots['drone_orientation'], 0, 2)
    
        self.plots_layout.addWidget(self.plots['solve_time'], 1, 0)
        self.plots_layout.addWidget(self.plots['iterations'], 1, 1)
        self.plots_layout.addWidget(self.plots['cost'], 1, 2)
        
        self.plots_layout.addWidget(self.plots['drone_predicted_positions'], 2, 0)
        self.plots_layout.addWidget(self.plots['drone_predicted_velocities'], 2, 1)
        self.plots_layout.addWidget(self.plots['predicted_controls'], 2, 2)
    
    def clear_plot_layout(self):
        """Clear all widgets from the plots layout"""
        while self.plots_layout.count():
            child = self.plots_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
    
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
        
        # Robot model display
        self.robot_model_label = QLabel("Robot Model: Detecting...")
        display_layout.addWidget(self.robot_model_label)
        
        # Plot mode selection
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["Joint Mode", "Drone Mode"])
        # self.plot_mode_combo.setCurrentText("Drone Mode")
        self.plot_mode_combo.currentTextChanged.connect(self.change_plot_mode)
        display_layout.addWidget(QLabel("Plot Mode:"))
        display_layout.addWidget(self.plot_mode_combo)
        
        # Initially disable joint mode until we detect a robot with arm
        self.joint_mode_enabled = True
        # self.update_mode_availability()
        
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
        
        # Service Control Group
        service_group = QGroupBox("Service Control")
        service_layout = QVBoxLayout(service_group)
        
        self.open_gripper_btn = QPushButton("Open Gripper")
        self.open_gripper_btn.clicked.connect(self.open_gripper)
        service_layout.addWidget(self.open_gripper_btn)
        
        self.close_gripper_btn = QPushButton("Close Gripper")
        self.close_gripper_btn.clicked.connect(self.close_gripper)
        service_layout.addWidget(self.close_gripper_btn)
        
        self.reset_beer_btn = QPushButton("Reset Beer Position")
        self.reset_beer_btn.clicked.connect(self.reset_beer)
        service_layout.addWidget(self.reset_beer_btn)
        
        self.start_trajectory_btn = QPushButton("Start Trajectory")
        self.start_trajectory_btn.clicked.connect(self.start_trajectory)
        service_layout.addWidget(self.start_trajectory_btn)
        
        self.start_arm_test_btn = QPushButton("Start Arm Test")
        self.start_arm_test_btn.clicked.connect(self.start_arm_test)
        service_layout.addWidget(self.start_arm_test_btn)
        
        self.initialize_trajectory_btn = QPushButton("Initialize Trajectory")
        self.initialize_trajectory_btn.clicked.connect(self.initialize_trajectory)
        service_layout.addWidget(self.initialize_trajectory_btn)
        
        # Add L1 Control buttons
        self.start_l1_control_btn = QPushButton("Start L1 Control")
        self.start_l1_control_btn.clicked.connect(self.start_l1_control)
        service_layout.addWidget(self.start_l1_control_btn)
        
        self.stop_l1_control_btn = QPushButton("Stop L1 Control")
        self.stop_l1_control_btn.clicked.connect(self.stop_l1_control)
        service_layout.addWidget(self.stop_l1_control_btn)
        
        service_group.setLayout(service_layout)
        control_layout.addWidget(service_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Running")
        status_layout.addWidget(self.status_label)
        
        self.data_count_label = QLabel("Data Points: 0")
        status_layout.addWidget(self.data_count_label)
        
        self.ros_status_label = QLabel("ROS: Connected" if ROS_AVAILABLE else "ROS: Not Available")
        status_layout.addWidget(self.ros_status_label)
        
        self.service_status_label = QLabel("Service Status: Ready")
        status_layout.addWidget(self.service_status_label)
        
        control_layout.addWidget(status_group)
        
        control_layout.addStretch()
        parent_layout.addWidget(control_widget, 0)  # 固定宽度，不拉伸
        control_widget.setFixedWidth(300)  # 设置固定宽度为300像素
    
    def create_info_panel(self, parent_layout):
        """Create the information panel"""
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        # Current state information
        state_group = QGroupBox("Current State")
        state_layout = QVBoxLayout(state_group)
        
        # Joint state labels
        self.joint1_pos_label = QLabel("Joint 1 Position: 0.000")
        self.joint2_pos_label = QLabel("Joint 2 Position: 0.000")
        self.joint1_vel_label = QLabel("Joint 1 Velocity: 0.000")
        self.joint2_vel_label = QLabel("Joint 2 Velocity: 0.000")
        
        # Drone state labels
        self.drone_x_label = QLabel("Drone X: 0.000")
        self.drone_y_label = QLabel("Drone Y: 0.000")
        self.drone_z_label = QLabel("Drone Z: 0.000")
        self.drone_vx_label = QLabel("Drone Vx: 0.000")
        self.drone_vy_label = QLabel("Drone Vy: 0.000")
        self.drone_vz_label = QLabel("Drone Vz: 0.000")
        
        # Store labels for dynamic display
        self.joint_labels = [self.joint1_pos_label, self.joint2_pos_label, 
                           self.joint1_vel_label, self.joint2_vel_label]
        self.drone_labels = [self.drone_x_label, self.drone_y_label, self.drone_z_label,
                           self.drone_vx_label, self.drone_vy_label, self.drone_vz_label]
        
        # Initially show joint labels
        for label in self.joint_labels:
            state_layout.addWidget(label)
        
        info_layout.addWidget(state_group)
        self.state_group = state_group
        self.state_layout = state_layout
        
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
        parent_layout.addWidget(info_widget, 0)  # 固定宽度，不拉伸
        info_widget.setFixedWidth(300)  # 设置固定宽度为250像素
    
    def update_display(self, data):
        """Update display with new data"""
        try:
            # Update robot model info if changed
            self.update_robot_model_info(data)
            
            # Add new data
            current_time = data.get('timestamp', time.time())
            self.time_data.append(current_time)
            
            # Joint data
            if data.get('joint_states'):
                self.position_data.append(data['joint_states']['position'])
                self.velocity_data.append(data['joint_states']['velocity'])
            else:
                self.position_data.append(np.array([0.0, 0.0]))
                self.velocity_data.append(np.array([0.0, 0.0]))
            
            # Drone data
            if data.get('drone_states'):
                self.drone_position_data.append(data['drone_states']['position'])
                self.drone_velocity_data.append(data['drone_states']['velocity'])
                self.drone_orientation_data.append(data['drone_states']['orientation'])
                self.drone_angular_velocity_data.append(data['drone_states']['angular_velocity'])
            else:
                self.drone_position_data.append(np.array([0.0, 0.0, 0.0]))
                self.drone_velocity_data.append(np.array([0.0, 0.0, 0.0]))
                self.drone_orientation_data.append(np.array([0.0, 0.0, 0.0, 1.0]))
                self.drone_angular_velocity_data.append(np.array([0.0, 0.0, 0.0]))
            
            # Drone target data
            if data.get('drone_target_state'):
                self.drone_target_position_data.append(data['drone_target_state']['position'])
                self.drone_target_velocity_data.append(data['drone_target_state']['velocity'])
            else:
                self.drone_target_position_data.append(np.array([0.0, 0.0, 0.0]))
                self.drone_target_velocity_data.append(np.array([0.0, 0.0, 0.0]))
            
            if data.get('control_input') is not None:
                control = data['control_input']
                # Pad or truncate control input to ensure consistent storage
                if len(control) >= 2:
                    self.control_data.append(control[:2])  # Store first 2 for joint display
                else:
                    self.control_data.append(np.array([0.0, 0.0]))
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
                self.drone_position_data = self.drone_position_data[-self.max_data_points:]
                self.drone_velocity_data = self.drone_velocity_data[-self.max_data_points:]
                self.drone_target_position_data = self.drone_target_position_data[-self.max_data_points:]
                self.drone_target_velocity_data = self.drone_target_velocity_data[-self.max_data_points:]
                self.drone_orientation_data = self.drone_orientation_data[-self.max_data_points:]
                self.drone_angular_velocity_data = self.drone_angular_velocity_data[-self.max_data_points:]
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
            if self.display_mode == 'joint':
                self.update_joint_info_labels(data)
            else:  # drone mode
                self.update_drone_info_labels(data)
        except Exception as e:
            print(f"Error updating info labels: {e}")
            
    def update_joint_info_labels(self, data):
        """Update joint information labels"""
        try:
            if data.get('joint_states'):
                pos = data['joint_states']['position']
                vel = data['joint_states']['velocity']
                
                self.joint1_pos_label.setText(f"Joint 1 Position: {pos[0]:.3f}")
                self.joint2_pos_label.setText(f"Joint 2 Position: {pos[1]:.3f}")
                self.joint1_vel_label.setText(f"Joint 1 Velocity: {vel[0]:.3f}")
                self.joint2_vel_label.setText(f"Joint 2 Velocity: {vel[1]:.3f}")
                
                # Update common labels
                self.update_common_info_labels(data)
                
                # Calculate joint errors
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
            print(f"Error updating joint info labels: {e}")
    
    def update_drone_info_labels(self, data):
        """Update drone information labels"""
        try:
            if data.get('drone_states'):
                pos = data['drone_states']['position']
                vel = data['drone_states']['velocity']
                
                self.drone_x_label.setText(f"Drone X: {pos[0]:.3f}")
                self.drone_y_label.setText(f"Drone Y: {pos[1]:.3f}")
                self.drone_z_label.setText(f"Drone Z: {pos[2]:.3f}")
                self.drone_vx_label.setText(f"Drone Vx: {vel[0]:.3f}")
                self.drone_vy_label.setText(f"Drone Vy: {vel[1]:.3f}")
                self.drone_vz_label.setText(f"Drone Vz: {vel[2]:.3f}")
                
                # Update common labels
                self.update_common_info_labels(data)
                
                # Calculate drone errors
                if data.get('drone_target_state') is not None:
                    target_pos = data['drone_target_state']['position']
                    target_vel = data['drone_target_state']['velocity']
                    
                    pos_error_x = pos[0] - target_pos[0]
                    pos_error_y = pos[1] - target_pos[1]
                    pos_error_z = pos[2] - target_pos[2]
                    vel_error_x = vel[0] - target_vel[0]
                    vel_error_y = vel[1] - target_vel[1]
                    vel_error_z = vel[2] - target_vel[2]
                    
                    self.pos_error1_label.setText(f"Position Error X: {pos_error_x:.3f}")
                    self.pos_error2_label.setText(f"Position Error Y: {pos_error_y:.3f}")
                    self.vel_error1_label.setText(f"Velocity Error X: {vel_error_x:.3f}")
                    self.vel_error2_label.setText(f"Velocity Error Y: {vel_error_y:.3f}")
                    
        except Exception as e:
            print(f"Error updating drone info labels: {e}")
    
    def update_common_info_labels(self, data):
        """Update common information labels (cost, solve time, etc.)"""
        try:
            if data.get('target_state') is not None and self.display_mode == 'joint':
                target = data['target_state']
                self.target1_label.setText(f"Target 1: {target[0]:.3f}")
                self.target2_label.setText(f"Target 2: {target[1]:.3f}")
            elif data.get('drone_target_state') is not None and self.display_mode == 'drone':
                target_pos = data['drone_target_state']['position']
                self.target1_label.setText(f"Target X: {target_pos[0]:.3f}")
                self.target2_label.setText(f"Target Y: {target_pos[1]:.3f}")
            
            if data.get('control_input') is not None:
                control = data['control_input']
                if self.display_mode == 'joint' and len(control) >= 8:
                    # For s500_uam, show arm joint controls (indices 6:8)
                    self.control1_label.setText(f"Joint Control 1: {control[6]:.3f}")
                    self.control2_label.setText(f"Joint Control 2: {control[7]:.3f}")
                    control_mag = np.linalg.norm(control[6:8])
                    self.control_mag_label.setText(f"Joint Control Magnitude: {control_mag:.3f}")
                elif self.display_mode == 'drone':
                    # Show first few control inputs for drone
                    self.control1_label.setText(f"Control 1: {control[0]:.3f}")
                    if len(control) > 1:
                        self.control2_label.setText(f"Control 2: {control[1]:.3f}")
                    else:
                        self.control2_label.setText(f"Control 2: 0.000")
                    control_mag = np.linalg.norm(control[:min(4, len(control))])
                    self.control_mag_label.setText(f"Control Magnitude: {control_mag:.3f}")
                else:
                    # Fallback
                    self.control1_label.setText(f"Control 1: {control[0]:.3f}")
                    if len(control) > 1:
                        self.control2_label.setText(f"Control 2: {control[1]:.3f}")
                    else:
                        self.control2_label.setText(f"Control 2: 0.000")
                    control_mag = np.linalg.norm(control)
                    self.control_mag_label.setText(f"Control Magnitude: {control_mag:.3f}")
            
            self.cost_label.setText(f"Current Cost: {data.get('mpc_cost', 0.0):.6f}")
            self.solve_time_label.setText(f"Solve Time: {data.get('solve_time', 0.0)*1000:.1f} ms")
            self.iterations_label.setText(f"Iterations: {data.get('iterations', 0)}")
            
        except Exception as e:
            print(f"Error updating common info labels: {e}")
    
    def update_state_labels_display(self):
        """Update which state labels are displayed based on current mode"""
        try:
            # Clear current labels
            for label in self.joint_labels + self.drone_labels:
                label.setParent(None)
            
            # Add appropriate labels based on mode
            if self.display_mode == 'joint':
                for label in self.joint_labels:
                    self.state_layout.addWidget(label)
            else:  # drone mode
                for label in self.drone_labels:
                    self.state_layout.addWidget(label)
                    
        except Exception as e:
            print(f"Error updating state labels display: {e}")
    
    def update_plots(self):
        """Update all plots"""
        try:
            if len(self.time_data) > 1:
                time_array = np.array(self.time_data)
                cost_array = np.array(self.cost_data)
                solve_time_array = np.array(self.solve_time_data)
                iterations_array = np.array(self.iterations_data)
                
                if self.display_mode == 'joint':
                    self.update_joint_plots(time_array)
                else:  # drone mode
                    self.update_drone_plots(time_array)
                
                # Update common plots (cost, solve time, iterations)
                self.cost_curve.setData(time_array, cost_array)
                self.solve_time_curve.setData(time_array, solve_time_array * 1000)  # Convert to ms
                self.iterations_curve.setData(time_array, iterations_array)
                
                # Update predicted states and controls plots with latest data
                if hasattr(self.ros_subscriber, 'predicted_states') or hasattr(self.ros_subscriber, 'predicted_controls'):
                    predicted_states = getattr(self.ros_subscriber, 'predicted_states', None)
                    predicted_controls = getattr(self.ros_subscriber, 'predicted_controls', None)
                    self.update_predicted_plots(predicted_states, predicted_controls)
                
                # Auto-scale if enabled
                if self.auto_scale_check.isChecked():
                    for plot in self.plots.values():
                        plot.autoRange()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def update_joint_plots(self, time_array):
        """Update joint-specific plots"""
        try:
            position_array = np.array(self.position_data)
            velocity_array = np.array(self.velocity_data)
            control_array = np.array(self.control_data)
            target_array = np.array(self.target_data)
            
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
            
        except Exception as e:
            print(f"Error updating joint plots: {e}")
    
    def update_drone_plots(self, time_array):
        """Update drone-specific plots"""
        try:
            drone_position_array = np.array(self.drone_position_data)
            drone_velocity_array = np.array(self.drone_velocity_data)
            drone_target_position_array = np.array(self.drone_target_position_data)
            drone_target_velocity_array = np.array(self.drone_target_velocity_data)
            drone_orientation_array = np.array(self.drone_orientation_data)
            
            # Update drone position plot
            self.drone_position_curves['x_actual'].setData(time_array, drone_position_array[:, 0])
            self.drone_position_curves['y_actual'].setData(time_array, drone_position_array[:, 1])
            self.drone_position_curves['z_actual'].setData(time_array, drone_position_array[:, 2])
            self.drone_position_curves['x_target'].setData(time_array, drone_target_position_array[:, 0])
            self.drone_position_curves['y_target'].setData(time_array, drone_target_position_array[:, 1])
            self.drone_position_curves['z_target'].setData(time_array, drone_target_position_array[:, 2])
            
            # Update drone velocity plot
            self.drone_velocity_curves['vx_actual'].setData(time_array, drone_velocity_array[:, 0])
            self.drone_velocity_curves['vy_actual'].setData(time_array, drone_velocity_array[:, 1])
            self.drone_velocity_curves['vz_actual'].setData(time_array, drone_velocity_array[:, 2])
            self.drone_velocity_curves['vx_target'].setData(time_array, drone_target_velocity_array[:, 0])
            self.drone_velocity_curves['vy_target'].setData(time_array, drone_target_velocity_array[:, 1])
            self.drone_velocity_curves['vz_target'].setData(time_array, drone_target_velocity_array[:, 2])
            
            # Update drone orientation plot
            self.drone_orientation_curves['roll'].setData(time_array, drone_orientation_array[:, 0])
            self.drone_orientation_curves['pitch'].setData(time_array, drone_orientation_array[:, 1])
            self.drone_orientation_curves['yaw'].setData(time_array, drone_orientation_array[:, 2])
            
        except Exception as e:
            print(f"Error updating drone plots: {e}")
    
    def update_predicted_plots(self, predicted_states=None, predicted_controls=None):
        """Update predicted states and controls plots"""
        try:
            # Use passed data or fallback to subscriber data
            if predicted_states is None and hasattr(self.ros_subscriber, 'predicted_states'):
                predicted_states = self.ros_subscriber.predicted_states
            if predicted_controls is None and hasattr(self.ros_subscriber, 'predicted_controls'):
                predicted_controls = self.ros_subscriber.predicted_controls
                
            # Update predicted positions plot (all steps)
            if predicted_states is not None and len(predicted_states) > 0:
                steps = np.arange(len(predicted_states))
                
                if self.display_mode == 'joint':
                    # States format for joints: state[7:9] for joint positions (s500_uam)
                    if self.current_robot_model == 's500_uam':
                        self.predicted_positions_curves['joint1'].setData(steps, predicted_states[:, 7])
                        self.predicted_positions_curves['joint2'].setData(steps, predicted_states[:, 8])
                        
                        # Add target positions (horizontal lines)
                        if len(self.target_data) > 0:
                            current_target = self.target_data[-1]  # Get latest target
                            target_steps = np.arange(len(predicted_states))
                            target_joint1 = np.full(len(predicted_states), current_target[0])
                            target_joint2 = np.full(len(predicted_states), current_target[1])
                            self.predicted_positions_curves['joint1_target'].setData(target_steps, target_joint1)
                            self.predicted_positions_curves['joint2_target'].setData(target_steps, target_joint2)
                        
                        # Update predicted velocities for joints (s500_uam: state[15:17])
                        self.predicted_velocities_curves['joint1'].setData(steps, predicted_states[:, 15])
                        self.predicted_velocities_curves['joint2'].setData(steps, predicted_states[:, 16])
                
                else:  # drone mode
                    # States format for drone: state[0:3] for position
                    self.drone_predicted_positions_curves['x_predicted'].setData(steps, predicted_states[:, 0])  # X position
                    self.drone_predicted_positions_curves['y_predicted'].setData(steps, predicted_states[:, 1])  # Y position
                    self.drone_predicted_positions_curves['z_predicted'].setData(steps, predicted_states[:, 2])  # Z position
                    self.drone_predicted_positions_curves['x_target'].setData(steps, self.drone_target_position_data[:, 0])
                    self.drone_predicted_positions_curves['y_target'].setData(steps, self.drone_target_position_data[:, 1])
                    self.drone_predicted_positions_curves['z_target'].setData(steps, self.drone_target_position_data[:, 2])
                        
                    # Update predicted velocities for drone
                    if self.current_robot_model == 's500':
                        # s500: 13D state [pos(3), quat(4), vel(3), angular_vel(3)]
                        self.drone_predicted_velocities_curves['vx_predicted'].setData(steps, predicted_states[:, 7])   # Vx
                        self.drone_predicted_velocities_curves['vy_predicted'].setData(steps, predicted_states[:, 8])   # Vy
                        self.drone_predicted_velocities_curves['vz_predicted'].setData(steps, predicted_states[:, 9])   # Vz
                    elif self.current_robot_model == 's500_uam':
                        # s500_uam: 17D state [pos(3), quat(4), joint_pos(2), vel(3), angular_vel(3), joint_vel(2)]
                        self.drone_predicted_velocities_curves['vx_predicted'].setData(steps, predicted_states[:, 9])   # Vx
                        self.drone_predicted_velocities_curves['vy_predicted'].setData(steps, predicted_states[:, 10])  # Vy
                        self.drone_predicted_velocities_curves['vz_predicted'].setData(steps, predicted_states[:, 11])  # Vz
        
            # Update predicted controls plot (all steps)
            if predicted_controls is not None and len(predicted_controls) > 0:
                steps = np.arange(len(predicted_controls))
                if self.display_mode == 'joint' and self.current_robot_model == 's500_uam':
                    # For s500_uam, joint controls are the last 2 elements
                    self.predicted_controls_curves['joint1'].setData(steps, predicted_controls[:, -2])
                    self.predicted_controls_curves['joint2'].setData(steps, predicted_controls[:, -1])
                    
        except Exception as e:
            print(f"Error updating predicted plots: {e}")
    
    def update_robot_model_info(self, data):
        """Update robot model information and UI availability"""
        robot_model = data.get('robot_model')
        has_arm = data.get('has_arm', False)
        
        if robot_model and robot_model != self.current_robot_model:
            self.current_robot_model = robot_model
            self.current_has_arm = has_arm
            
            # Update robot model display
            state_dim = data.get('state_dim', 'Unknown')
            control_dim = data.get('control_dim', 'Unknown')
            arm_status = "with 2DOF arm" if has_arm else "no arm"
            self.robot_model_label.setText(f"Robot: {robot_model} ({state_dim}D state, {control_dim}D control, {arm_status})")
            
            # Update mode availability
            # self.update_mode_availability()
            
            print(f"Robot model updated: {robot_model}, has_arm: {has_arm}")
    
    def update_mode_availability(self):
        """Update which modes are available based on current robot model"""
        # Always enable drone mode
        drone_mode_index = self.plot_mode_combo.findText("Drone Mode")
        if drone_mode_index >= 0:
            self.plot_mode_combo.model().item(drone_mode_index).setEnabled(True)
        
        # Enable/disable joint mode based on whether robot has arm
        joint_mode_index = self.plot_mode_combo.findText("Joint Mode")
        if joint_mode_index >= 0:
            self.plot_mode_combo.model().item(joint_mode_index).setEnabled(self.current_has_arm)
            
        # If current mode is joint but robot has no arm, switch to drone mode
        if self.display_mode == 'joint' and not self.current_has_arm:
            self.plot_mode_combo.setCurrentText("Drone Mode")
            self.change_plot_mode("Drone Mode")
    
    def change_plot_mode(self, mode_text):
        """Change the plot display mode"""
        if mode_text == "Joint Mode" and self.current_has_arm:
            self.display_mode = 'joint'
            self.show_joint_plots()
        else:  # Drone Mode (default for both s500 and s500_uam)
            self.display_mode = 'drone'
            self.show_drone_plots()
        
        # Update state labels display
        self.update_state_labels_display()
        
        print(f"Plot mode changed to: {self.display_mode}")
    
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
        self.drone_position_data.clear()
        self.drone_velocity_data.clear()
        self.drone_target_position_data.clear()
        self.drone_target_velocity_data.clear()
        self.drone_orientation_data.clear()
        self.drone_angular_velocity_data.clear()
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
    
    # Service Control Methods
    def open_gripper(self):
        """Open the gripper"""
        success, message = self.service_handler.call_service('open_gripper')
        if success:
            print("Gripper opened successfully")
            self.service_status_label.setText("Service Status: Gripper Opened")
        else:
            print(f"Failed to open gripper: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
            
    def close_gripper(self):
        """Close the gripper"""
        success, message = self.service_handler.call_service('close_gripper')
        if success:
            print("Gripper closed successfully")
            self.service_status_label.setText("Service Status: Gripper Closed")
        else:
            print(f"Failed to close gripper: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
            
    def reset_beer(self):
        """Reset the beer position"""
        success, message = self.service_handler.call_service('reset_beer')
        if success:
            print("Beer position reset successfully")
            self.service_status_label.setText("Service Status: Beer Reset")
        else:
            print(f"Failed to reset beer position: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
            
    def start_trajectory(self):
        """Start the trajectory"""
        success, message = self.service_handler.call_service('start_trajectory')
        if success:
            print("Trajectory started successfully")
            self.service_status_label.setText("Service Status: Trajectory Started")
        else:
            print(f"Failed to start trajectory: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds

    def start_arm_test(self):
        """Start the arm test"""
        success, message = self.service_handler.call_service('start_arm_test')
        if success:
            print("Arm test started successfully")
            self.service_status_label.setText("Service Status: Arm Test Started")
        else:
            print(f"Failed to start arm test: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
            
    def initialize_trajectory(self):
        """Initialize the trajectory"""
        success, message = self.service_handler.call_service('initialize_trajectory')
        if success:
            print("Trajectory initialized successfully")
            self.service_status_label.setText("Service Status: Trajectory Initialized")
        else:
            print(f"Failed to initialize trajectory: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
            
    def start_l1_control(self):
        """Start L1 adaptive control"""
        success, message = self.service_handler.call_service('start_l1_control')
        if success:
            print("L1 control started successfully")
            self.service_status_label.setText("Service Status: L1 Control Started")
        else:
            print(f"Failed to start L1 control: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
            
    def stop_l1_control(self):
        """Stop L1 adaptive control"""
        success, message = self.service_handler.call_service('stop_l1_control')
        if success:
            print("L1 control stopped successfully")
            self.service_status_label.setText("Service Status: L1 Control Stopped")
        else:
            print(f"Failed to stop L1 control: {message}")
            self.service_status_label.setText(f"Service Status: Failed - {message}")
        self.service_status_timer.start(3000)  # Clear status after 3 seconds
    
    def clear_service_status(self):
        """Clear the service status after a delay"""
        self.service_status_label.setText("Service Status: Ready")
    
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
    print("\nService Control Available:")
    print("- /open_gripper")
    print("- /close_gripper")
    print("- /reset_beer")
    print("- /start_trajectory")
    print("- /initialize_trajectory")
    print("- /start_l1_control")
    print("- /stop_l1_control")
    print("- /start_arm_test")
    
    # Start the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 