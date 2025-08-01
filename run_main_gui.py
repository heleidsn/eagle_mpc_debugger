#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Main GUI for eagle_mpc_debugger
'''

import sys
import os
import rospy
import subprocess
import yaml
import signal
import psutil
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QFormLayout, QGroupBox, QPushButton, 
                            QLabel, QComboBox, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QTextEdit, QMessageBox, QLineEdit)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon, QPixmap
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from eagle_mpc_msgs.msg import MpcState
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Point

# Add necessary directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # Add current directory
sys.path.append(os.path.join(current_dir, 'scripts'))  # Add scripts directory
sys.path.append(os.path.join(current_dir, 'utils'))  # Add utils directory

class SystemController:
    def __init__(self):
        self.launch_process = None
        self.planning_process = None
        self.gepetto_process = None
        
    def _stop_process_safely(self, process, process_name="Process"):
        """Safely stop a process with timeout and fallback to force kill"""
        if process:
            try:
                # First try graceful termination
                process.terminate()
                
                # Wait for process to terminate (max 5 seconds)
                try:
                    process.wait(timeout=5)
                    return True, f"{process_name} stopped successfully"
                except subprocess.TimeoutExpired:
                    # If graceful termination fails, force kill
                    process.kill()
                    process.wait(timeout=2)
                    return True, f"{process_name} force killed"
            except Exception as e:
                return False, f"Failed to stop {process_name}: {str(e)}"
        return True, f"{process_name} was not running"
        
    def launch_system(self, config):
        """Launch the system with the given configuration"""
        try:
            # Generate launch file
            launch_file = self._generate_launch_file(config)
            
            # Launch the system
            cmd = f"roslaunch eagle_mpc_debugger {launch_file}"
            self.launch_process = subprocess.Popen(cmd, shell=True)
            return True, "System launched successfully"
        except Exception as e:
            return False, str(e)
            
    def stop_system(self):
        """Stop the running system"""
        try:
            success, message = self._stop_process_safely(self.launch_process, "System")
            self.launch_process = None
            return success, message
        except Exception as e:
            return False, str(e)
            
    def _generate_launch_file(self, config):
        """Generate launch file based on configuration"""
        # Implementation depends on your launch file structure
        return "eagle_mpc_debugger.launch"
        
    def launch_planning(self, config):
        """Launch the planning process"""
        try:
            # Set up environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{current_dir}:{env.get('PYTHONPATH', '')}"
            
            # Launch the planning script with command line arguments
            cmd = f"python3 {os.path.join(current_dir, 'run_planning.py')} --robot {config['robot_name']} --trajectory {config['trajectory_name']} --dt {config['dt_traj_opt']} {'--use-squash' if config['use_squash'] else ''}"
            self.planning_process = subprocess.Popen(cmd, shell=True, env=env)
            return True, "Planning process started"
        except Exception as e:
            return False, str(e)
            
    def launch_planning_different_dt(self, config):
        """Launch the planning process with different dt values"""
        try:
            # Set up environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{current_dir}:{env.get('PYTHONPATH', '')}"
            
            # Launch the planning script with command line arguments
            cmd = f"python3 {os.path.join(current_dir, 'scripts/run_planning_different_dt.py')} --robot {config['robot_name']} --trajectory {config['trajectory_name']} --dt {config['dt_traj_opt']} {'--use-squash' if config['use_squash'] else ''}"
            self.planning_process = subprocess.Popen(cmd, shell=True, env=env)
            return True, "Planning process with different DT started"
        except Exception as e:
            return False, str(e)
            
    def stop_planning(self):
        """Stop the running planning process"""
        try:
            success, message = self._stop_process_safely(self.planning_process, "Planning process")
            self.planning_process = None
            
            # Also kill any related Python processes
            kill_cmd = "pkill -9 -f 'python.*run_planning'"
            subprocess.run(kill_cmd, shell=True, stderr=subprocess.PIPE)
            
            return success, message
        except Exception as e:
            return False, str(e)

    def launch_gepetto_gui(self):
        """Launch gepetto-gui in a separate process"""
        try:
            if self.gepetto_process:
                return False, "Gepetto-GUI is already running"
            
            cmd = "gepetto-gui"
            self.gepetto_process = subprocess.Popen(cmd, shell=True)
            return True, "Gepetto-GUI launched successfully"
        except Exception as e:
            return False, str(e)

    def stop_gepetto_gui(self):
        """Stop the running gepetto-gui process and all related processes"""
        try:
            # Kill the main gepetto-gui process if it exists
            success, message = self._stop_process_safely(self.gepetto_process, "Gepetto-GUI")
            self.gepetto_process = None

            # Kill all gepetto-related processes
            kill_cmd = "killall -9 gepetto-gui gepetto-viewer gepetto-viewer-server gepetto-viewer-corba"
            subprocess.run(kill_cmd, shell=True, stderr=subprocess.PIPE)

            # Kill any remaining Python processes that might be running gepetto
            kill_python_cmd = "pkill -9 -f 'python.*gepetto'"
            subprocess.run(kill_python_cmd, shell=True, stderr=subprocess.PIPE)

            # Wait a moment to ensure processes are terminated
            import time
            time.sleep(1)

            return True, "Gepetto-GUI and all related processes stopped successfully"
        except Exception as e:
            return False, str(e)

class ROSServiceHandler:
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

class ROSCoreController:
    def __init__(self):
        self.roscore_process = None
        self.roscore_port = 11311  # Default ROS port
        
    def start_roscore(self):
        """Start roscore process"""
        try:
            if self.is_roscore_running():
                return True, "roscore is already running"
                
            # Start roscore in a new process
            cmd = f"roscore -p {self.roscore_port}"
            self.roscore_process = subprocess.Popen(cmd, shell=True, 
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE)
            return True, "roscore started successfully"
        except Exception as e:
            return False, str(e)
            
    def stop_roscore(self):
        """Stop roscore process"""
        try:
            if self.roscore_process:
                # Kill the roscore process and its children
                try:
                    parent = psutil.Process(self.roscore_process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.terminate()
                    parent.terminate()
                    
                    # Wait for processes to terminate
                    for child in children:
                        try:
                            child.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            child.kill()
                    try:
                        parent.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        parent.kill()
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process already terminated or access denied
                    
                self.roscore_process = None
                
            # Also kill any remaining roscore processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'roscore' in proc.info['name'].lower():
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            return True, "roscore stopped successfully"
        except Exception as e:
            return False, str(e)
            
    def is_roscore_running(self):
        """Check if roscore is running"""
        try:
            # Check if port 11311 is in use
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', self.roscore_port))
            sock.close()
            return result == 0
        except:
            return False

class EagleMPCDebuggerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open Aerial Manipulator Control GUI")
        self.setGeometry(100, 100, 1600, 800)
        
        # Set window icon
        icon_path = os.path.join(current_dir, 'resources', 'icon', 'aerial_manipulator_sw.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize controllers
        self.system_controller = SystemController()
        self.service_handler = ROSServiceHandler()
        self.roscore_controller = ROSCoreController()
        
        # Define trajectory mapping for each robot
        self.robot_trajectories = {
            "s500_uam": ["catch_vicon", "catch_vicon_real", "displacement", "arm_test"],
            "s500": ["hover", "displacement", "displacement_real"],
            "hexacopter370_flying_arm_3": ["eagle_catch_nc"]
        }
        
        # Create main layout with three sections
        self.main_layout = QHBoxLayout()
        
        # Create three panels
        self.left_panel = self.create_left_panel()
        self.center_panel = self.create_center_panel()
        self.right_panel = self.create_right_panel()
        
        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel, 1)
        self.main_layout.addWidget(self.center_panel, 1)
        self.main_layout.addWidget(self.right_panel, 1)
        
        # Set main window widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # Update every 100ms
        
        # Initialize state variables
        self.current_state = None
        self.arm_state = None
        
        # Initialize ROS node only if roscore is running
        if self.roscore_controller.is_roscore_running():
            self.init_ros_node()
        
    def init_ros_node(self):
        """Initialize ROS node and subscribers"""
        try:
            if not rospy.core.is_initialized():
                rospy.init_node('eagle_mpc_gui', anonymous=False)
            self.mpc_state_sub = rospy.Subscriber("/mpc/state", MpcState, self.mpc_state_callback)
            self.arm_state_sub = rospy.Subscriber("/arm_controller/joint_states", JointState, self.arm_state_callback)
            return True
        except Exception as e:
            self.log_message(f"Failed to initialize ROS node: {str(e)}")
            return False
            
    def create_left_panel(self):
        """Create the left panel with icon, task configuration and planning"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Icon Display Group
        icon_group = QGroupBox()
        icon_layout = QVBoxLayout()
        self.icon_label = QLabel()  # Make icon_label a class member
        icon_path = os.path.join(current_dir, 'resources', 'icon', 'aerial_manipulator_sw.png')
        if os.path.exists(icon_path):
            icon_pixmap = QPixmap(icon_path)
            scaled_pixmap = icon_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
            self.icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(self.icon_label)
        icon_group.setLayout(icon_layout)
        layout.addWidget(icon_group)
        
        # Task Configuration Group
        task_config_group = QGroupBox("Task Configuration")
        task_config_layout = QFormLayout()
        
        # Robot selection
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(list(self.robot_trajectories.keys()))
        self.robot_combo.currentTextChanged.connect(self.update_trajectory_options)
        task_config_layout.addRow("Robot:", self.robot_combo)
        
        # Trajectory selection
        self.trajectory_combo = QComboBox()
        self.update_trajectory_options(self.robot_combo.currentText())
        task_config_layout.addRow("Trajectory:", self.trajectory_combo)
        
        # DT selection
        self.dt_spin = QSpinBox()
        self.dt_spin.setRange(10, 100)
        self.dt_spin.setValue(50)
        self.dt_spin.setSingleStep(5)
        task_config_layout.addRow("DT (ms):", self.dt_spin)
        
        # Use squash option
        self.use_squash_check = QCheckBox("Use Squash")
        self.use_squash_check.setChecked(True)
        task_config_layout.addRow("Squash:", self.use_squash_check)
        
        # Gepetto visualization checkbox
        self.gepetto_vis_check = QCheckBox("Gepetto Visualization")
        self.gepetto_vis_check.setChecked(True)
        task_config_layout.addRow("Gepetto Visualization:", self.gepetto_vis_check)
        
        # Save results checkbox
        self.save_results_check = QCheckBox("Save Results")
        self.save_results_check.setChecked(False)
        task_config_layout.addRow("Save Results:", self.save_results_check)
        
        task_config_group.setLayout(task_config_layout)
        layout.addWidget(task_config_group)
        
        # Planning Control Group
        planning_group = QGroupBox("Planning Control")
        planning_layout = QVBoxLayout()
        
        self.plan_btn = QPushButton("Run Planning")
        self.plan_btn.clicked.connect(self.run_planning)
        planning_layout.addWidget(self.plan_btn)
        
        self.plan_dt_btn = QPushButton("Run Planning (Different DT)")
        self.plan_dt_btn.clicked.connect(self.run_planning_different_dt)
        planning_layout.addWidget(self.plan_dt_btn)
        
        self.stop_plan_btn = QPushButton("Stop Planning")
        self.stop_plan_btn.clicked.connect(self.stop_planning)
        planning_layout.addWidget(self.stop_plan_btn)
        
        self.launch_gepetto_btn = QPushButton("Launch Gepetto-GUI")
        self.launch_gepetto_btn.clicked.connect(self.launch_gepetto_gui)
        planning_layout.addWidget(self.launch_gepetto_btn)
        
        self.close_gepetto_btn = QPushButton("Close Gepetto-GUI")
        self.close_gepetto_btn.clicked.connect(self.close_gepetto_gui)
        planning_layout.addWidget(self.close_gepetto_btn)
        
        planning_group.setLayout(planning_layout)
        layout.addWidget(planning_group)
        
        panel.setLayout(layout)
        return panel
        
    def create_center_panel(self):
        """Create the center panel with controller configuration, simulation and service controls"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Controller Configuration Group
        controller_config_group = QGroupBox("Controller Configuration")
        controller_config_layout = QFormLayout()
        
        # Control mode selection
        self.control_mode_combo = QComboBox()
        self.control_mode_combo.addItems(["MPC", "Geometric", "PX4",])
        controller_config_layout.addRow("Control Mode:", self.control_mode_combo)
        
        # Arm control options
        self.arm_enabled_check = QCheckBox("Enable Arm")
        self.arm_enabled_check.setChecked(True)
        controller_config_layout.addRow("Arm Control:", self.arm_enabled_check)
        
        self.arm_control_mode_combo = QComboBox()
        self.arm_control_mode_combo.addItems(["position", "position_velocity", "position_velocity_effort", "effort"])
        controller_config_layout.addRow("Arm Control Mode:", self.arm_control_mode_combo)
        
        # Simulation mode
        self.simulation_check = QCheckBox("Simulation Mode")
        self.simulation_check.setChecked(True)
        controller_config_layout.addRow("Simulation:", self.simulation_check)
        
        # L1 control parameters
        self.l1_version_combo = QComboBox()
        self.l1_version_combo.addItems(["v1", "v2", "v3"])
        self.l1_version_combo.setCurrentText("v2")
        controller_config_layout.addRow("L1 Version:", self.l1_version_combo)
        
        self.As_coef_spin = QDoubleSpinBox()
        self.As_coef_spin.setRange(-10.0, 10.0)
        self.As_coef_spin.setValue(-1.0)
        controller_config_layout.addRow("As Coefficient:", self.As_coef_spin)
        
        self.filter_time_spin = QDoubleSpinBox()
        self.filter_time_spin.setRange(0.1, 5.0)
        self.filter_time_spin.setValue(0.4)
        controller_config_layout.addRow("Filter Time Constant:", self.filter_time_spin)

        # Add MPC controller button
        self.start_mpc_btn = QPushButton("Start MPC Controller")
        self.start_mpc_btn.clicked.connect(self.start_mpc_controller)
        controller_config_layout.addRow(self.start_mpc_btn)
        
        # Add Stop controller button
        self.stop_mpc_btn = QPushButton("Stop MPC Controller")
        self.stop_mpc_btn.clicked.connect(self.stop_mpc_controller)
        controller_config_layout.addRow(self.stop_mpc_btn)
        
        controller_config_group.setLayout(controller_config_layout)
        
        
        # Simulation Control Group
        simulation_group = QGroupBox("Simulation Control")
        simulation_layout = QVBoxLayout()
        
        # Add world selection
        world_selection_layout = QHBoxLayout()
        world_label = QLabel("Gazebo World:")
        self.world_combo = QComboBox()
        self.world_combo.addItems(["empty", "table_beer", "table_beer_with_stand"])
        self.world_combo.setCurrentText("table_beer_with_stand")
        world_selection_layout.addWidget(world_label)
        world_selection_layout.addWidget(self.world_combo)
        simulation_layout.addLayout(world_selection_layout)
        
        # Add camera selection
        camera_selection_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["True", "False"])
        self.camera_combo.setCurrentText("False")
        camera_selection_layout.addWidget(camera_label)
        camera_selection_layout.addWidget(self.camera_combo)
        simulation_layout.addLayout(camera_selection_layout)
        
        # Add arm control mode selection
        arm_control_mode_selection_layout = QHBoxLayout()
        arm_control_mode_label = QLabel("Arm Control Mode:")
        self.arm_control_mode_combo = QComboBox()
        self.arm_control_mode_combo.addItems(["position", "velocity", "effort"])
        self.arm_control_mode_combo.setCurrentText("effort")
        arm_control_mode_selection_layout.addWidget(arm_control_mode_label)
        arm_control_mode_selection_layout.addWidget(self.arm_control_mode_combo)
        simulation_layout.addLayout(arm_control_mode_selection_layout)
        
        # ROSCore Control
        self.start_roscore_btn = QPushButton("Start ROSCore")
        self.start_roscore_btn.clicked.connect(self.start_roscore)
        simulation_layout.addWidget(self.start_roscore_btn)
        
        self.stop_roscore_btn = QPushButton("Stop ROSCore")
        self.stop_roscore_btn.clicked.connect(self.stop_roscore)
        simulation_layout.addWidget(self.stop_roscore_btn)
        
        # Simulation Control
        self.launch_simulation_btn = QPushButton("Launch Simulation")
        self.launch_simulation_btn.clicked.connect(self.launch_simulation)
        simulation_layout.addWidget(self.launch_simulation_btn)
        
        self.stop_simulation_btn = QPushButton("Stop Simulation")
        self.stop_simulation_btn.clicked.connect(self.stop_simulation)
        simulation_layout.addWidget(self.stop_simulation_btn)

        # Add QGroundControl button
        self.start_qgc_btn = QPushButton("Start QGroundControl")
        self.start_qgc_btn.clicked.connect(self.start_qgroundcontrol)
        simulation_layout.addWidget(self.start_qgc_btn)

        # Add PlotJuggler button
        self.start_plotjuggler_btn = QPushButton("Start PlotJuggler")
        self.start_plotjuggler_btn.clicked.connect(self.start_plotjuggler)
        simulation_layout.addWidget(self.start_plotjuggler_btn)
        
        simulation_group.setLayout(simulation_layout)
        
        
        # Service Control Group
        service_group = QGroupBox("Service Control")
        service_layout = QVBoxLayout()
        
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
        
        service_group.setLayout(service_layout)
        
        layout.addWidget(simulation_group)
        
        layout.addWidget(controller_config_group)
        
        layout.addWidget(service_group)
        
        panel.setLayout(layout)
        return panel
        
    def create_right_panel(self):
        """Create the right panel with system status and logging"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # System Status Group
        status_group = QGroupBox("System Status")
        status_layout = QFormLayout()
        
        self.system_status = QLabel("Stopped")
        status_layout.addRow("Status:", self.system_status)
        
        self.simulation_status = QLabel("Stopped")
        status_layout.addRow("Simulation:", self.simulation_status)
        
        self.control_mode_status = QLabel("None")
        status_layout.addRow("Control Mode:", self.control_mode_status)
        
        self.arm_status = QLabel("Disabled")
        status_layout.addRow("Arm Status:", self.arm_status)
        
        self.gripper_status = QLabel("Open")
        status_layout.addRow("Gripper Status:", self.gripper_status)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # MPC Status Group
        mpc_group = QGroupBox("MPC Status")
        mpc_layout = QFormLayout()
        
        self.mpc_status = QLabel("Not Running")
        mpc_layout.addRow("MPC Status:", self.mpc_status)
        
        self.mpc_iterations = QLabel("0")
        mpc_layout.addRow("Iterations:", self.mpc_iterations)
        
        self.mpc_solve_time = QLabel("0.0")
        mpc_layout.addRow("Solve Time:", self.mpc_solve_time)
        
        mpc_group.setLayout(mpc_layout)
        layout.addWidget(mpc_group)
        
        # Log Display
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        panel.setLayout(layout)
        return panel
        
    def update_status(self):
        """Update the status panel with current system state"""
        # Update roscore status
        roscore_running = self.roscore_controller.is_roscore_running()
        self.system_status.setText("Running" if roscore_running else "Stopped")
        
        if self.current_state:
            self.control_mode_status.setText(self.current_state.control_mode)
            self.mpc_status.setText(f"Running (Cost: {self.current_state.mpc_final_cost:.2f})")
            
        if self.arm_state:
            self.arm_status.setText("Enabled" if self.use_squash_check.isChecked() else "Disabled")
            
    def log_message(self, message):
        """Add a message to the log display"""
        self.log_text.append(message)
        
    def run_planning(self):
        """Run the standard planning process"""
        config = {
            'robot_name': self.robot_combo.currentText(),
            'trajectory_name': self.trajectory_combo.currentText(),
            'dt_traj_opt': self.dt_spin.value(),
            'use_squash': self.use_squash_check.isChecked()
        }
        
        success, message = self.system_controller.launch_planning(config)
        if success:
            self.log_message("Planning process started...")
        else:
            self.log_message(f"Failed to start planning: {message}")
            QMessageBox.critical(self, "Error", message)
            
    def run_planning_different_dt(self):
        """Run the planning process with different dt values"""
        config = {
            'robot_name': self.robot_combo.currentText(),
            'trajectory_name': self.trajectory_combo.currentText(),
            'dt_traj_opt': self.dt_spin.value(),
            'use_squash': self.use_squash_check.isChecked()
        }
        
        success, message = self.system_controller.launch_planning_different_dt(config)
        if success:
            self.log_message("Planning process with different DT started...")
        else:
            self.log_message(f"Failed to start planning: {message}")
            QMessageBox.critical(self, "Error", message)
            
    def stop_planning(self):
        """Stop the running planning process"""
        try:
            success, message = self.system_controller.stop_planning()
            if success:
                self.log_message("Planning process stopped")
            else:
                self.log_message(f"Failed to stop planning: {message}")
                
            # Also kill any related Python processes that might be running planning
            kill_cmd = "pkill -9 -f 'python.*run_planning'"
            subprocess.run(kill_cmd, shell=True, stderr=subprocess.PIPE)
            
            # Kill any remaining ROS nodes related to planning
            kill_ros_cmd = "rosnode kill /run_planning"
            subprocess.run(kill_ros_cmd, shell=True, stderr=subprocess.PIPE)
            
        except Exception as e:
            error_msg = f"Failed to stop planning: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def launch_system(self):
        """Launch the system with current configuration"""
        config = {
            'robot_name': self.robot_combo.currentText(),
            'trajectory_name': self.trajectory_combo.currentText(),
            'dt_traj_opt': self.dt_spin.value(),
            'use_squash': self.use_squash_check.isChecked(),
            'use_simulation': self.simulation_check.isChecked(),
            'control_mode': self.control_mode_combo.currentText(),
            'arm_enabled': self.arm_enabled_check.isChecked(),
            'arm_control_mode': self.arm_control_mode_combo.currentText(),
            'max_thrust': 32.2656,
            'max_angular_velocity': 2.0944,
            'min_thrust_cmd': 0.0,
            'max_thrust_cmd': 1.0,
            'l1_version': self.l1_version_combo.currentText(),
            'As_coef': self.As_coef_spin.value(),
            'filter_time_constant': self.filter_time_spin.value()
        }
        
        success, message = self.system_controller.launch_system(config)
        if success:
            self.system_status.setText("Running")
            self.log_message("System launched successfully")
        else:
            self.log_message(f"Failed to launch system: {message}")
            QMessageBox.critical(self, "Error", message)
            
    def stop_system(self):
        """Stop the running system"""
        success, message = self.system_controller.stop_system()
        if success:
            self.system_status.setText("Stopped")
            self.log_message("System stopped successfully")
        else:
            self.log_message(f"Failed to stop system: {message}")
            QMessageBox.critical(self, "Error", message)
            
    def open_gripper(self):
        """Open the gripper"""
        success, message = self.service_handler.call_service('open_gripper')
        if success:
            self.gripper_status.setText("Open")
            self.log_message("Gripper opened")
        else:
            self.log_message(f"Failed to open gripper: {message}")
            
    def close_gripper(self):
        """Close the gripper"""
        success, message = self.service_handler.call_service('close_gripper')
        if success:
            self.gripper_status.setText("Closed")
            self.log_message("Gripper closed")
        else:
            self.log_message(f"Failed to close gripper: {message}")
            
    def reset_beer(self):
        """Reset the beer position"""
        success, message = self.service_handler.call_service('reset_beer')
        if success:
            self.log_message("Beer position reset")
        else:
            self.log_message(f"Failed to reset beer position: {message}")
            
    def start_trajectory(self):
        """Start the trajectory"""
        success, message = self.service_handler.call_service('start_trajectory')
        if success:
            self.log_message("Trajectory started")
        else:
            self.log_message(f"Failed to start trajectory: {message}")

    def start_arm_test(self):
        """Start the arm test"""
        success, message = self.service_handler.call_service('start_arm_test')
        if success:
            self.log_message("Arm test started")
        else:
            self.log_message(f"Failed to start arm test: {message}")
            
    def initialize_trajectory(self):
        """Initialize the trajectory"""
        success, message = self.service_handler.call_service('initialize_trajectory')
        if success:
            self.log_message("Trajectory initialized")
        else:
            self.log_message(f"Failed to initialize trajectory: {message}")
            
    def mpc_state_callback(self, msg):
        """Callback for MPC state messages"""
        self.current_state = msg
        
    def arm_state_callback(self, msg):
        """Callback for arm state messages"""
        self.arm_state = msg

    def start_roscore(self):
        """Start roscore"""
        success, message = self.roscore_controller.start_roscore()
        if success:
            self.log_message("roscore started successfully")
            # Initialize ROS node after roscore is started
            if self.init_ros_node():
                self.log_message("ROS node initialized successfully")
        else:
            self.log_message(f"Failed to start roscore: {message}")
            QMessageBox.critical(self, "Error", message)
            
    def stop_roscore(self):
        """Stop roscore"""
        success, message = self.roscore_controller.stop_roscore()
        if success:
            self.log_message("roscore stopped successfully")
            # Reset ROS-related state
            self.current_state = None
            self.arm_state = None
        else:
            self.log_message(f"Failed to stop roscore: {message}")
            QMessageBox.critical(self, "Error", message)

    def update_trajectory_options(self, robot_name):
        """Update trajectory options based on selected robot"""
        self.trajectory_combo.clear()
        if robot_name in self.robot_trajectories:
            self.trajectory_combo.addItems(self.robot_trajectories[robot_name])
            
        # Update robot icon based on selection
        icon_name = f"{robot_name}.png"
        icon_path = os.path.join(current_dir, 'resources', 'icon', icon_name)
        
        # If specific robot icon doesn't exist, use default icon
        if not os.path.exists(icon_path):
            icon_path = os.path.join(current_dir, 'resources', 'icon', 'aerial_manipulator_sw.png')
            
        if os.path.exists(icon_path):
            icon_pixmap = QPixmap(icon_path)
            scaled_pixmap = icon_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)

    def launch_simulation(self):
        """Launch the simulation environment"""
        try:
            # Set up environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{current_dir}:{env.get('PYTHONPATH', '')}"
            
            # Get selected robot name and world
            robot_name = self.robot_combo.currentText()
            world_name = self.world_combo.currentText()
            use_camera = self.camera_combo.currentText()
            arm_control_mode = self.arm_control_mode_combo.currentText()
            
            # Select appropriate launch file based on robot name
            if robot_name == "s500_uam":
                launch_file = "s500_uam_sitl.launch"
            elif robot_name == "s500":
                launch_file = "s500_sitl.launch"
            elif robot_name == "hexacopter370_flying_arm_3":
                launch_file = "hexacopter370_flying_arm_3_sitl.launch"
            else:
                raise ValueError(f"No simulation launch file available for robot: {robot_name}")
            
            # Launch the simulation script with world parameter
            cmd = f"roslaunch eagle_mpc_debugger {launch_file} world_name:={world_name} use_camera:={use_camera} arm_control_mode:={arm_control_mode}"
            self.simulation_process = subprocess.Popen(cmd, shell=True, env=env)
            self.log_message(f"Simulation environment launched successfully for {robot_name} with world: {world_name}")
            self.simulation_status.setText("Running")
        except Exception as e:
            error_msg = f"Failed to launch simulation: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def stop_simulation(self):
        """Stop the simulation environment"""
        try:
            # Kill the main simulation process if it exists
            if hasattr(self, 'simulation_process') and self.simulation_process:
                # First try graceful termination
                self.simulation_process.terminate()
                
                # Wait for process to terminate (max 5 seconds)
                try:
                    self.simulation_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # If graceful termination fails, force kill
                    self.simulation_process.kill()
                    self.simulation_process.wait(timeout=2)
                
                self.simulation_process = None

            # Kill all related ROS nodes
            kill_cmd = "rosnode kill /gazebo /gazebo_gui /rosout /rosmaster /rviz /rviz_gui /robot_state_publisher /joint_state_publisher /joint_state_publisher_gui /move_group /controller_spawner /controller_manager /robot_state_publisher /joint_state_publisher /joint_state_publisher_gui /move_group /controller_spawner /controller_manager /groundtruth_pub"
            subprocess.run(kill_cmd, shell=True, stderr=subprocess.PIPE)

            # Kill all gazebo-related processes
            kill_gazebo_cmd = "killall -9 gazebo gzserver gzclient gazebo_gui"
            subprocess.run(kill_gazebo_cmd, shell=True, stderr=subprocess.PIPE)

            # Kill all rviz-related processes
            kill_rviz_cmd = "killall -9 rviz rviz_gui"
            subprocess.run(kill_rviz_cmd, shell=True, stderr=subprocess.PIPE)

            # Kill all ros-related processes
            kill_ros_cmd = "killall -9 rosmaster rosout roscore"
            subprocess.run(kill_ros_cmd, shell=True, stderr=subprocess.PIPE)

            # Kill any remaining python processes that might be running the simulation
            kill_python_cmd = "pkill -9 -f 'python.*simulation'"
            subprocess.run(kill_python_cmd, shell=True, stderr=subprocess.PIPE)

            # Clean up any remaining ROS master
            cleanup_cmd = "killall -9 rosmaster"
            subprocess.run(cleanup_cmd, shell=True, stderr=subprocess.PIPE)

            # Wait a moment to ensure processes are terminated
            import time
            time.sleep(1)

            self.log_message("Simulation environment stopped successfully")
            self.simulation_status.setText("Stopped")
        except Exception as e:
            error_msg = f"Failed to stop simulation: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def launch_gepetto_gui(self):
        """Launch gepetto-gui in a separate process"""
        try:
            cmd = "gepetto-gui"
            subprocess.Popen(cmd, shell=True)
            self.log_message("Gepetto-GUI launched successfully")
        except Exception as e:
            self.log_message(f"Failed to launch Gepetto-GUI: {str(e)}")
            QMessageBox.warning(self, "Warning", f"Failed to launch Gepetto-GUI: {str(e)}")

    def close_gepetto_gui(self):
        """Close the running gepetto-gui process"""
        success, message = self.system_controller.stop_gepetto_gui()
        if success:
            self.log_message("Gepetto-GUI closed successfully")
        else:
            self.log_message(f"Failed to close Gepetto-GUI: {message}")
            QMessageBox.warning(self, "Warning", f"Failed to close Gepetto-GUI: {message}")

    def start_mpc_controller(self):
        """Start the MPC controller"""
        try:
            # Set up environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{current_dir}:{env.get('PYTHONPATH', '')}"
            
            # Launch the MPC controller script
            cmd = f"python3 {os.path.join(current_dir, 'trajectory_publisher_mpc.py')}"
            self.mpc_controller_process = subprocess.Popen(cmd, shell=True, env=env)
            self.log_message("MPC controller started successfully")
        except Exception as e:
            error_msg = f"Failed to start MPC controller: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def stop_mpc_controller(self):
        """Stop the MPC controller"""
        try:
            if hasattr(self, 'mpc_controller_process') and self.mpc_controller_process:
                # First try graceful termination
                self.mpc_controller_process.terminate()
                
                # Wait for process to terminate (max 5 seconds)
                try:
                    self.mpc_controller_process.wait(timeout=5)
                    self.log_message("MPC controller stopped successfully")
                except subprocess.TimeoutExpired:
                    # If graceful termination fails, force kill
                    self.log_message("Graceful termination failed, forcing kill...")
                    self.mpc_controller_process.kill()
                    self.mpc_controller_process.wait(timeout=2)
                    self.log_message("MPC controller force killed")
                
                self.mpc_controller_process = None
                
                # Also kill any related Python processes that might be running MPC
                kill_cmd = "pkill -9 -f 'python.*trajectory_publisher_mpc'"
                subprocess.run(kill_cmd, shell=True, stderr=subprocess.PIPE)
                
                # Kill any remaining ROS nodes related to MPC
                kill_ros_cmd = "rosnode kill /trajectory_publisher_mpc"
                subprocess.run(kill_ros_cmd, shell=True, stderr=subprocess.PIPE)
                
        except Exception as e:
            error_msg = f"Failed to stop MPC controller: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def start_qgroundcontrol(self):
        """Start QGroundControl"""
        try:
            qgc_path = os.path.join(current_dir, 'resources', 'qgroundcontrol', 'QGroundControl.AppImage')
            if not os.path.exists(qgc_path):
                raise FileNotFoundError(f"QGroundControl not found at: {qgc_path}")
            
            # Make the AppImage executable if it's not already
            if not os.access(qgc_path, os.X_OK):
                os.chmod(qgc_path, os.stat(qgc_path).st_mode | 0o111)
            
            # Launch QGroundControl
            cmd = f"{qgc_path}"
            self.qgc_process = subprocess.Popen(cmd, shell=True)
            self.log_message("QGroundControl started successfully")
        except Exception as e:
            error_msg = f"Failed to start QGroundControl: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def stop_qgroundcontrol(self):
        """Stop QGroundControl"""
        try:
            if hasattr(self, 'qgc_process') and self.qgc_process:
                self.qgc_process.terminate()
                self.qgc_process = None
                self.log_message("QGroundControl stopped successfully")
        except Exception as e:
            error_msg = f"Failed to stop QGroundControl: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def start_plotjuggler(self):
        """Start PlotJuggler"""
        try:
            # Launch PlotJuggler using rosrun
            cmd = "rosrun plotjuggler plotjuggler"
            self.plotjuggler_process = subprocess.Popen(cmd, shell=True)
            self.log_message("PlotJuggler started successfully")
        except Exception as e:
            error_msg = f"Failed to start PlotJuggler: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def stop_plotjuggler(self):
        """Stop PlotJuggler"""
        try:
            if hasattr(self, 'plotjuggler_process') and self.plotjuggler_process:
                self.plotjuggler_process.terminate()
                self.plotjuggler_process = None
                self.log_message("PlotJuggler stopped successfully")
        except Exception as e:
            error_msg = f"Failed to stop PlotJuggler: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EagleMPCDebuggerGUI()
    gui.show()
    sys.exit(app.exec_()) 