#!/usr/bin/env python3
'''
Author: Lei He
Date: 2025-09-11
Description: Numeric simulation script for MPC controller with precise timing control
Simulation step: 1ms, Control step: 20ms
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import yaml
from datetime import datetime
from pathlib import Path

class PIDController:
    """Simple PID controller for angular velocity to torque conversion"""
    def __init__(self, kp, ki, kd, dt=0.001):
        self.kp = kp
        self.ki = ki  
        self.kd = kd
        self.dt = dt
        
        self.error_sum = 0.0
        self.last_error = 0.0
        self.initialized = False
        
    def reset(self):
        """Reset PID controller state"""
        self.error_sum = 0.0
        self.last_error = 0.0
        self.initialized = False
        
    def update(self, setpoint, current_value):
        """Update PID controller and return control output"""
        error = setpoint - current_value
        
        # Clamp error to prevent overflow
        error = np.clip(error, -100.0, 100.0)
        
        # Integral term with windup protection
        self.error_sum += error * self.dt
        self.error_sum = np.clip(self.error_sum, -10.0, 10.0)  # Anti-windup
        
        # Derivative term
        if self.initialized:
            error_rate = (error - self.last_error) / self.dt
            # Clamp derivative to prevent overflow
            error_rate = np.clip(error_rate, -1000.0, 1000.0)
        else:
            error_rate = 0.0
            self.initialized = True
            
        # PID output
        output = self.kp * error + self.ki * self.error_sum + self.kd * error_rate
        
        # Clamp output to prevent overflow
        output = np.clip(output, -1000.0, 1000.0)
        
        self.last_error = error
        return output

# Try to import crocoddyl and related packages (only needed for MPC mode)
try:
    import crocoddyl
    import pinocchio as pin
    import example_robot_data
    CROCODDYL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Crocoddyl not available: {e}")
    print("MPC mode will not be available.")
    CROCODDYL_AVAILABLE = False

# Import utility modules
from utils.create_problem import get_opt_traj, create_mpc_controller, create_state_update_model, create_state_update_model_quadrotor
from utils.u_convert import thrustToForceTorqueAll

# Import L1 controllers
from l1_control.L1AdaptiveController_v1 import L1AdaptiveController_V1
from l1_control.L1AdaptiveController_v2 import L1AdaptiveControllerAll
from l1_control.L1AdaptiveController_v3 import L1AdaptiveControllerRefactored

class NumericSimulator:
    def __init__(self, config_file='config.yaml'):
        """
        Initialize numeric simulator for MPC controller
        
        Args:
            config_file: Path to configuration file
        """
        if not CROCODDYL_AVAILABLE:
            raise ValueError("Crocoddyl is required for numeric simulation")
        
        # Load configuration
        self.load_config(config_file)
        
        # Timing parameters (now loaded from config)
        self.control_interval = int(self.control_dt / self.simulation_dt)
        
        # Set numpy print options
        np.set_printoptions(precision=4, suppress=True)
        
        # Initialize robot model and dynamics
        self.init_robot_model()
        
        # Initialize MPC controller
        self.init_mpc_controller()
        
        # Initialize L1 controller if enabled
        if self.enable_l1_control:
            self.init_l1_controller()
        
        # Initialize simulation state update model
        self.init_simulation_model()
        
        # Initialize data storage
        self.init_data_storage()
        
        print(f"Numeric simulator initialized:")
        print(f"  Simulation dt: {self.simulation_dt*1000:.1f} ms")
        print(f"  Control dt: {self.control_dt*1000:.1f} ms")
        print(f"  Control interval: {self.control_interval} simulation steps")
        print(f"  Robot: {self.robot_name}")
        print(f"  Trajectory: {self.trajectory_name}")
        print(f"  Simulation actuation mode: {self.simulation_actuation_mode}")
        print(f"  Control dimension: {self.control_dim}")
        
        # Validate actuation mode compatibility
        if self.control_dim != 2 and self.simulation_actuation_mode not in ['full', 'quad', 'thrust_angular_velocity']:
            print(f"Warning: Unrecognized simulation_actuation_mode '{self.simulation_actuation_mode}' for full system")
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        # Default configuration
        default_config = {
            'robot_name': 's500_uam',
            'trajectory_name': 'catch_vicon',
            'dt_traj_opt': 50,  # ms
            'use_squash': True,
            'yaml_path': 'config/yaml',
            'simulation_time': 10.0,  # seconds
            'initial_state': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.2, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Timing parameters
            'simulation_dt': 0.001,  # seconds - simulation time step (1ms)
            'control_dt': 0.02,      # seconds - control time step (20ms)
            'enable_disturbance': False,
            'disturbance_start_time': 5.0,
            'disturbance_force': [0.0, 0.0, 0.0],
            'disturbance_torque': [0.0, 0.0, 0.0],
            'enable_joint_sin_disturbance': False,
            'joint_sin_amplitude': [0.1, 0.05],  # amplitude for joint1, joint2 (Nm)
            'joint_sin_frequency': [2.0, 1.5],   # frequency for joint1, joint2 (Hz)
            'enable_noise': False,
            'state_noise_std': 0.001,
            'save_results': True,
            'plot_results': True,
            'results_dir': 'results/numeric_sim',
            # L1 adaptive control configuration
            'enable_l1_control': False,
            'l1_version': 'v2',  # v1, v2, v3
            'l1_start_time': 2.0,  # seconds - when to start L1 control
            'l1_as_coef': -0.01,
            'l1_filter_time_constant': [0.3, 0.3, 0.3],
            # Control delay configuration
            'control_delay': 0.0,  # seconds - delay in control application
            'delay_use_hover_control': True,  # Use hover control during delay instead of zero
            # Simulation comparison modes
            'simulation_modes': ['mpc_only'],  # options: 'mpc_only', 'l1_only', 'mpc_l1', 'comparison'
            # Simulation actuation model
            'simulation_actuation_mode': 'thrust_angular_velocity',  # options: 'full', 'quad', 'thrust_angular_velocity'
            # PID controller parameters for thrust + angular velocity mode
            'pid_pitch_kp': 0.5,
            'pid_pitch_ki': 0.0,
            'pid_pitch_kd': 0.0,
        }
        
        # Try to load config file if it exists
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                print(f"Loaded configuration from: {config_file}")
        else:
            print(f"Config file {config_file} not found, using default configuration")
        
        # Set configuration as attributes
        for key, value in default_config.items():
            setattr(self, key, value)
        
        # Initialize control delay buffer
        self.init_control_delay_buffer()
        
        # Initialize PID controller for thrust + angular velocity mode
        if self.simulation_actuation_mode == 'thrust_angular_velocity':
            self.pitch_pid = PIDController(
                kp=self.pid_pitch_kp,
                ki=self.pid_pitch_ki, 
                kd=self.pid_pitch_kd,
                dt=self.simulation_dt
            )
            print(f"PID controller initialized with Kp={self.pid_pitch_kp}, Ki={self.pid_pitch_ki}, Kd={self.pid_pitch_kd}")
        
        # Calculate tracking error recording interval based on trajectory optimization dt
        self.tracking_error_interval = max(1, int(self.dt_traj_opt / self.simulation_dt / 1000))
        print(f"Tracking error recording interval: every {self.tracking_error_interval} simulation steps ({self.dt_traj_opt}ms)")
    
    def init_control_delay_buffer(self):
        """Initialize control delay buffer to store past control commands"""
        # Calculate delay steps based on control delay and simulation timestep
        self.delay_steps = int(self.control_delay / self.simulation_dt)
        print(f"Control delay: {self.control_delay}s ({self.delay_steps} simulation steps)")
        
        # Initialize control command buffer (FIFO queue)
        self.control_buffer = []
        
        # Store metadata for debugging
        self.delayed_control_data = []  # For plotting/analysis
    
    def get_delayed_control(self, new_control, current_time):
        """
        Get control command with specified delay
        
        Args:
            new_control: Newly computed control command
            current_time: Current simulation time
            
        Returns:
            delayed_control: Control command to apply (delayed)
        """
        # Add new control to buffer with timestamp
        control_entry = {
            'control': new_control.copy() if new_control is not None else None,
            'time': current_time,
            'delayed_control': None  # Will be filled when applied
        }
        self.control_buffer.append(control_entry)
        
        # If no delay, return the new control immediately
        if self.delay_steps <= 0:
            control_entry['delayed_control'] = new_control
            return new_control.copy() if new_control is not None else None
        
        # If we have enough buffered controls, get the delayed one
        if len(self.control_buffer) > self.delay_steps:
            delayed_entry = self.control_buffer[-(self.delay_steps + 1)]
            delayed_control = delayed_entry['control']
            delayed_entry['delayed_control'] = delayed_control
            
            # Store for analysis
            self.delayed_control_data.append({
                'time': current_time,
                'original_control': new_control.copy() if new_control is not None else None,
                'delayed_control': delayed_control.copy() if delayed_control is not None else None,
                'delay_steps': self.delay_steps
            })
            
            return delayed_control.copy() if delayed_control is not None else None
        else:
            # Not enough data in buffer yet - during the delay period
            # We should use zero control (or hover control) instead of repeating previous
            control_entry['delayed_control'] = None
            return None
    
    def init_robot_model(self):
        """Initialize robot model from URDF"""
        # Get URDF path based on robot name
        if self.robot_name == 's500_uam':
            self.urdf_path = 'models/urdf/s500_uam_simple.urdf'
        elif self.robot_name == 's500':
            self.urdf_path = 'models/urdf/s500_simple.urdf'
        else:
            # Try to find a matching URDF file
            possible_paths = [
                f'models/urdf/{self.robot_name}.urdf',
                f'models/urdf/{self.robot_name}_simple.urdf',
                f'models/urdf/{self.robot_name}_arm_effort.urdf'
            ]
            self.urdf_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.urdf_path = path
                    break
            
            if self.urdf_path is None:
                raise ValueError(f"Unknown robot name: {self.robot_name}. Available URDF files: {os.listdir('models/urdf/')}")
        
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        # Load robot model
        self.robot = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
        self.robot_state = crocoddyl.StateMultibody(self.robot)
        self.robot_data = self.robot.createData()

        # State and control dimensions
        self.state_dim = self.robot.nq + self.robot.nv
        self.control_dim = self.robot.nv
        
        print(f"Robot model loaded: {self.robot_name}")
        print(f"  nq (position DOF): {self.robot.nq}")
        print(f"  nv (velocity DOF): {self.robot.nv}")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Control dimension: {self.control_dim}")
    
    def init_mpc_controller(self):
        """Initialize MPC controller"""
        try:
            # Load trajectory and create MPC controller
            self.trajectory, self.traj_state_ref, _, self.trajectory_obj = get_opt_traj(
                self.robot_name,
                self.trajectory_name,
                self.dt_traj_opt,
                self.use_squash,
                self.yaml_path
            )
            
            # Store reference control inputs from trajectory optimization
            self.traj_u_ref = self.trajectory.us_squash if hasattr(self.trajectory, 'us_squash') else []
            print(f"Reference control trajectory loaded: {len(self.traj_u_ref)} steps")
            if len(self.traj_u_ref) > 0:
                print(f"  Control dimension: {self.traj_u_ref[0].shape}")
                print(f"  First control sample: {self.traj_u_ref[0][:3]}")  # Show first 3 elements
            
            # Create MPC controller
            # mpc_name = "rail"
            mpc_name = "rail_with_ref_control"
            mpc_yaml = f'{self.yaml_path}/mpc/{self.robot_name}_mpc.yaml'
            
            self.mpc_controller = create_mpc_controller(
                mpc_name,
                self.trajectory_obj,
                self.traj_state_ref,
                self.traj_u_ref,
                self.dt_traj_opt,
                mpc_yaml
            )
            
            # Get platform parameters for force/torque conversion
            self.platform_params = self.trajectory_obj.platform_params
            
            print(f"MPC controller initialized")
            print(f"  Trajectory duration: {len(self.traj_state_ref)} steps")
            print(f"  MPC horizon: {len(self.mpc_controller.solver.xs)}")
            
        except Exception as e:
            print(f"Error initializing MPC controller: {e}")
            raise
    
    def init_l1_controller(self):
        """Initialize L1 adaptive controller"""
        try:
            dt_controller = 1.0 / (1.0 / self.control_dt)  # Control frequency
            robot_model = self.trajectory_obj.robot_model  # Use the full robot model from trajectory
            
            if self.l1_version == 'v1':
                self.l1_controller = L1AdaptiveController_V1(
                    dt=dt_controller,
                    robot_model=robot_model,
                    As_coef=self.l1_as_coef,
                    filter_time_constant=self.l1_filter_time_constant[0]
                )
            elif self.l1_version == 'v2':
                self.l1_controller = L1AdaptiveControllerAll(
                    dt=dt_controller,
                    robot_model=robot_model,
                    As_coef=self.l1_as_coef,
                    filter_time_constant=self.l1_filter_time_constant
                )
            elif self.l1_version == 'v3':
                self.l1_controller = L1AdaptiveControllerRefactored(
                    dt=dt_controller,
                    robot_model=robot_model,
                    As_coef=self.l1_as_coef,
                    filter_time_constant=self.l1_filter_time_constant[0]
                )
            else:
                raise ValueError(f"Unknown L1 controller version: {self.l1_version}")
            
            # Initialize L1 controller
            self.l1_controller.init_controller()
            
            print(f"L1 controller ({self.l1_version}) initialized")
            print(f"  Start time: {self.l1_start_time} seconds")
            print(f"  As coefficient: {self.l1_as_coef}")
            print(f"  Filter time constants: {self.l1_filter_time_constant}")
            
        except Exception as e:
            print(f"Error initializing L1 controller: {e}")
            raise
    
    def init_simulation_model(self):
        """Initialize simulation state update model"""
        try:
            # For arm-only simulation, use the robot model directly loaded from URDF
            # instead of the full trajectory model which includes the drone
            if self.control_dim == 2:  # Arm-only model
                print(f"Using arm-only robot model for simulation")
                
                # Create simple state update model using ABA
                self.use_simple_dynamics = True
                print(f"Simulation model initialized with simple dynamics and {self.simulation_dt*1000:.1f} ms timestep")
            else:
                # For full system, choose between full and quad actuation models
                if self.simulation_actuation_mode == 'full':
                    # Use full actuation model (original behavior)
                    self.sim_update_model = create_state_update_model(
                        self.trajectory_obj.robot_model_path, 
                        int(self.simulation_dt * 1000)  # Convert to ms
                    )
                    self.sim_update_data = self.sim_update_model.createData()
                    self.use_simple_dynamics = False
                    print(f"Simulation model initialized with full actuation crocoddyl dynamics and {self.simulation_dt*1000:.1f} ms timestep")
                    print(f"Control dimension: {self.control_dim} (force/torque format)")
                    
                elif self.simulation_actuation_mode == 'quad':
                    # Use quadrotor-specific actuation model (4 motors direct thrust control)
                    self.sim_update_model, self.sim_update_data = create_state_update_model_quadrotor(
                        self.trajectory_obj.robot_model.copy(), 
                        self.trajectory_obj.platform_params, 
                        int(self.simulation_dt * 1000)  # Convert to ms
                    )
                    self.use_simple_dynamics = False
                    print(f"Simulation model initialized with quadrotor actuation crocoddyl dynamics and {self.simulation_dt*1000:.1f} ms timestep")
                    print(f"Using 4-motor direct thrust control")
                    print(f"Control dimension: {self.control_dim} (direct motor thrust format)")
                    
                elif self.simulation_actuation_mode == 'thrust_angular_velocity':
                    # Use exactly the same model as 'full' mode since we're using force/torque interface
                    self.sim_update_model = create_state_update_model(
                        self.trajectory_obj.robot_model_path, 
                        int(self.simulation_dt * 1000)  # Convert to ms
                    )
                    self.sim_update_data = self.sim_update_model.createData()
                    self.use_simple_dynamics = False
                    print(f"Simulation model initialized with thrust+angular_velocity mode using force/torque interface")
                    print(f"Control dimension: {self.control_dim} (force/torque format)")
                else:
                    raise ValueError(f"Unknown simulation_actuation_mode: {self.simulation_actuation_mode}. Expected 'full', 'quad', or 'thrust_angular_velocity'")
            
        except Exception as e:
            print(f"Error initializing simulation model: {e}")
            raise
    
    def init_data_storage(self):
        """Initialize data storage for results"""
        self.time_data = []
        self.state_data = []
        self.control_data = []
        self.mpc_solve_times = []
        self.mpc_costs = []
        self.mpc_iterations = []
        self.trajectory_index_data = []
        
        # Tracking error data storage
        self.tracking_error_data = []
        
        # L1 controller data storage
        self.l1_u_ad_data = []
        self.l1_u_baseline_data = []
        self.l1_z_tilde_data = []
        self.l1_sig_hat_data = []
        self.l1_z_hat_data = []
        self.l1_compute_times = []
        
        # Disturbance data storage
        self.disturbance_data = []
        
        # Pitch control data storage (for thrust_angular_velocity mode)
        self.pitch_control_data = []
        
        # Track MPC updates
        self.last_mpc_update_step = -1
        self.current_control = None
        self.current_mpc_index = 0
        self.current_l1_control = None
        self.l1_start_step = int(self.l1_start_time / self.simulation_dt) if hasattr(self, 'l1_start_time') else 0
    
    def get_reference_state(self, current_time):
        """Get reference state based on current time"""
        # Calculate trajectory index based on time
        traj_index = int(current_time * 1000 / self.dt_traj_opt)
        traj_index = min(traj_index, len(self.traj_state_ref) - 1)
        
        return self.traj_state_ref[traj_index], traj_index
    
    def get_reference_control(self, current_time):
        """Get reference control input based on current time"""
        if not self.traj_u_ref:
            return np.zeros(self.control_dim)
        
        # Calculate trajectory index based on time
        traj_index = int(current_time * 1000 / self.dt_traj_opt)
        traj_index = min(traj_index, len(self.traj_u_ref) - 1)
        
        return self.traj_u_ref[traj_index]
    
    def calculate_tracking_errors(self, current_state, current_time):
        """Calculate tracking errors for current state"""
        # Get reference state
        ref_state, _ = self.get_reference_state(current_time)
        
        # Calculate position error
        position_error = current_state[:3] - ref_state[:3]
        position_error_norm = np.linalg.norm(position_error)
        
        # Convert quaternions to Euler angles for attitude error calculation
        from scipy.spatial.transform import Rotation as R
        
        try:
            # Current state quaternion
            current_quat = current_state[3:7]
            current_euler = R.from_quat(current_quat).as_euler('xyz', degrees=True)
            
            # Reference state quaternion
            ref_quat = ref_state[3:7]
            ref_euler = R.from_quat(ref_quat).as_euler('xyz', degrees=True)
            
            # Calculate attitude error
            attitude_error = current_euler - ref_euler
            attitude_error_norm = np.linalg.norm(attitude_error)
            
        except Exception as e:
            print(f"Warning: Could not compute attitude error: {e}")
            attitude_error = np.zeros(3)
            attitude_error_norm = 0.0
        
        # Calculate arm joint errors if robot has arm
        arm_joint_error = np.zeros(2)
        arm_joint_error_norm = 0.0
        if self.robot.nq > 7 and len(current_state) > 8 and len(ref_state) > 8:
            try:
                arm_joint_error = np.degrees(current_state[7:9]) - np.degrees(ref_state[7:9])
                arm_joint_error_norm = np.linalg.norm(arm_joint_error)
            except Exception as e:
                print(f"Warning: Could not compute arm joint error: {e}")
        
        # Calculate end-effector tracking error if robot has arm
        gripper_pos_error = np.zeros(3)
        gripper_pos_error_norm = 0.0
        gripper_pitch_error = 0.0
        
        if self.robot.nq > 7:  # Has arm joints
            try:
                gripper_frame_id = self.robot.getFrameId("gripper_link")
                
                # Calculate actual gripper pose
                q_actual = np.zeros(self.robot.nq)
                q_actual[:len(current_state[:self.robot.nq])] = current_state[:self.robot.nq]
                pin.forwardKinematics(self.robot, self.robot_data, q_actual)
                pin.updateFramePlacements(self.robot, self.robot_data)
                gripper_pose_actual = self.robot_data.oMf[gripper_frame_id].copy()
                
                # Calculate reference gripper pose
                q_ref = np.zeros(self.robot.nq)
                q_ref[:len(ref_state[:self.robot.nq])] = ref_state[:self.robot.nq]
                pin.forwardKinematics(self.robot, self.robot_data, q_ref)
                pin.updateFramePlacements(self.robot, self.robot_data)
                gripper_pose_ref = self.robot_data.oMf[gripper_frame_id].copy()
                
                # Position error
                gripper_pos_error = gripper_pose_actual.translation - gripper_pose_ref.translation
                gripper_pos_error_norm = np.linalg.norm(gripper_pos_error)
                
                # Orientation error (only pitch angle)
                gripper_rot_actual = R.from_matrix(gripper_pose_actual.rotation)
                gripper_rot_ref = R.from_matrix(gripper_pose_ref.rotation)
                # Extract only pitch angle (rotation around y-axis, index 1 in 'xyz' euler)
                gripper_pitch_actual = gripper_rot_actual.as_euler('xyz', degrees=True)[1]
                gripper_pitch_ref = gripper_rot_ref.as_euler('xyz', degrees=True)[1]
                gripper_pitch_error = gripper_pitch_actual - gripper_pitch_ref
                
                # Debug: print first few calculations
                if current_time < 1.0:  # Only print for first second
                    print(f"Debug gripper error at t={current_time:.3f}:")
                    print(f"  nq: {self.robot.nq}, current_state length: {len(current_state)}")
                    print(f"  Arm joints actual: {q_actual[7:9] if self.robot.nq > 8 else 'N/A'}")
                    print(f"  Arm joints ref: {q_ref[7:9] if self.robot.nq > 8 else 'N/A'}")
                    print(f"  Gripper pos error: {gripper_pos_error}")
                    print(f"  Gripper pos error norm: {gripper_pos_error_norm}")
                
            except Exception as e:
                print(f"Warning: Could not compute gripper tracking error: {e}")
        else:
            # Debug: check why condition is not met
            print(f"Debug: Robot nq={self.robot.nq}, not computing gripper error (needs nq > 7)")
        
        # Package tracking error data
        tracking_error = {
            'time': current_time,
            'position_error': position_error.copy(),
            'position_error_norm': position_error_norm,
            'attitude_error': attitude_error.copy(),
            'attitude_error_norm': attitude_error_norm,
            'arm_joint_error': arm_joint_error.copy(),
            'arm_joint_error_norm': arm_joint_error_norm,
            'gripper_pos_error': gripper_pos_error.copy(),
            'gripper_pos_error_norm': gripper_pos_error_norm,
            'gripper_pitch_error': gripper_pitch_error
        }
        
        return tracking_error
    
    def solve_mpc(self, current_state, current_time):
        """Solve MPC optimization problem"""
        start_time = time.time()
        
        # Get reference state and index
        ref_state, traj_index = self.get_reference_state(current_time)
        
        # Update MPC problem
        self.mpc_controller.problem.x0 = current_state.copy()
        
        # Calculate MPC reference index (in ms)
        mpc_ref_index = int(current_time * 1000)
        self.mpc_controller.updateProblem(mpc_ref_index)
        
        try:
            # Solve MPC problem
            success = self.mpc_controller.solver.solve(
                self.mpc_controller.solver.xs,
                self.mpc_controller.solver.us,
                self.mpc_controller.iters
            )
            
            if not success or self.mpc_controller.safe_cb.cost > 20000:
                print(f"Warning: MPC solver failed at time {current_time:.3f}s, cost: {self.mpc_controller.safe_cb.cost}")
                # Use previous control if available
                if self.current_control is not None:
                    return self.current_control, time.time() - start_time, traj_index
                else:
                    return np.zeros(self.control_dim), time.time() - start_time, traj_index
            
            # Get control command
            control_squash = self.mpc_controller.solver.us_squash[0]
            # print(f"control_squash: {control_squash}")
            
            # Process control based on actuation mode and control dimension
            if self.control_dim == 2:
                # Arm-only simulation: directly use last 2 elements as joint torques
                control_ft = thrustToForceTorqueAll(
                    control_squash,
                    self.platform_params.tau_f
                )
                # Extract only the arm joint torques for 2DOF arm simulation
                if len(control_ft) > 2:
                    control_ft = control_ft[-2:]
            else:
                # Full system: process based on simulation actuation mode
                if self.simulation_actuation_mode == 'full':
                    # Convert to force/torque (original behavior)
                    control_ft = thrustToForceTorqueAll(
                        control_squash,
                        self.platform_params.tau_f
                    )
                elif self.simulation_actuation_mode == 'quad':
                    # For quad mode, use direct motor thrusts (control_squash directly)
                    # control_squash contains [motor1, motor2, motor3, motor4, arm_joint1, arm_joint2]
                    control_ft = control_squash.copy()
                elif self.simulation_actuation_mode == 'thrust_angular_velocity':
                    # For thrust + angular velocity mode, use total thrust + angular velocity control
                    control_ft_original = thrustToForceTorqueAll(
                        control_squash,
                        self.platform_params.tau_f
                    )
                    # Set current time for data recording
                    self.current_time = current_time
                    control_ft = self.process_thrust_angular_velocity_control(control_ft_original, current_state)
                else:
                    raise ValueError(f"Unknown simulation_actuation_mode: {self.simulation_actuation_mode}")
            
            # print(f"control_ft ({self.simulation_actuation_mode} mode): {control_ft}")
            
            solve_time = time.time() - start_time
            
            return control_ft, solve_time, traj_index
            
        except Exception as e:
            print(f"Error in MPC solver at time {current_time:.3f}s: {e}")
            if self.current_control is not None:
                return self.current_control, time.time() - start_time, traj_index
            else:
                return np.zeros(self.control_dim), time.time() - start_time, traj_index
    
    def process_thrust_angular_velocity_control(self, control_squash, current_state):
        """
        Process control for thrust + angular velocity mode
        
        Args:
            control_squash: MPC control output (motor thrusts + arm joints)
            current_state: Current system state
            
        Returns:
            control_ft: Control in force/torque format for simulation
        """
        # Get next step angular velocity from MPC planning
        next_state = self.mpc_controller.solver.xs[1]  # Next step planned state
        nq = self.mpc_controller.robot_model.nq
        
        # Extract planned angular velocity (pitch component, index 1)
        planned_angular_velocity = next_state[nq+3:nq+6]  # [wx, wy, wz]
        planned_pitch_rate = planned_angular_velocity[1]  # pitch rate (wy)
        
        # Get planned pitch rate for next step (if available)
        planned_pitch_rate_next = 0.0
        if len(self.mpc_controller.solver.xs) > 2:
            next_next_state = self.mpc_controller.solver.xs[2]
            planned_angular_velocity_next = next_next_state[nq+3:nq+6]
            planned_pitch_rate_next = planned_angular_velocity_next[1]
        
        # Get current angular velocity
        current_angular_velocity = current_state[nq+3:nq+6]
        current_pitch_rate = current_angular_velocity[1]
        
        # Use PID to convert pitch angular velocity command to torque
        pitch_torque = self.pitch_pid.update(planned_pitch_rate, current_pitch_rate)
        print(f"planned_pitch_rate: {planned_pitch_rate}, current_pitch_rate: {current_pitch_rate}, pitch_torque: {pitch_torque}")
        
        # Convert to force/torque format
        # [Fx, Fy, Fz, Mx, My, Mz, arm_joint1, arm_joint2]
        control_ft = control_squash.copy()
        
        # Store original planned pitch torque
        planned_pitch_torque = control_ft[4] if len(control_ft) > 4 else 0.0
        
        # Set pitch torque (My)
        print(f"planned pitch torque: {planned_pitch_torque} -> {pitch_torque}")
        control_ft[4] = pitch_torque
        
        # Record pitch control data
        pitch_data = {
            'time': getattr(self, 'current_time', 0.0),
            'planned_pitch_torque': planned_pitch_torque,
            'pid_pitch_torque': pitch_torque,
            'planned_pitch_rate': planned_pitch_rate,
            'planned_pitch_rate_next': planned_pitch_rate_next,
            'actual_pitch_rate': current_pitch_rate
        }
        self.pitch_control_data.append(pitch_data)
        
        # Keep arm joint commands unchanged
        if len(control_squash) > 4:
            control_ft[-2:] = control_squash[-2:]  # arm joint torques
            
        return control_ft
    
    def solve_l1_control(self, current_state, u_baseline, current_time):
        """
        Solve L1 adaptive control
        
        Args:
            current_state: Current state
            u_baseline: Baseline control from MPC
            current_time: Current simulation time
            
        Returns:
            u_adaptive: Adaptive control command
            compute_time: Computation time
        """
        if not self.enable_l1_control:
            return np.zeros_like(u_baseline), 0.0
        
        start_time = time.time()
        
        try:
            # Get reference state
            ref_state, _ = self.get_reference_state(current_time)
            self.l1_controller.current_state = current_state.copy()
            self.l1_controller.z_ref_all = ref_state.copy()
            
            self.l1_controller.u_mpc = u_baseline.copy()
            

            self.l1_controller.z_ref = self.l1_controller.get_state_angle_single_rad(self.l1_controller.z_ref_all)
            self.l1_controller.z_real = self.l1_controller.get_state_angle_single_rad(self.l1_controller.current_state)
            
            # Update L1 controller step by step
            self.l1_controller.update_z_hat_vel()
            self.l1_controller.update_z_tilde()
            self.l1_controller.update_sig_hat_all_v2_new()
            self.l1_controller.update_u_ad()
            
            u_adaptive = self.l1_controller.u_ad.copy()
                
            
            compute_time = time.time() - start_time
            return u_adaptive, compute_time
            
        except Exception as e:
            print(f"Error in L1 control computation at time {current_time:.3f}s: {e}")
            compute_time = time.time() - start_time
            return np.zeros_like(u_baseline), compute_time
    
    def apply_disturbance(self, current_time):
        """Apply external disturbance if enabled"""
        if not self.enable_disturbance or current_time < self.disturbance_start_time:
            return np.zeros(self.control_dim)
        
        disturbance = np.zeros(self.control_dim)
        
        # Apply disturbance based on control dimensions
        if self.control_dim >= 6:
            # For full body dynamics (6DOF + joints)
            disturbance[:3] = self.disturbance_force[:3] if len(self.disturbance_force) >= 3 else [0, 0, 0]
            disturbance[3:6] = self.disturbance_torque[:3] if len(self.disturbance_torque) >= 3 else [0, 0, 0]
            
            # Apply joint sin disturbance if enabled and joints exist
            if (hasattr(self, 'enable_joint_sin_disturbance') and self.enable_joint_sin_disturbance and 
                self.control_dim > 6):  # Has joint controls
                
                joint_start_idx = 6  # Joint controls start after 6DOF (force + torque)
                max_joints = self.control_dim - joint_start_idx
                
                # Get sin disturbance parameters with defaults
                amplitudes = getattr(self, 'joint_sin_amplitude', [0.1, 0.05])
                frequencies = getattr(self, 'joint_sin_frequency', [2.0, 1.5])
                
                # Apply sin disturbance to each joint
                for joint_idx in range(min(max_joints, len(amplitudes), len(frequencies))):
                    control_idx = joint_start_idx + joint_idx
                    if control_idx < self.control_dim:
                        amplitude = amplitudes[joint_idx]
                        frequency = frequencies[joint_idx] 
                        
                        # Calculate sin disturbance
                        sin_disturbance = amplitude * np.sin(2 * np.pi * frequency * current_time) + amplitude * 1.5 * np.cos(2 * np.pi * frequency * 1.5 * current_time)
                        disturbance[control_idx] += sin_disturbance
                        
        elif self.control_dim >= 3:
            # For simplified dynamics (only translational forces)
            disturbance[:3] = self.disturbance_force[:3] if len(self.disturbance_force) >= 3 else [0, 0, 0]
        
        return disturbance
    
    def integrate_arm_dynamics(self, current_state, control_input):
        """
        Integrate arm dynamics using simple Euler integration with ABA
        
        Args:
            current_state: Current state [q1, q2, dq1, dq2]
            control_input: Control torques [tau1, tau2]
            
        Returns:
            next_state: Next state after integration
        """
        # Extract position and velocity
        q = current_state[:self.robot.nq]  # Joint positions
        v = current_state[self.robot.nq:]  # Joint velocities
        
        # Compute accelerations using ABA (Articulated Body Algorithm)
        a = pin.aba(self.robot, self.robot_data, q, v, control_input)
        
        # Euler integration
        next_q = q + v * self.simulation_dt
        next_v = v + a * self.simulation_dt
        
        # Combine to form next state
        next_state = np.concatenate([next_q, next_v])
        
        return next_state
    
    def convert_control_for_simulation(self, control_input):
        """
        Convert control input to appropriate format for simulation model
        
        Args:
            control_input: Control input from MPC
            
        Returns:
            converted_control: Control input in format expected by simulation model
        """
        if self.control_dim == 2:
            # Arm-only simulation: no conversion needed
            return control_input
            
        # For full system, conversion depends on both MPC and simulation actuation modes
        # Note: MPC always outputs in the format specified by its configuration
        # We need to ensure compatibility between MPC output and simulation model input
        
        if self.simulation_actuation_mode == 'full':
            # Simulation expects force/torque format
            # If control_input is already in force/torque format, return as-is
            return control_input
            
        elif self.simulation_actuation_mode == 'quad':
            # Simulation expects direct motor thrusts
            # If control_input is in force/torque format, we need to convert back
            # This is a more complex conversion and may need platform-specific parameters
            
            # Check if control_input is already in motor thrust format
            if len(control_input) >= 4:
                # Assume it's already in the correct format
                return control_input
            else:
                # If it's in force/torque format, we need to convert
                # This would require inverse transformation of thrustToForceTorqueAll
                print(f"Warning: Control conversion from force/torque to motor thrusts not fully implemented")
                return control_input
        
        elif self.simulation_actuation_mode == 'thrust_angular_velocity':
            # Simulation expects force/torque format (same as 'full' mode)
            # control_input should already be in force/torque format from process_thrust_angular_velocity_control
            return control_input
                
        else:
            raise ValueError(f"Unknown simulation_actuation_mode: {self.simulation_actuation_mode}")
    
    def add_noise(self, state):
        """Add noise to state if enabled"""
        if not self.enable_noise:
            return state
        
        noise = np.random.normal(0, self.state_noise_std, state.shape)
        return state + noise
    
    def simulate(self):
        """Run numeric simulation"""
        print(f"\nStarting numeric simulation...")
        print(f"Simulation duration: {self.simulation_time} seconds")
        print(f"Total simulation steps: {int(self.simulation_time / self.simulation_dt)}")
        print(f"MPC updates every {self.control_interval} steps ({self.control_dt*1000:.1f} ms)")
        
        # Initialize state
        current_state = np.array(self.initial_state)
        current_time = 0.0
        sim_step = 0
        
        # Start simulation loop
        start_sim_time = time.time()
        
        while current_time < self.simulation_time:
            # Store current data
            self.time_data.append(current_time)
            self.state_data.append(current_state.copy())
            
            # Check if we need to update MPC (every control_interval steps)
            if sim_step % self.control_interval == 0:
                # Solve MPC problem
                print(f"Solving MPC problem at time {current_time:.3f}s")
                control_ft, solve_time, traj_index = self.solve_mpc(current_state, current_time)
                
                # Store MPC data
                self.current_control = control_ft.copy()
                self.current_mpc_index = traj_index
                self.mpc_solve_times.append(solve_time)
                self.mpc_costs.append(self.mpc_controller.solver.cost)
                self.mpc_iterations.append(self.mpc_controller.solver.iter)
                self.last_mpc_update_step = sim_step
                
                # Solve L1 adaptive control if enabled and time has passed start time
                if self.enable_l1_control and current_time >= self.l1_start_time:
                    u_adaptive, l1_compute_time = self.solve_l1_control(current_state, control_ft, current_time)
                    self.current_l1_control = u_adaptive.copy()
                    self.l1_compute_times.append(l1_compute_time)
                    
                    # Store L1 data
                    self.l1_u_baseline_data.append(control_ft.copy())
                    self.l1_u_ad_data.append(u_adaptive.copy())
                    
                    # Store additional L1 debugging data
                    if hasattr(self.l1_controller, 'z_tilde'):
                        self.l1_z_tilde_data.append(self.l1_controller.z_tilde.copy())
                    if hasattr(self.l1_controller, 'sig_hat'):
                        self.l1_sig_hat_data.append(self.l1_controller.sig_hat.copy())
                    if hasattr(self.l1_controller, 'z_hat'):
                        self.l1_z_hat_data.append(self.l1_controller.z_hat.copy())
                    
                    print(f"Step {sim_step:6d}, Time {current_time:6.3f}s, MPC: {solve_time*1000:6.2f}ms, "
                          f"L1: {l1_compute_time*1000:6.2f}ms, Cost: {self.mpc_controller.solver.cost:8.3f}, Traj: {traj_index}")
                else:
                    self.current_l1_control = None
                    if self.enable_l1_control:
                        # Fill with zeros for consistency
                        self.l1_u_baseline_data.append(control_ft.copy())
                        self.l1_u_ad_data.append(np.zeros_like(control_ft))
                        self.l1_compute_times.append(0.0)
                        self.l1_z_tilde_data.append(np.zeros(self.l1_controller.state_dim_euler))
                        self.l1_sig_hat_data.append(np.zeros(self.l1_controller.state_dim_euler))
                        self.l1_z_hat_data.append(np.zeros(self.l1_controller.state_dim_euler))
                    
                    print(f"Step {sim_step:6d}, Time {current_time:6.3f}s, MPC solved in {solve_time*1000:6.2f}ms, Iters: {self.mpc_controller.solver.iter}, "
                          f"Cost: {self.mpc_controller.solver.cost:8.3f}, Traj index: {traj_index}")
            
            # Determine final control command based on simulation mode
            control_input = self.current_control if self.current_control is not None else np.zeros(self.control_dim)
            
            # Apply L1 adaptive control if enabled and active
            if self.enable_l1_control and current_time >= self.l1_start_time and self.current_l1_control is not None:
                control_input = control_input + self.current_l1_control
            
            # Apply control delay
            if self.control_delay > 0:
                original_control_input = control_input.copy()
                control_input = self.get_delayed_control(control_input, current_time)
                # Handle case where delayed control is None (initial steps)
                if control_input is None:
                    if getattr(self, 'delay_use_hover_control', True) and self.simulation_actuation_mode == 'quad':
                        # Use hover control (approximately mg/4 for each motor) during delay period
                        hover_thrust = 5  # Approximate hover thrust per motor (N)
                        if len(original_control_input) >= 4:
                            control_input = np.zeros_like(original_control_input)
                            control_input[:4] = hover_thrust  # Set motor thrusts to hover value
                        else:
                            control_input = np.zeros_like(original_control_input)
                    else:
                        # Use zeros with the same shape as the original control
                        control_input = np.zeros_like(original_control_input)
            
            # Convert control to appropriate format for simulation model
            control_input = self.convert_control_for_simulation(control_input)
            
            # Apply disturbance
            disturbance = self.apply_disturbance(current_time)
            
            # Store disturbance data for plotting
            self.disturbance_data.append(disturbance.copy())
            
            if self.simulation_actuation_mode == 'quad':
                # cannot add disturbance to quad mode
                total_control = control_input.copy()
            else:
                # For 'full' and 'thrust_angular_velocity' modes, disturbance can be added
                total_control = control_input + disturbance
            
            # Store control data
            self.control_data.append(total_control.copy())
            self.trajectory_index_data.append(self.current_mpc_index)
            
            # Update state using simulation model
            if self.use_simple_dynamics:
                # Simple dynamics integration for arm-only simulation
                next_state = self.integrate_arm_dynamics(current_state, total_control)
            else:
                # Use crocoddyl dynamics
                self.sim_update_model.calc(self.sim_update_data, current_state, total_control)
                next_state = self.sim_update_data.xnext.copy()
            
            # Add noise if enabled
            next_state = self.add_noise(next_state)
            
            # Calculate and store tracking errors after next state is computed
            # Record at trajectory optimization frequency (every tracking_error_interval steps)
            if sim_step % self.tracking_error_interval == 0:
                tracking_error = self.calculate_tracking_errors(next_state, current_time + self.simulation_dt)
                self.tracking_error_data.append(tracking_error)
            
            # Update for next iteration
            current_state = next_state
            current_time += self.simulation_dt
            sim_step += 1
            
            # Progress indicator
            if sim_step % 1000 == 0:  # Every 1000 steps (1 second)
                progress = current_time / self.simulation_time * 100
                elapsed_time = time.time() - start_sim_time
                print(f"Progress: {progress:5.1f}%, Elapsed: {elapsed_time:6.2f}s, "
                      f"Avg MPC time: {np.mean(self.mpc_solve_times)*1000:6.2f}ms")
        
        total_sim_time = time.time() - start_sim_time
        print(f"\nSimulation completed!")
        print(f"Total simulation time: {total_sim_time:.2f} seconds")
        print(f"Simulation steps: {sim_step}")
        print(f"MPC updates: {len(self.mpc_solve_times)}")
        print(f"Average MPC solve time: {np.mean(self.mpc_solve_times)*1000:.2f} ms")
        print(f"Max MPC solve time: {np.max(self.mpc_solve_times)*1000:.2f} ms")
        print(f"Average MPC cost: {np.mean(self.mpc_costs):.3f}")
        if self.mpc_iterations:
            print(f"Average MPC iterations: {np.mean(self.mpc_iterations):.1f}")
            print(f"Max MPC iterations: {np.max(self.mpc_iterations)}")
    
    def plot_result_original(self, save_dir=None):
        """Original plot function - Plot simulation results and optionally save to files"""
        if not self.plot_results and save_dir is None:
            return
        
        print("Plotting results...")
        
        # Convert data to numpy arrays
        time_array = np.array(self.time_data)
        state_array = np.array(self.state_data)
        control_array = np.array(self.control_data)
        traj_index_array = np.array(self.trajectory_index_data)
        
        # Create reference trajectory for comparison
        ref_states = []
        for t in time_array:
            ref_state, _ = self.get_reference_state(t)
            ref_states.append(ref_state)
        ref_state_array = np.array(ref_states)
        
        # Determine subplot layout based on L1 control
        if self.enable_l1_control and len(self.l1_u_ad_data) > 0:
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle(f'Numeric Simulation Results - {self.robot_name} (MPC + L1)', fontsize=16)
        else:
            fig, axes = plt.subplots(3, 3, figsize=(15, 8))
            fig.suptitle(f'Numeric Simulation Results - {self.robot_name}', fontsize=16)
        
        # Plot position states
        position_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        for i in range(3):
            ax = axes[0, i]
            ax.plot(time_array, state_array[:, i], 'b-', linewidth=2, label='Actual')
            ax.plot(time_array, ref_state_array[:, i], 'r--', linewidth=2, label='Reference')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(position_labels[i])
            ax.legend()
            ax.grid(True)
            ax.set_title(position_labels[i])
        
        # Plot orientation states (quaternion to Euler)
        from scipy.spatial.transform import Rotation as R
        orientation_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
        
        # Convert quaternions to Euler angles
        actual_quats = state_array[:, 3:7]
        ref_quats = ref_state_array[:, 3:7]
        
        actual_euler = R.from_quat(actual_quats).as_euler('xyz', degrees=True)
        ref_euler = R.from_quat(ref_quats).as_euler('xyz', degrees=True)
        
        for i in range(3):
            ax = axes[1, i]
            ax.plot(time_array, actual_euler[:, i], 'b-', linewidth=2, label='Actual')
            ax.plot(time_array, ref_euler[:, i], 'r--', linewidth=2, label='Reference')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(orientation_labels[i])
            ax.legend()
            ax.grid(True)
            ax.set_title(orientation_labels[i])
        
        # Plot arm joint angles (if available)
        if self.robot.nq > 7:  # Has arm joints
            arm_labels = ['Joint 1 (deg)', 'Joint 2 (deg)', 'MPC Cost']
            for i in range(2):
                ax = axes[2, i]
                ax.plot(time_array, np.degrees(state_array[:, 7+i]), 'b-', linewidth=2, label='Actual')
                ax.plot(time_array, np.degrees(ref_state_array[:, 7+i]), 'r--', linewidth=2, label='Reference')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(arm_labels[i])
                ax.legend()
                ax.grid(True)
                ax.set_title(arm_labels[i])
            
            # Plot MPC cost
            ax = axes[2, 2]
            mpc_times = np.arange(len(self.mpc_costs)) * self.control_dt
            ax.plot(mpc_times, self.mpc_costs, 'g-', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('MPC Cost')
            ax.grid(True)
            ax.set_title('MPC Cost Evolution')
        else:
            # Plot control inputs for non-arm robots
            control_labels = ['Force X (N)', 'Force Y (N)', 'Force Z (N)']
            for i in range(3):
                ax = axes[2, i]
                ax.plot(time_array, control_array[:, i], 'g-', linewidth=2)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(control_labels[i])
                ax.grid(True)
                ax.set_title(control_labels[i])
        
        # Plot L1 controller data if available
        if self.enable_l1_control and len(self.l1_u_ad_data) > 0:
            l1_time_steps = np.arange(len(self.l1_u_ad_data)) * self.control_interval
            l1_time_array = l1_time_steps * self.simulation_dt
            l1_u_baseline_array = np.array(self.l1_u_baseline_data)
            l1_u_ad_array = np.array(self.l1_u_ad_data)
            
            # L1 baseline vs adaptive control
            l1_labels = ['L1 Force X (N)', 'L1 Force Y (N)', 'L1 Force Z (N)']
            for i in range(min(3, l1_u_baseline_array.shape[1])):
                ax = axes[3, i]
                ax.plot(l1_time_array, l1_u_baseline_array[:, i], 'b-', linewidth=2, label='Baseline (MPC)')
                ax.plot(l1_time_array, l1_u_ad_array[:, i], 'r-', linewidth=2, label='Adaptive (L1)')
                ax.plot(l1_time_array, l1_u_baseline_array[:, i] + l1_u_ad_array[:, i], 'g--', linewidth=2, label='Total')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(l1_labels[i])
                ax.legend()
                ax.grid(True)
                ax.set_title(l1_labels[i])
        
        plt.tight_layout()
        
        # Create second figure for performance metrics
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        fig2.suptitle('Performance Metrics', fontsize=16)
        
        # MPC solve times
        mpc_time_steps = np.arange(len(self.mpc_solve_times)) * self.control_interval
        mpc_time_array = mpc_time_steps * self.simulation_dt
        
        axes2[0, 0].plot(mpc_time_array, np.array(self.mpc_solve_times) * 1000, 'r-', linewidth=2)
        axes2[0, 0].axhline(y=self.control_dt * 1000, color='k', linestyle='--', 
                           label=f'Control period ({self.control_dt*1000:.1f} ms)')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('MPC Solve Time (ms)')
        axes2[0, 0].set_title('MPC Computational Performance')
        axes2[0, 0].legend()
        axes2[0, 0].grid(True)
        
        # MPC cost evolution
        axes2[0, 1].plot(mpc_time_array, self.mpc_costs, 'b-', linewidth=2)
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('MPC Cost')
        axes2[0, 1].set_title('MPC Cost Evolution')
        axes2[0, 1].grid(True)
        
        # Position tracking error
        position_error = np.linalg.norm(state_array[:, :3] - ref_state_array[:, :3], axis=1)
        axes2[1, 0].plot(time_array, position_error, 'm-', linewidth=2)
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Position Error (m)')
        axes2[1, 0].set_title('Position Tracking Error')
        axes2[1, 0].grid(True)
        
        # Trajectory index evolution or L1 performance
        if self.enable_l1_control and len(self.l1_compute_times) > 0:
            # L1 computation time
            l1_time_steps = np.arange(len(self.l1_compute_times)) * self.control_interval
            l1_perf_time_array = l1_time_steps * self.simulation_dt
            axes2[1, 1].plot(l1_perf_time_array, np.array(self.l1_compute_times) * 1000, 'r-', linewidth=2)
            axes2[1, 1].set_xlabel('Time (s)')
            axes2[1, 1].set_ylabel('L1 Compute Time (ms)')
            axes2[1, 1].set_title('L1 Controller Performance')
            axes2[1, 1].grid(True)
        else:
            # Trajectory index evolution
            axes2[1, 1].plot(time_array, traj_index_array, 'c-', linewidth=2)
            axes2[1, 1].set_xlabel('Time (s)')
            axes2[1, 1].set_ylabel('Trajectory Index')
            axes2[1, 1].set_title('Reference Trajectory Progress')
            axes2[1, 1].grid(True)
        
        # Create third figure for L1 debugging data if available
        if self.enable_l1_control and len(self.l1_z_tilde_data) > 0:
            fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
            fig3.suptitle('L1 Controller Debug Information', fontsize=16)
            
            # Create time array that matches the L1 debug data size
            # Ensure consistency with main L1 time array computation
            l1_debug_steps = np.arange(len(self.l1_z_tilde_data)) * self.control_interval
            l1_debug_time_array = (l1_debug_steps * self.simulation_dt) + self.l1_start_time
            
            # Ensure L1 debug time array doesn't exceed simulation time
            if len(l1_debug_time_array) > 0 and l1_debug_time_array[-1] > time_array[-1]:
                valid_indices = l1_debug_time_array <= time_array[-1]
                l1_debug_time_array = l1_debug_time_array[valid_indices]
            l1_z_tilde_array = np.array(self.l1_z_tilde_data)
            
            # Plot z_tilde (state error) - ensure data length matches time array
            plot_length = min(len(l1_debug_time_array), len(l1_z_tilde_array))
            time_to_plot = l1_debug_time_array[:plot_length]
            z_tilde_to_plot = l1_z_tilde_array[:plot_length]
            
            # Add phase backgrounds to z_tilde position error plot (no legend, as it's separate figure)
            self._add_l1_phase_backgrounds(axes3[0, 0], time_array[0], time_array[-1], self.l1_start_time, self.disturbance_start_time)
            
            axes3[0, 0].plot(time_to_plot, z_tilde_to_plot[:, 0], 'b-', label='X error')
            axes3[0, 0].plot(time_to_plot, z_tilde_to_plot[:, 1], 'r-', label='Y error')
            axes3[0, 0].plot(time_to_plot, z_tilde_to_plot[:, 2], 'g-', label='Z error')
            axes3[0, 0].set_xlabel('Time (s)')
            axes3[0, 0].set_ylabel('Position Error (m)')
            axes3[0, 0].set_title('L1 State Predictor Error (z_tilde)')
            axes3[0, 0].legend()
            axes3[0, 0].grid(True)
            
            # Plot velocity errors if available
            if z_tilde_to_plot.shape[1] > 7:
                # Add phase backgrounds to velocity error plot
                self._add_l1_phase_backgrounds(axes3[0, 1], time_array[0], time_array[-1], self.l1_start_time, self.disturbance_start_time)
                
                axes3[0, 1].plot(time_to_plot, z_tilde_to_plot[:, 7], 'b-', label='VX error')
                axes3[0, 1].plot(time_to_plot, z_tilde_to_plot[:, 8], 'r-', label='VY error')
                axes3[0, 1].plot(time_to_plot, z_tilde_to_plot[:, 9], 'g-', label='VZ error')
                axes3[0, 1].set_xlabel('Time (s)')
                axes3[0, 1].set_ylabel('Velocity Error (m/s)')
                axes3[0, 1].set_title('L1 Velocity Error (z_tilde)')
                axes3[0, 1].legend()
                axes3[0, 1].grid(True)
            
            # Plot sigma_hat if available
            if len(self.l1_sig_hat_data) > 0:
                l1_sig_hat_debug_array = np.array(self.l1_sig_hat_data)
                # Ensure data length matches for sigma_hat
                sig_hat_plot_length = min(len(time_to_plot), len(l1_sig_hat_debug_array))
                sig_hat_time_to_plot = time_to_plot[:sig_hat_plot_length]
                sig_hat_to_plot = l1_sig_hat_debug_array[:sig_hat_plot_length]
                
                # Add phase backgrounds to sigma_hat plot
                self._add_l1_phase_backgrounds(axes3[1, 0], time_array[0], time_array[-1], self.l1_start_time, self.disturbance_start_time)
                
                if sig_hat_to_plot.shape[1] > 11:  # Check if force estimates exist
                    axes3[1, 0].plot(sig_hat_time_to_plot, sig_hat_to_plot[:, 9], 'b-', label='F_X estimate')
                    axes3[1, 0].plot(sig_hat_time_to_plot, sig_hat_to_plot[:, 10], 'r-', label='F_Y estimate')
                    axes3[1, 0].plot(sig_hat_time_to_plot, sig_hat_to_plot[:, 11], 'g-', label='F_Z estimate')
                axes3[1, 0].set_xlabel('Time (s)')
                axes3[1, 0].set_ylabel('Force Disturbance (N)')
                axes3[1, 0].set_title('L1 Disturbance Estimate (σ_hat)')
                axes3[1, 0].legend()
                axes3[1, 0].grid(True)
            
            # Plot L1 adaptive control magnitude using the L1 control data that matches debug data
            if len(self.l1_u_ad_data) > 0:
                # Only use the L1 data that corresponds to when L1 was active (matches z_tilde data length)
                l1_u_ad_active = np.array(self.l1_u_ad_data[-len(self.l1_z_tilde_data):])
                # Ensure data length matches for u_ad
                u_ad_plot_length = min(len(time_to_plot), len(l1_u_ad_active))
                u_ad_time_to_plot = time_to_plot[:u_ad_plot_length]
                u_ad_to_plot = l1_u_ad_active[:u_ad_plot_length]
                
                l1_adaptive_magnitude = np.linalg.norm(u_ad_to_plot, axis=1)
                axes3[1, 1].plot(u_ad_time_to_plot, l1_adaptive_magnitude, 'm-', linewidth=2)
            
            # Add phase backgrounds to L1 control effort plot
            self._add_l1_phase_backgrounds(axes3[1, 1], time_array[0], time_array[-1], self.l1_start_time, self.disturbance_start_time)
            
            axes3[1, 1].axvline(x=self.l1_start_time, color='k', linestyle='--', alpha=0.7, label='L1 Start')
            axes3[1, 1].set_xlabel('Time (s)')
            axes3[1, 1].set_ylabel('L1 Control Magnitude (N)')
            axes3[1, 1].set_title('L1 Adaptive Control Effort')
            axes3[1, 1].legend()
            axes3[1, 1].grid(True)
            
            plt.tight_layout()
        
        plt.tight_layout()
        
        # Save or show plots
        if save_dir is not None:
            # Save first figure
            fig1_path = save_dir / 'simulation_results.png'
            fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"  - Main results plot saved: {fig1_path}")
            
            # Save second figure
            fig2_path = save_dir / 'performance_metrics.png'
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            print(f"  - Performance metrics plot saved: {fig2_path}")
            
            # Save third figure if it exists
            if self.enable_l1_control and len(self.l1_z_tilde_data) > 0:
                fig3_path = save_dir / 'l1_debug_info.png'
                fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
                print(f"  - L1 debug plot saved: {fig3_path}")
            
            # Close all figures to free memory
            plt.close('all')
        else:
            # Show plots interactively
            plt.show()
    
    def _add_l1_phase_backgrounds(self, ax, sim_start, sim_end, l1_start, disturbance_start, show_legend=False):
        """Add colored background regions to show different simulation phases"""
        phase_colors = {
            'mpc_only': 'lightblue',        # Phase 1: Pure MPC control
            'mpc_disturb': 'lightcoral',    # Phase 2: MPC + disturbance  
            'mpc_disturb_l1': 'lightgreen'  # Phase 3: MPC + disturbance + L1
        }
        phase_alpha = 0.2
        
        # Phase 1: Pure MPC control (before disturbance starts)
        if disturbance_start > sim_start:
            ax.axvspan(sim_start, disturbance_start, alpha=phase_alpha, 
                      color=phase_colors['mpc_only'], label='Phase 1: Pure MPC' if show_legend else '', zorder=0)
        
        # Phase 2: MPC + disturbance (disturbance active, L1 not yet active)
        if l1_start > disturbance_start:
            ax.axvspan(disturbance_start, l1_start, alpha=phase_alpha, 
                      color=phase_colors['mpc_disturb'], label='Phase 2: MPC + Disturbance' if show_legend else '', zorder=0)
        
        # Phase 3: MPC + disturbance + L1 (both disturbance and L1 active)
        if sim_end > l1_start:
            ax.axvspan(l1_start, sim_end, alpha=phase_alpha, 
                      color=phase_colors['mpc_disturb_l1'], label='Phase 3: MPC + Disturbance + L1' if show_legend else '', zorder=0)
    
    def plot_result(self, save_dir=None):
        """New comprehensive plot function showing states, controls, and tracking errors"""
        if not self.plot_results and save_dir is None:
            return
        
        print("Plotting comprehensive simulation results...")
        
        
        # Convert data to numpy arrays
        time_array = np.array(self.time_data)
        state_array = np.array(self.state_data)
        control_array = np.array(self.control_data)
        
        # Create reference trajectory for comparison
        ref_states = []
        ref_controls = []
        for t in time_array:
            ref_state, _ = self.get_reference_state(t)
            ref_control = self.get_reference_control(t)
            ref_states.append(ref_state)
            
            # Convert reference control to the same format as actual control
            if len(ref_control) > 0 and getattr(self, 'simulation_actuation_mode', None) == 'quad':
                # ref_control is in [motor1, motor2, motor3, motor4, joint1, joint2] format
                # actual control_array is in the same format after conversion
                ref_controls.append(ref_control)
            elif len(ref_control) > 0:
                # If not in quad mode, we might need force/torque conversion
                # For now, keep as is - this handles the FT mode
                ref_controls.append(ref_control)
            else:
                # No reference control available
                ref_controls.append(np.zeros(self.control_dim))
                
        ref_state_array = np.array(ref_states)
        ref_control_array = np.array(ref_controls)
        
        # Check if reference control array has valid data
        has_ref_controls = ref_control_array.size > 0 and not np.all(ref_control_array == 0)
        if not has_ref_controls:
            print("Warning: No reference control data available for plotting comparison")
        
        # Convert quaternions to Euler angles
        from scipy.spatial.transform import Rotation as R
        actual_quats = state_array[:, 3:7]
        ref_quats = ref_state_array[:, 3:7]
        actual_euler = R.from_quat(actual_quats).as_euler('xyz', degrees=True)
        ref_euler = R.from_quat(ref_quats).as_euler('xyz', degrees=True)
        
        # Calculate end-effector (gripper) tracking errors if robot has arm
        gripper_pos_actual = []
        gripper_pos_ref = []
        gripper_pitch_actual = []
        gripper_pitch_ref = []
        
        if self.robot.nq > 7:  # Has arm joints, calculate gripper tracking errors
            try:
                gripper_frame_id = self.robot.getFrameId("gripper_link")
                print(f"Computing gripper tracking errors for {len(state_array)} time steps...")
                
                for i in range(len(state_array)):
                    # Calculate actual gripper pose
                    q_actual = np.zeros(self.robot.nq)
                    q_actual[:len(state_array[i, :self.robot.nq])] = state_array[i, :self.robot.nq]
                    pin.forwardKinematics(self.robot, self.robot_data, q_actual)
                    pin.updateFramePlacements(self.robot, self.robot_data)
                    gripper_pose_actual = self.robot_data.oMf[gripper_frame_id]
                    gripper_pos_actual.append(gripper_pose_actual.translation.copy())
                    gripper_rot_actual = R.from_matrix(gripper_pose_actual.rotation)
                    # Only store pitch angle (index 1 in 'xyz' euler)
                    gripper_pitch_actual.append(gripper_rot_actual.as_euler('xyz', degrees=True)[1])
                    
                    # Calculate reference gripper pose
                    q_ref = np.zeros(self.robot.nq)
                    q_ref[:len(ref_state_array[i, :self.robot.nq])] = ref_state_array[i, :self.robot.nq]
                    pin.forwardKinematics(self.robot, self.robot_data, q_ref)
                    pin.updateFramePlacements(self.robot, self.robot_data)
                    gripper_pose_ref = self.robot_data.oMf[gripper_frame_id]
                    gripper_pos_ref.append(gripper_pose_ref.translation.copy())
                    gripper_rot_ref = R.from_matrix(gripper_pose_ref.rotation)
                    # Only store pitch angle (index 1 in 'xyz' euler)
                    gripper_pitch_ref.append(gripper_rot_ref.as_euler('xyz', degrees=True)[1])
                
                gripper_pos_actual = np.array(gripper_pos_actual)
                gripper_pos_ref = np.array(gripper_pos_ref)
                gripper_pitch_actual = np.array(gripper_pitch_actual)
                gripper_pitch_ref = np.array(gripper_pitch_ref)
                
                print("Successfully computed gripper tracking errors")
                
            except Exception as e:
                print(f"Warning: Could not compute gripper tracking errors: {e}")
                gripper_pos_actual = None
                gripper_pos_ref = None
                gripper_pitch_actual = None
                gripper_pitch_ref = None
        else:
            gripper_pos_actual = None
            gripper_pos_ref = None
            gripper_pitch_actual = None
            gripper_pitch_ref = None
        
        # === Figure 1: States and Controls ===
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f'Simulation Results - {self.robot_name} ({self.trajectory_name})', fontsize=16)
        
        # === First Row: Combined Position, Combined Angles, Arm angles ===
        # Combined Position plot (x,y,z in one subplot)
        ax = axes[0, 0]
        ax.plot(time_array, state_array[:, 0], 'b-', linewidth=2, label='X Actual')
        ax.plot(time_array, ref_state_array[:, 0], 'b--', linewidth=2, label='X Ref')
        ax.plot(time_array, state_array[:, 1], 'r-', linewidth=2, label='Y Actual')
        ax.plot(time_array, ref_state_array[:, 1], 'r--', linewidth=2, label='Y Ref')
        ax.plot(time_array, state_array[:, 2], 'g-', linewidth=2, label='Z Actual')
        ax.plot(time_array, ref_state_array[:, 2], 'g--', linewidth=2, label='Z Ref')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Position (m)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.set_title('Position (X, Y, Z)', fontsize=9)
        
        # Combined Attitude plot (pitch, roll, yaw in one subplot)
        ax = axes[0, 1]
        angle_order = [1, 0, 2]  # pitch, roll, yaw from xyz Euler
        angle_colors = ['b', 'r', 'g']
        angle_labels = ['Pitch', 'Roll', 'Yaw']
        for i in range(3):
            angle_idx = angle_order[i]
            color = angle_colors[i]
            ax.plot(time_array, actual_euler[:, angle_idx], color=color, linestyle='-', linewidth=2, label=f'{angle_labels[i]} Actual')
            ax.plot(time_array, ref_euler[:, angle_idx], color=color, linestyle='--', linewidth=2, label=f'{angle_labels[i]} Ref')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Angle (deg)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.set_title('Attitude (Pitch, Roll, Yaw)', fontsize=9)
        
        # Arm joint angles (if available) - now combining both joints in one plot
        if self.robot.nq > 7:  # Has arm joints
            ax = axes[0, 2]
            ax.plot(time_array, np.degrees(state_array[:, 7]), 'b-', linewidth=2, label='Joint 1 Actual')
            ax.plot(time_array, np.degrees(ref_state_array[:, 7]), 'b--', linewidth=2, label='Joint 1 Ref')
            ax.plot(time_array, np.degrees(state_array[:, 8]), 'r-', linewidth=2, label='Joint 2 Actual')
            ax.plot(time_array, np.degrees(ref_state_array[:, 8]), 'r--', linewidth=2, label='Joint 2 Ref')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Joint Angle (deg)', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_title('Arm Joint Angles', fontsize=9)
        else:
            # Fill with empty plot for non-arm robots
            ax = axes[0, 2]
            ax.text(0.5, 0.5, 'No arm joints', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Arm Joint Angles (N/A)', fontsize=9)
        
        # === Second Row: Velocities ===
        # Combined Linear velocities (VX, VY, VZ)
        ax = axes[1, 0]
        velocity_indices = [9, 10, 11]  # Linear velocities
        velocity_colors = ['b', 'r', 'g']
        velocity_labels = ['VX', 'VY', 'VZ']
        for i in range(3):
            vel_idx = velocity_indices[i]
            color = velocity_colors[i]
            if vel_idx < state_array.shape[1]:
                ax.plot(time_array, state_array[:, vel_idx], color=color, linestyle='-', linewidth=2, label=f'{velocity_labels[i]} Actual')
                ax.plot(time_array, ref_state_array[:, vel_idx], color=color, linestyle='--', linewidth=2, label=f'{velocity_labels[i]} Ref')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Linear Velocity (m/s)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.set_title('Linear Velocities (VX, VY, VZ)', fontsize=9)
        
        # Combined Angular velocities (ωX, ωY, ωZ)
        ax = axes[1, 1]
        angular_indices = [12, 13, 14]  # Angular velocities
        angular_labels = ['ωX', 'ωY', 'ωZ']
        for i in range(3):
            ang_idx = angular_indices[i]
            color = velocity_colors[i]
            if ang_idx < state_array.shape[1]:
                ax.plot(time_array, state_array[:, ang_idx], color=color, linestyle='-', linewidth=2, label=f'{angular_labels[i]} Actual')
                ax.plot(time_array, ref_state_array[:, ang_idx], color=color, linestyle='--', linewidth=2, label=f'{angular_labels[i]} Ref')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True)
        ax.set_title('Angular Velocities (ωX, ωY, ωZ)', fontsize=9)
        
        # Arm joint velocities (if available)
        if self.robot.nq > 7:
            ax = axes[1, 2]
            vel_idx1, vel_idx2 = 15, 16  # Assuming arm velocities follow the pattern
            if vel_idx1 < state_array.shape[1] and vel_idx2 < state_array.shape[1]:
                ax.plot(time_array, state_array[:, vel_idx1], 'b-', linewidth=2, label='Joint 1 Vel Actual')
                ax.plot(time_array, ref_state_array[:, vel_idx1], 'b--', linewidth=2, label='Joint 1 Vel Ref')
                ax.plot(time_array, state_array[:, vel_idx2], 'r-', linewidth=2, label='Joint 2 Vel Actual')
                ax.plot(time_array, ref_state_array[:, vel_idx2], 'r--', linewidth=2, label='Joint 2 Vel Ref')
                ax.set_xlabel('Time (s)', fontsize=9)
                ax.set_ylabel('Joint Velocity (rad/s)', fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(True)
                ax.set_title('Arm Joint Velocities', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Arm Joint Velocities (N/A)', fontsize=9)
        else:
            ax = axes[1, 2]
            ax.text(0.5, 0.5, 'No arm joints', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Arm Joint Velocities (N/A)', fontsize=9)
        
        # === Third Row: Control Inputs ===
        # Combined Force controls - adapt to control mode
        ax = axes[2, 0]
        force_colors = ['b', 'r', 'g', 'orange']
        
        if getattr(self, 'simulation_actuation_mode', None) == 'quad':
            # In quad mode, show motor thrusts instead of forces
            motor_labels = ['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4']
            for i in range(min(4, control_array.shape[1])):
                ax.plot(time_array, control_array[:, i], color=force_colors[i], linestyle='-', linewidth=2, label=f'{motor_labels[i]} Actual')
                if has_ref_controls and i < ref_control_array.shape[1]:
                    ax.plot(time_array, ref_control_array[:, i], color=force_colors[i], linestyle='--', linewidth=2, label=f'{motor_labels[i]} Ref')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Motor Thrust (N)', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_title('Motor Thrusts', fontsize=9)
        else:
            # In FT mode, show forces
            force_labels = ['FX', 'FY', 'FZ']
            for i in range(3):
                if i < control_array.shape[1]:
                    ax.plot(time_array, control_array[:, i], color=force_colors[i], linestyle='-', linewidth=2, label=f'{force_labels[i]} Actual')
                if has_ref_controls and i < ref_control_array.shape[1]:
                    ax.plot(time_array, ref_control_array[:, i], color=force_colors[i], linestyle='--', linewidth=2, label=f'{force_labels[i]} Ref')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Force (N)', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_title('Forces (FX, FY, FZ)', fontsize=9)
        
        # Combined Torque controls - adapt to control mode
        ax = axes[2, 1]
        
        if getattr(self, 'simulation_actuation_mode', None) == 'quad':
            # In quad mode, we could show derived forces/torques or hide this plot
            ax.text(0.5, 0.5, 'Quad Mode:\nForces/Torques derived\nfrom motor thrusts', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Body Forces/Torques (Derived)', fontsize=9)
        else:
            # In FT mode, show torques
            torque_labels = ['TX', 'TY', 'TZ']
            for i in range(3):
                control_idx = i + 3
                if control_idx < control_array.shape[1]:
                    ax.plot(time_array, control_array[:, control_idx], color=force_colors[i], linestyle='-', linewidth=2, label=f'{torque_labels[i]} Actual')
                if has_ref_controls and control_idx < ref_control_array.shape[1]:
                    ax.plot(time_array, ref_control_array[:, control_idx], color=force_colors[i], linestyle='--', linewidth=2, label=f'{torque_labels[i]} Ref')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Torque (Nm)', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_title('Torques (TX, TY, TZ)', fontsize=9)
        
        # Arm joint control inputs (if available) - handle different control formats
        if self.robot.nq > 7:  # Has arm joints
            ax = axes[2, 2]
            
            if getattr(self, 'simulation_actuation_mode', None) == 'quad':
                # In quad mode: [motor1, motor2, motor3, motor4, joint1, joint2]
                joint1_idx, joint2_idx = 4, 5
                if control_array.shape[1] > joint1_idx:
                    ax.plot(time_array, control_array[:, joint1_idx], 'b-', linewidth=2, label='Joint 1 Actual')
                    if has_ref_controls and ref_control_array.shape[1] > joint1_idx:
                        ax.plot(time_array, ref_control_array[:, joint1_idx], 'b--', linewidth=2, label='Joint 1 Ref')
                if control_array.shape[1] > joint2_idx:
                    ax.plot(time_array, control_array[:, joint2_idx], 'r-', linewidth=2, label='Joint 2 Actual')
                    if has_ref_controls and ref_control_array.shape[1] > joint2_idx:
                        ax.plot(time_array, ref_control_array[:, joint2_idx], 'r--', linewidth=2, label='Joint 2 Ref')
            else:
                # In FT mode: [FX, FY, FZ, TX, TY, TZ, joint1, joint2]
                joint1_idx, joint2_idx = 6, 7
                if control_array.shape[1] > joint1_idx:
                    ax.plot(time_array, control_array[:, joint1_idx], 'b-', linewidth=2, label='Joint 1 Actual')
                    if has_ref_controls and ref_control_array.shape[1] > joint1_idx:
                        ax.plot(time_array, ref_control_array[:, joint1_idx], 'b--', linewidth=2, label='Joint 1 Ref')
                if control_array.shape[1] > joint2_idx:
                    ax.plot(time_array, control_array[:, joint2_idx], 'r-', linewidth=2, label='Joint 2 Actual')
                    if has_ref_controls and ref_control_array.shape[1] > joint2_idx:
                        ax.plot(time_array, ref_control_array[:, joint2_idx], 'r--', linewidth=2, label='Joint 2 Ref')
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Joint Torque (Nm)', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
            ax.set_title('Arm Joint Torques', fontsize=9)
        else:
            ax = axes[2, 2]
            ax.text(0.5, 0.5, f'No arm joints\n(nq={self.robot.nq})', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Arm Joint Torques (N/A)', fontsize=9)
            
        plt.tight_layout()
        
        # === Figure 2: Tracking Errors ===
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 10))
        fig2.suptitle('Tracking Errors', fontsize=16)
        
        # Use recorded tracking error data (computed during simulation)
        if len(self.tracking_error_data) > 0:
            tracking_error_time = np.array([te['time'] for te in self.tracking_error_data])
            position_errors = np.array([te['position_error'] for te in self.tracking_error_data])
            attitude_errors = np.array([te['attitude_error'] for te in self.tracking_error_data])
            position_error_norm = np.array([te['position_error_norm'] for te in self.tracking_error_data])
            attitude_error_norm = np.array([te['attitude_error_norm'] for te in self.tracking_error_data])
            
            # Extract arm joint errors if available
            arm_errors = np.array([te['arm_joint_error'] for te in self.tracking_error_data])
            
            # Extract gripper errors if available
            gripper_pos_errors = np.array([te['gripper_pos_error'] for te in self.tracking_error_data])
            gripper_pitch_errors = np.array([te['gripper_pitch_error'] for te in self.tracking_error_data])
            gripper_pos_error_norm = np.array([te['gripper_pos_error_norm'] for te in self.tracking_error_data])
            
            print(f"Using recorded tracking error data: {len(self.tracking_error_data)} data points (every {self.dt_traj_opt}ms)")
        else:
            # This should not happen in normal operation
            print("Warning: No tracking error data available, skipping tracking error plots")
            return
        
        # Combined Position tracking errors (X, Y, Z in one plot)
        axes2[0, 0].plot(tracking_error_time, position_errors[:, 0], 'b-', linewidth=2, label='X Error')
        axes2[0, 0].plot(tracking_error_time, position_errors[:, 1], 'r-', linewidth=2, label='Y Error')
        axes2[0, 0].plot(tracking_error_time, position_errors[:, 2], 'g-', linewidth=2, label='Z Error')
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Position Error (m)')
        axes2[0, 0].legend(fontsize=8)
        axes2[0, 0].grid(True)
        axes2[0, 0].set_title('Base Position Tracking Errors (X, Y, Z)')
        
        # Combined Attitude tracking errors (Pitch, Roll, Yaw in one plot)
        angle_order = [1, 0, 2]  # pitch, roll, yaw from xyz Euler
        angle_colors = ['b', 'r', 'g']
        angle_labels = ['Pitch', 'Roll', 'Yaw']
        
        for i in range(3):
            angle_idx = angle_order[i]
            color = angle_colors[i]
            axes2[0, 1].plot(tracking_error_time, attitude_errors[:, angle_idx], color=color, linewidth=2, label=f'{angle_labels[i]} Error')
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Attitude Error (deg)')
        axes2[0, 1].legend(fontsize=8)
        axes2[0, 1].grid(True)
        axes2[0, 1].set_title('Base Attitude Tracking Errors (Pitch, Roll, Yaw)')
        
        # Arm joint tracking errors (if available)
        if self.robot.nq > 7 and arm_errors is not None:
            axes2[0, 2].plot(tracking_error_time, arm_errors[:, 0], 'b-', linewidth=2, label='Joint 1 Error')
            axes2[0, 2].plot(tracking_error_time, arm_errors[:, 1], 'r-', linewidth=2, label='Joint 2 Error')
            axes2[0, 2].set_xlabel('Time (s)')
            axes2[0, 2].set_ylabel('Joint Angle Error (deg)')
            axes2[0, 2].legend(fontsize=8)
            axes2[0, 2].grid(True)
            axes2[0, 2].set_title('Arm Joint Angle Errors')
        else:
            axes2[0, 2].text(0.5, 0.5, 'No arm joints', ha='center', va='center', transform=axes2[0, 2].transAxes)
            axes2[0, 2].set_title('Arm Joint Errors (N/A)')
        
        # End-effector position tracking errors (if gripper data available)
        if gripper_pos_errors is not None and len(gripper_pos_errors) > 0:
            axes2[1, 0].plot(tracking_error_time, gripper_pos_errors[:, 0], 'b-', linewidth=2, label='EE X Error')
            axes2[1, 0].plot(tracking_error_time, gripper_pos_errors[:, 1], 'r-', linewidth=2, label='EE Y Error')
            axes2[1, 0].plot(tracking_error_time, gripper_pos_errors[:, 2], 'g-', linewidth=2, label='EE Z Error')
            axes2[1, 0].set_xlabel('Time (s)')
            axes2[1, 0].set_ylabel('EE Position Error (m)')
            axes2[1, 0].legend(fontsize=8)
            axes2[1, 0].grid(True)
            axes2[1, 0].set_title('End-Effector Position Tracking Errors')
        else:
            axes2[1, 0].text(0.5, 0.5, 'No end-effector data', ha='center', va='center', transform=axes2[1, 0].transAxes)
            axes2[1, 0].set_title('End-Effector Position Errors (N/A)')
        
        # End-effector pitch tracking error (if gripper data available)
        if gripper_pitch_errors is not None and len(gripper_pitch_errors) > 0:
            axes2[1, 1].plot(tracking_error_time, gripper_pitch_errors, 'g-', linewidth=2, label='EE Pitch Error')
            axes2[1, 1].set_xlabel('Time (s)')
            axes2[1, 1].set_ylabel('EE Pitch Error (deg)')
            axes2[1, 1].legend(fontsize=8)
            axes2[1, 1].grid(True)
            axes2[1, 1].set_title('End-Effector Pitch Tracking Error')
        else:
            axes2[1, 1].text(0.5, 0.5, 'No end-effector data', ha='center', va='center', transform=axes2[1, 1].transAxes)
            axes2[1, 1].set_title('End-Effector Pitch Error (N/A)')
        
        # Position error magnitude
        axes2[1, 2].plot(tracking_error_time, position_error_norm, 'm-', linewidth=2, label='Base Position')
        
        # End-effector position error magnitude (if available)
        if gripper_pos_error_norm is not None and len(gripper_pos_error_norm) > 0:
            axes2[1, 2].plot(tracking_error_time, gripper_pos_error_norm, 'c-', linewidth=2, label='End-Effector')
        
        axes2[1, 2].set_xlabel('Time (s)')
        axes2[1, 2].set_ylabel('Position Error Magnitude (m)')
        axes2[1, 2].legend(fontsize=8)
        axes2[1, 2].grid(True)
        axes2[1, 2].set_title('Position Error Magnitudes')
        
        # Attitude error magnitude
        axes2[2, 0].plot(tracking_error_time, attitude_error_norm, 'c-', linewidth=2, label='Base Attitude')
        
        # End-effector pitch error magnitude (if available)
        if gripper_pitch_errors is not None and len(gripper_pitch_errors) > 0:
            axes2[2, 0].plot(tracking_error_time, np.abs(gripper_pitch_errors), 'm-', linewidth=2, label='End-Effector Pitch')
        
        axes2[2, 0].set_xlabel('Time (s)')
        axes2[2, 0].set_ylabel('Attitude Error Magnitude (deg)')
        axes2[2, 0].legend(fontsize=8)
        axes2[2, 0].grid(True)
        axes2[2, 0].set_title('Attitude Error Magnitudes')
        
        # Combined velocity tracking errors - use same frequency as other tracking errors
        if len(self.tracking_error_data) > 0:
            # Extract velocity data at MPC update frequency (same as other tracking errors)
            velocity_errors_x = []
            velocity_errors_y = []
            velocity_errors_z = []
            
            for te in self.tracking_error_data:
                # Get current time and states
                t = te['time']
                # Find closest state from simulation data
                closest_idx = np.argmin(np.abs(time_array - t))
                if closest_idx < len(state_array) and state_array.shape[1] > 11:
                    vel_actual = state_array[closest_idx, 9:12]
                    vel_ref = ref_state_array[closest_idx, 9:12]
                    vel_error = vel_actual - vel_ref
                    velocity_errors_x.append(vel_error[0])
                    velocity_errors_y.append(vel_error[1])
                    velocity_errors_z.append(vel_error[2])
                else:
                    velocity_errors_x.append(0.0)
                    velocity_errors_y.append(0.0)
                    velocity_errors_z.append(0.0)
            
            if len(velocity_errors_x) > 0 and state_array.shape[1] > 11:
                axes2[2, 1].plot(tracking_error_time, velocity_errors_x, 'b-', linewidth=2, label='VX Error')
                axes2[2, 1].plot(tracking_error_time, velocity_errors_y, 'r-', linewidth=2, label='VY Error')
                axes2[2, 1].plot(tracking_error_time, velocity_errors_z, 'g-', linewidth=2, label='VZ Error')
                axes2[2, 1].set_xlabel('Time (s)')
                axes2[2, 1].set_ylabel('Velocity Error (m/s)')
                axes2[2, 1].legend(fontsize=8)
                axes2[2, 1].grid(True)
                axes2[2, 1].set_title('Linear Velocity Errors (VX, VY, VZ)')
            else:
                axes2[2, 1].text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=axes2[2, 1].transAxes)
                axes2[2, 1].set_title('Velocity Errors (N/A)')
        else:
            axes2[2, 1].text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=axes2[2, 1].transAxes)
            axes2[2, 1].set_title('Velocity Errors (N/A)')
        
        # Combined tracking error summary (RMS values over time windows) - more dense sampling
        if gripper_pos_error_norm is not None and len(gripper_pos_error_norm) > 0:
            # Calculate RMS errors over sliding windows with smaller window size for denser sampling
            # Use 1/4 of MPC horizon or minimum 2 steps for more dense RMS calculation
            mpc_horizon_length = len(self.mpc_controller.solver.xs)
            window_size = max(2, mpc_horizon_length // 4)  # Smaller window for denser sampling
            step_size = max(1, window_size // 2)  # Overlapping windows for smoother curves
            
            rms_pos_base = []
            rms_pos_ee = []
            rms_att_base = []
            rms_att_ee = []
            time_windows = []
            
            for i in range(0, len(tracking_error_time) - window_size + 1, step_size):
                end_idx = min(i + window_size, len(tracking_error_time))
                time_windows.append(tracking_error_time[i:end_idx].mean())
                
                # Base position and attitude RMS
                rms_pos_base.append(np.sqrt(np.mean(position_error_norm[i:end_idx]**2)))
                rms_att_base.append(np.sqrt(np.mean(attitude_error_norm[i:end_idx]**2)))
                
                # End-effector position and pitch RMS
                rms_pos_ee.append(np.sqrt(np.mean(gripper_pos_error_norm[i:end_idx]**2)))
                rms_att_ee.append(np.sqrt(np.mean(np.abs(gripper_pitch_errors[i:end_idx])**2)))
            
            axes2[2, 2].plot(time_windows, rms_pos_base, 'b-', linewidth=2, marker='o', markersize=3, label='Base Position RMS')
            axes2[2, 2].plot(time_windows, rms_pos_ee, 'r-', linewidth=2, marker='s', markersize=3, label='EE Position RMS')
            axes2[2, 2].set_xlabel('Time (s)')
            axes2[2, 2].set_ylabel('RMS Error (m)')
            axes2[2, 2].legend(fontsize=8)
            axes2[2, 2].grid(True)
            axes2[2, 2].set_title('RMS Tracking Error Evolution')
        else:
            # Fallback: Calculate base RMS only when no gripper data - also use denser sampling
            if len(tracking_error_time) > 0:
                mpc_horizon_length = len(self.mpc_controller.solver.xs)
                window_size = max(2, mpc_horizon_length // 4)  # Smaller window for denser sampling
                step_size = max(1, window_size // 2)  # Overlapping windows for smoother curves
                
                rms_pos_base = []
                rms_att_base = []
                time_windows = []
                
                for i in range(0, len(tracking_error_time) - window_size + 1, step_size):
                    end_idx = min(i + window_size, len(tracking_error_time))
                    time_windows.append(tracking_error_time[i:end_idx].mean())
                    
                    # Base position and attitude RMS only
                    rms_pos_base.append(np.sqrt(np.mean(position_error_norm[i:end_idx]**2)))
                    rms_att_base.append(np.sqrt(np.mean(attitude_error_norm[i:end_idx]**2)))
                
                axes2[2, 2].plot(time_windows, rms_pos_base, 'b-', linewidth=2, marker='o', markersize=3, label='Base Position RMS')
                axes2[2, 2].plot(time_windows, rms_att_base, 'c-', linewidth=2, marker='^', markersize=3, label='Base Attitude RMS')
                axes2[2, 2].set_xlabel('Time (s)')
                axes2[2, 2].set_ylabel('RMS Error')
                axes2[2, 2].legend(fontsize=8)
                axes2[2, 2].grid(True)
                axes2[2, 2].set_title('Base RMS Tracking Error Evolution')
            else:
                axes2[2, 2].text(0.5, 0.5, 'No tracking error data', ha='center', va='center', transform=axes2[2, 2].transAxes)
                axes2[2, 2].set_title('RMS Error Summary (N/A)')
        
        plt.tight_layout()
        
        # Create third figure for performance metrics (MPC, L1, etc.)
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
        fig3.suptitle('Performance Metrics', fontsize=16)
        
        # MPC solve times and iterations
        if len(self.mpc_solve_times) > 0:
            mpc_time_steps = np.arange(len(self.mpc_solve_times)) * self.control_interval
            mpc_time_array = mpc_time_steps * self.simulation_dt
            
            # Create twin axis for iterations
            ax1 = axes3[0, 0]
            ax2 = ax1.twinx()
            
            # Plot solve time
            line1 = ax1.plot(mpc_time_array, np.array(self.mpc_solve_times) * 1000, 'r-', linewidth=2, label='Solve Time')
            ax1.axhline(y=self.control_dt * 1000, color='k', linestyle='--', alpha=0.7,
                       label=f'Control period ({self.control_dt*1000:.1f} ms)')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('MPC Solve Time (ms)', color='r')
            ax1.tick_params(axis='y', labelcolor='r')
            
            # Plot iterations
            if len(self.mpc_iterations) > 0:
                line2 = ax2.plot(mpc_time_array, self.mpc_iterations, 'b-', linewidth=2, label='Iterations')
                ax2.set_ylabel('MPC Iterations', color='b')
                ax2.tick_params(axis='y', labelcolor='b')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels() if len(self.mpc_iterations) > 0 else ([], [])
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax1.set_title('MPC Computational Performance')
            ax1.grid(True, alpha=0.3)
        
        # MPC cost evolution
        if len(self.mpc_costs) > 0:
            axes3[0, 1].plot(mpc_time_array, self.mpc_costs, 'b-', linewidth=2)
            axes3[0, 1].set_xlabel('Time (s)')
            axes3[0, 1].set_ylabel('MPC Cost')
            axes3[0, 1].set_title('MPC Cost Evolution')
            axes3[0, 1].grid(True)
        
        # Control effort (total control magnitude)
        control_effort = np.linalg.norm(control_array, axis=1)
        axes3[1, 0].plot(time_array, control_effort, 'g-', linewidth=2)
        axes3[1, 0].set_xlabel('Time (s)')
        axes3[1, 0].set_ylabel('Control Effort (N/Nm)')
        axes3[1, 0].set_title('Total Control Effort')
        axes3[1, 0].grid(True)
        
        # Trajectory progress or L1 performance
        if self.enable_l1_control and len(self.l1_compute_times) > 0:
            l1_time_steps = np.arange(len(self.l1_compute_times)) * self.control_interval
            l1_perf_time_array = l1_time_steps * self.simulation_dt
            axes3[1, 1].plot(l1_perf_time_array, np.array(self.l1_compute_times) * 1000, 'r-', linewidth=2)
            axes3[1, 1].set_xlabel('Time (s)')
            axes3[1, 1].set_ylabel('L1 Compute Time (ms)')
            axes3[1, 1].set_title('L1 Controller Performance')
        else:
            traj_index_array = np.array(self.trajectory_index_data)
            axes3[1, 1].plot(time_array, traj_index_array, 'c-', linewidth=2)
            axes3[1, 1].set_xlabel('Time (s)')
            axes3[1, 1].set_ylabel('Trajectory Index')
            axes3[1, 1].set_title('Reference Trajectory Progress')
        axes3[1, 1].grid(True)
        
        plt.tight_layout()
        
        # === Figure 4: L1 Disturbance Estimation ===
        fig4 = None
        if self.enable_l1_control and len(self.l1_sig_hat_data) > 0 and len(self.disturbance_data) > 0:
            # Determine the number of subplots needed based on available data
            disturbance_array = np.array(self.disturbance_data)
            l1_sig_hat_array = np.array(self.l1_sig_hat_data)
            l1_u_ad_array = np.array(self.l1_u_ad_data)
            
            # Determine how many data channels we have
            total_disturbance_channels = disturbance_array.shape[1] if len(disturbance_array.shape) > 1 else 0
            
            # Decide layout: if we have more than 6 channels, use 3 rows; otherwise 2 rows
            if total_disturbance_channels > 6:
                fig4, axes4 = plt.subplots(3, 3, figsize=(15, 8))
                plot_layout = (3, 3)
            else:
                fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
                plot_layout = (2, 3)
                
            fig4.suptitle('L1 Disturbance Estimation Performance', fontsize=16)
            
            # Create time arrays
            full_time_array = time_array  # All simulation steps
            
            # L1 time array - should match the actual L1 data collection timing
            # L1 data is collected at control intervals starting from l1_start_time
            l1_time_steps = np.arange(len(self.l1_sig_hat_data)) * self.control_interval  
            l1_time_array = l1_time_steps * self.simulation_dt
            
            # Define simulation phases for background coloring
            sim_start = full_time_array[0]
            sim_end = full_time_array[-1]
            l1_start = self.l1_start_time
            disturbance_start = self.disturbance_start_time
            
            # Define colors for different line types (consistent across all subplots)
            line_colors = {
                'actual': '#1f77b4',      # Blue for actual disturbance
                'estimate': '#ff7f0e',    # Orange for L1 estimate 
                'adaptive': '#2ca02c'     # Green for L1 adaptive control
            }
            
            # Channel definitions
            channel_info = [
                {'name': 'Force X', 'unit': 'N', 'sig_hat_idx': 8},
                {'name': 'Force Y', 'unit': 'N', 'sig_hat_idx': 9},
                {'name': 'Force Z', 'unit': 'N', 'sig_hat_idx': 10},
                {'name': 'Torque X', 'unit': 'Nm', 'sig_hat_idx': 11},
                {'name': 'Torque Y', 'unit': 'Nm', 'sig_hat_idx': 12},
                {'name': 'Torque Z', 'unit': 'Nm', 'sig_hat_idx': 13},
                {'name': 'Joint 1', 'unit': 'Nm', 'sig_hat_idx': 14},
                {'name': 'Joint 2', 'unit': 'Nm', 'sig_hat_idx': 15},
                {'name': 'Joint 3', 'unit': 'Nm', 'sig_hat_idx': 16}
            ]
            
            # Plot each available channel
            plotted_channels = 0
            handles_labels_collected = False
            legend_handles = []
            legend_labels = []
            phase_handles = []
            phase_labels = []
            
            for i in range(min(total_disturbance_channels, len(channel_info))):
                if plotted_channels >= plot_layout[0] * plot_layout[1]:
                    break
                    
                row = plotted_channels // plot_layout[1]
                col = plotted_channels % plot_layout[1]
                ax = axes4[row, col]
                
                # Add phase background colors (collect legend items only from first plot)
                if plotted_channels == 0:
                    # Manually add phase backgrounds and collect handles for legend
                    phase_colors = {
                        'mpc_only': 'lightblue',        # Phase 1: Pure MPC control
                        'mpc_disturb': 'lightcoral',    # Phase 2: MPC + disturbance  
                        'mpc_disturb_l1': 'lightgreen'  # Phase 3: MPC + disturbance + L1
                    }
                    phase_alpha = 0.2
                    
                    # Phase 1: Pure MPC control (before disturbance starts)
                    if disturbance_start > sim_start:
                        p1 = ax.axvspan(sim_start, disturbance_start, alpha=phase_alpha, 
                                      color=phase_colors['mpc_only'], zorder=0)
                        phase_handles.append(p1)
                        phase_labels.append('Phase 1: Pure MPC')
                    
                    # Phase 2: MPC + disturbance (disturbance active, L1 not yet active)
                    if l1_start > disturbance_start:
                        p2 = ax.axvspan(disturbance_start, l1_start, alpha=phase_alpha, 
                                      color=phase_colors['mpc_disturb'], zorder=0)
                        phase_handles.append(p2)
                        phase_labels.append('Phase 2: MPC + Disturbance')
                    
                    # Phase 3: MPC + disturbance + L1 (both disturbance and L1 active)
                    if sim_end > l1_start:
                        p3 = ax.axvspan(l1_start, sim_end, alpha=phase_alpha, 
                                      color=phase_colors['mpc_disturb_l1'], zorder=0)
                        phase_handles.append(p3)
                        phase_labels.append('Phase 3: MPC + Disturbance + L1')
                else:
                    # For other plots, just add backgrounds without legend
                    self._add_l1_phase_backgrounds(ax, sim_start, sim_end, l1_start, disturbance_start, show_legend=False)
                
                channel = channel_info[i]
                
                # Plot actual disturbance (full time)
                if i < disturbance_array.shape[1]:
                    line1, = ax.plot(full_time_array, disturbance_array[:, i], 
                                   color=line_colors['actual'], linestyle='-', linewidth=2, 
                                   label='Actual Disturbance', alpha=0.8)
                    if not handles_labels_collected:
                        legend_handles.append(line1)
                        legend_labels.append('Actual Disturbance')
                
                # Plot L1 estimation (sig_hat)
                sig_hat_idx = channel['sig_hat_idx']
                if sig_hat_idx < l1_sig_hat_array.shape[1]:
                    sig_hat_data = l1_sig_hat_array[:, sig_hat_idx]
                    line2, = ax.plot(l1_time_array, sig_hat_data, 
                                   color=line_colors['estimate'], linestyle='--', linewidth=2, 
                                   label='L1 Estimate (σ_hat)', alpha=0.8)
                    if not handles_labels_collected:
                        legend_handles.append(line2)
                        legend_labels.append('L1 Estimate (σ_hat)')
                
                # Plot L1 adaptive control
                if i < l1_u_ad_array.shape[1]:
                    line3, = ax.plot(l1_time_array, -l1_u_ad_array[:, i], 
                                   color=line_colors['adaptive'], linestyle=':', linewidth=2, 
                                   label='L1 Adaptive (-u_ad)', alpha=0.8)
                    if not handles_labels_collected:
                        legend_handles.append(line3)
                        legend_labels.append('L1 Adaptive (-u_ad)')
                
                handles_labels_collected = True
                
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.set_ylabel(f'{channel["name"]} ({channel["unit"]})', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{channel["name"]} Disturbance Estimation', fontsize=11)
                
                plotted_channels += 1
            
            # Hide unused subplots
            for i in range(plotted_channels, plot_layout[0] * plot_layout[1]):
                row = i // plot_layout[1]
                col = i % plot_layout[1]
                axes4[row, col].set_visible(False)
            
            # Add unified legend below the title
            all_handles = legend_handles + phase_handles
            all_labels = legend_labels + phase_labels
            
            if all_handles:
                # Arrange in two rows: first row for line plots, second row for phases
                line_handles = legend_handles
                line_labels = legend_labels
                
                if line_handles and phase_handles:
                    # Create two-row legend: lines on top, phases on bottom
                    legend1 = fig4.legend(line_handles, line_labels, 
                                         loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                                         ncol=len(line_labels), fontsize=10, 
                                         frameon=True, fancybox=True, shadow=True)
                    legend2 = fig4.legend(phase_handles, phase_labels, 
                                         loc='upper center', bbox_to_anchor=(0.5, 0.88), 
                                         ncol=len(phase_labels), fontsize=10, 
                                         frameon=True, fancybox=True, shadow=True)
                    # Add the first legend back (matplotlib removes it when adding second)
                    fig4.add_artist(legend1)
                elif line_handles:
                    fig4.legend(line_handles, line_labels, 
                               loc='upper center', bbox_to_anchor=(0.5, 0.92), 
                               ncol=len(line_labels), fontsize=11, 
                               frameon=True, fancybox=True, shadow=True)
            
            # Adjust layout to make room for the legend
            plt.tight_layout()
            if phase_handles:
                # More space needed for two-row legend
                plt.subplots_adjust(top=0.80)
            else:
                # Less space for single-row legend
                plt.subplots_adjust(top=0.85)
        
        # Save or show plots
        if save_dir is not None:
            # Save all figures
            fig1_path = save_dir / '1_comprehensive_states_controls.png'
            fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"  - States and controls plot saved: {fig1_path}")
            
            fig2_path = save_dir / '2_tracking_errors.png'
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            print(f"  - Tracking errors plot saved: {fig2_path}")
            
            fig3_path = save_dir / '3_performance_metrics.png'
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            print(f"  - Performance metrics plot saved: {fig3_path}")
            
            # Save L1 disturbance estimation plot if it exists
            if fig4 is not None:
                fig4_path = save_dir / '4_l1_disturbance_estimation.png'
                fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
                print(f"  - L1 disturbance estimation plot saved: {fig4_path}")
            
            # Save pitch control analysis plot if it exists
            if self.simulation_actuation_mode == 'thrust_angular_velocity' and len(self.pitch_control_data) > 0:
                fig5 = self.plot_pitch_control_analysis()
                if fig5 is not None:
                    fig5_path = save_dir / '5_pitch_control_analysis.png'
                    fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
                    print(f"  - Pitch control analysis plot saved: {fig5_path}")
            
            # Close all figures to free memory
            plt.close('all')
        else:
            # Create pitch control analysis plot first (before showing other plots)
            fig5 = None
            if self.simulation_actuation_mode == 'thrust_angular_velocity' and len(self.pitch_control_data) > 0:
                fig5 = self.plot_pitch_control_analysis()
            
            # Show all plots interactively (including pitch control analysis)
            plt.show()
    
    def plot_pitch_control_analysis(self):
        """Plot pitch control analysis for thrust_angular_velocity mode"""
        if not self.pitch_control_data:
            print("No pitch control data available for plotting")
            return None
            
        print(f"Plotting pitch control analysis... (Data points: {len(self.pitch_control_data)})")
        
        # Extract data
        times = np.array([pd['time'] for pd in self.pitch_control_data])
        planned_pitch_torques = np.array([pd['planned_pitch_torque'] for pd in self.pitch_control_data])
        pid_pitch_torques = np.array([pd['pid_pitch_torque'] for pd in self.pitch_control_data])
        planned_pitch_rates = np.array([pd['planned_pitch_rate'] for pd in self.pitch_control_data])
        planned_pitch_rates_next = np.array([pd['planned_pitch_rate_next'] for pd in self.pitch_control_data])
        actual_pitch_rates = np.array([pd['actual_pitch_rate'] for pd in self.pitch_control_data])
        
        print(f"Time range: {times[0]:.3f} - {times[-1]:.3f} seconds")
        print(f"Pitch torque range: {np.min(planned_pitch_torques):.3f} - {np.max(planned_pitch_torques):.3f} N⋅m")
        print(f"Pitch rate range: {np.min(planned_pitch_rates):.3f} - {np.max(planned_pitch_rates):.3f} rad/s")
        
        # Convert angular rates from rad/s to deg/s for better readability
        planned_pitch_rates_deg = np.degrees(planned_pitch_rates)
        planned_pitch_rates_next_deg = np.degrees(planned_pitch_rates_next)
        actual_pitch_rates_deg = np.degrees(actual_pitch_rates)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Pitch Control Analysis (Thrust + Angular Velocity Mode)', fontsize=16)
        
        # Plot 1: Pitch Torques
        ax1 = axes[0]
        ax1.plot(times, planned_pitch_torques, 'b-', linewidth=2, label='Planned Pitch Torque (MPC)', alpha=0.8)
        ax1.plot(times, pid_pitch_torques, 'r-', linewidth=2, label='PID Pitch Torque (Applied)', alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pitch Torque (N⋅m)')
        ax1.set_title('Pitch Torque Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        planned_mean = np.mean(np.abs(planned_pitch_torques))
        pid_mean = np.mean(np.abs(pid_pitch_torques))
        ax1.text(0.02, 0.98, f'Planned Mean: {planned_mean:.3f} N⋅m\nPID Mean: {pid_mean:.3f} N⋅m', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Pitch Rates
        ax2 = axes[1]
        ax2.plot(times, planned_pitch_rates_deg, 'g-', linewidth=2, label='Planned Pitch Rate', alpha=0.8)
        ax2.plot(times, planned_pitch_rates_next_deg, 'c--', linewidth=1.5, label='Planned Pitch Rate (Next)', alpha=0.7)
        ax2.plot(times, actual_pitch_rates_deg, 'k-', linewidth=2, label='Actual Pitch Rate', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pitch Rate (deg/s)')
        ax2.set_title('Pitch Rate Tracking')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate tracking error
        pitch_rate_error = planned_pitch_rates_deg - actual_pitch_rates_deg
        rmse = np.sqrt(np.mean(pitch_rate_error**2))
        max_error = np.max(np.abs(pitch_rate_error))
        
        ax2.text(0.02, 0.98, f'RMSE: {rmse:.2f} deg/s\nMax Error: {max_error:.2f} deg/s', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        return fig
    
    def plot_planned_trajectory(self, save_dir=None):
        """Plot only the planned trajectory without simulation results"""
        print("Plotting planned trajectory...")
        
        # Create time array for the full trajectory
        total_time = self.simulation_time
        time_array = np.arange(0, total_time, self.simulation_dt)
        
        # Create reference trajectory
        ref_states = []
        for t in time_array:
            ref_state, _ = self.get_reference_state(t)
            ref_states.append(ref_state)
        ref_state_array = np.array(ref_states)
        
        # Convert quaternions to Euler angles
        from scipy.spatial.transform import Rotation as R
        ref_quats = ref_state_array[:, 3:7]
        ref_euler = R.from_quat(ref_quats).as_euler('xyz', degrees=True)
        
        # === Figure: Planned Trajectory ===
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f'Planned Trajectory - {self.robot_name} ({self.trajectory_name})', fontsize=16)
        
        # === First Row: Position and Attitude ===
        # Position (X, Y, Z)
        position_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        position_colors = ['b', 'r', 'g']
        for i in range(3):
            ax = axes[0, i]
            ax.plot(time_array, ref_state_array[:, i], color=position_colors[i], linewidth=2, label='Planned')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel(position_labels[i], fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(position_labels[i], fontsize=10)
        
        # === Second Row: Attitude ===
        # Attitude (Pitch, Roll, Yaw)
        attitude_labels = ['Pitch (deg)', 'Roll (deg)', 'Yaw (deg)']
        angle_order = [1, 0, 2]  # pitch, roll, yaw from xyz Euler
        for i in range(3):
            ax = axes[1, i]
            angle_idx = angle_order[i]
            ax.plot(time_array, ref_euler[:, angle_idx], color=position_colors[i], linewidth=2, label='Planned')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel(attitude_labels[i], fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(attitude_labels[i], fontsize=10)
        
        # === Third Row: Arm Joints or Velocities ===
        if self.robot.nq > 7:  # Has arm joints
            # Arm joint angles
            joint_labels = ['Joint 1 (deg)', 'Joint 2 (deg)']
            for i in range(2):
                ax = axes[2, i]
                if (7 + i) < ref_state_array.shape[1]:
                    ax.plot(time_array, np.degrees(ref_state_array[:, 7 + i]), 
                           color=position_colors[i], linewidth=2, label='Planned')
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.set_ylabel(joint_labels[i], fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(joint_labels[i], fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(joint_labels[i] + ' (N/A)', fontsize=10)
            
            # 3D trajectory plot
            ax = fig.add_subplot(3, 3, 9, projection='3d')
            ax.plot(ref_state_array[:, 0], ref_state_array[:, 1], ref_state_array[:, 2], 
                   'b-', linewidth=2, label='Planned Path')
            ax.scatter(ref_state_array[0, 0], ref_state_array[0, 1], ref_state_array[0, 2], 
                      color='g', s=100, label='Start', marker='o')
            ax.scatter(ref_state_array[-1, 0], ref_state_array[-1, 1], ref_state_array[-1, 2], 
                      color='r', s=100, label='End', marker='s')
            ax.set_xlabel('X (m)', fontsize=9)
            ax.set_ylabel('Y (m)', fontsize=9)
            ax.set_zlabel('Z (m)', fontsize=9)
            ax.set_ylim(-1.0, 1.0)
            ax.legend(fontsize=8)
            ax.set_title('3D Trajectory', fontsize=10)
        else:
            # Linear velocities for non-arm robots
            velocity_labels = ['VX (m/s)', 'VY (m/s)', 'VZ (m/s)']
            velocity_indices = [9, 10, 11]
            for i in range(3):
                ax = axes[2, i]
                vel_idx = velocity_indices[i]
                if vel_idx < ref_state_array.shape[1]:
                    ax.plot(time_array, ref_state_array[:, vel_idx], 
                           color=position_colors[i], linewidth=2, label='Planned')
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.set_ylabel(velocity_labels[i], fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(velocity_labels[i], fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(velocity_labels[i] + ' (N/A)', fontsize=10)
        
        plt.tight_layout()
        
        # Save or show plot
        if save_dir is not None:
            fig_path = save_dir / '0_planned_trajectory.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"  - Planned trajectory plot saved: {fig_path}")
            plt.close(fig)
        else:
            plt.show()

    def save_and_plot(self):
        """Save simulation results to files"""
        if not self.save_results and self.plot_results:
            self.plot_result()
            return
        
        print("Saving results...")
        
        # Create results directory
        results_dir = Path(self.results_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = results_dir / f"{self.robot_name}_{self.trajectory_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_data = False
        if self.save_data:
            # Convert data to numpy arrays
            time_array = np.array(self.time_data)
            state_array = np.array(self.state_data)
            control_array = np.array(self.control_data)
            
            # Save main data
            np.save(save_dir / 'time_data.npy', time_array)
            np.save(save_dir / 'state_data.npy', state_array)
            np.save(save_dir / 'control_data.npy', control_array)
            np.save(save_dir / 'mpc_solve_times.npy', self.mpc_solve_times)
            np.save(save_dir / 'mpc_costs.npy', self.mpc_costs)
            np.save(save_dir / 'mpc_iterations.npy', self.mpc_iterations)
            np.save(save_dir / 'trajectory_index_data.npy', self.trajectory_index_data)
            
            # Save disturbance data
            np.save(save_dir / 'disturbance_data.npy', self.disturbance_data)
            
            # Save pitch control data if available (for thrust_angular_velocity mode)
            if len(self.pitch_control_data) > 0:
                pitch_times = np.array([pd['time'] for pd in self.pitch_control_data])
                planned_pitch_torques = np.array([pd['planned_pitch_torque'] for pd in self.pitch_control_data])
                pid_pitch_torques = np.array([pd['pid_pitch_torque'] for pd in self.pitch_control_data])
                planned_pitch_rates = np.array([pd['planned_pitch_rate'] for pd in self.pitch_control_data])
                planned_pitch_rates_next = np.array([pd['planned_pitch_rate_next'] for pd in self.pitch_control_data])
                actual_pitch_rates = np.array([pd['actual_pitch_rate'] for pd in self.pitch_control_data])
                
                np.save(save_dir / 'pitch_control_times.npy', pitch_times)
                np.save(save_dir / 'planned_pitch_torques.npy', planned_pitch_torques)
                np.save(save_dir / 'pid_pitch_torques.npy', pid_pitch_torques)
                np.save(save_dir / 'planned_pitch_rates.npy', planned_pitch_rates)
                np.save(save_dir / 'planned_pitch_rates_next.npy', planned_pitch_rates_next)
                np.save(save_dir / 'actual_pitch_rates.npy', actual_pitch_rates)
            
            # Save L1 controller data if available
            if self.enable_l1_control and len(self.l1_u_ad_data) > 0:
                np.save(save_dir / 'l1_u_baseline_data.npy', self.l1_u_baseline_data)
                np.save(save_dir / 'l1_u_ad_data.npy', self.l1_u_ad_data)
                np.save(save_dir / 'l1_compute_times.npy', self.l1_compute_times)
                
                if len(self.l1_z_tilde_data) > 0:
                    np.save(save_dir / 'l1_z_tilde_data.npy', self.l1_z_tilde_data)
                if len(self.l1_sig_hat_data) > 0:
                    np.save(save_dir / 'l1_sig_hat_data.npy', self.l1_sig_hat_data)
                if len(self.l1_z_hat_data) > 0:
                    np.save(save_dir / 'l1_z_hat_data.npy', self.l1_z_hat_data)
            
            # Save tracking error data if available
            if len(self.tracking_error_data) > 0:
                # Convert tracking error data to numpy format
                tracking_error_times = np.array([te['time'] for te in self.tracking_error_data])
                position_errors = np.array([te['position_error'] for te in self.tracking_error_data])
                attitude_errors = np.array([te['attitude_error'] for te in self.tracking_error_data])
                position_error_norms = np.array([te['position_error_norm'] for te in self.tracking_error_data])
                attitude_error_norms = np.array([te['attitude_error_norm'] for te in self.tracking_error_data])
                arm_joint_errors = np.array([te['arm_joint_error'] for te in self.tracking_error_data])
                arm_joint_error_norms = np.array([te['arm_joint_error_norm'] for te in self.tracking_error_data])
                gripper_pos_errors = np.array([te['gripper_pos_error'] for te in self.tracking_error_data])
                gripper_pos_error_norms = np.array([te['gripper_pos_error_norm'] for te in self.tracking_error_data])
                gripper_pitch_errors = np.array([te['gripper_pitch_error'] for te in self.tracking_error_data])
                
                # Save tracking error arrays
                np.save(save_dir / 'tracking_error_times.npy', tracking_error_times)
                np.save(save_dir / 'position_errors.npy', position_errors)
                np.save(save_dir / 'attitude_errors.npy', attitude_errors)
                np.save(save_dir / 'position_error_norms.npy', position_error_norms)
                np.save(save_dir / 'attitude_error_norms.npy', attitude_error_norms)
                np.save(save_dir / 'arm_joint_errors.npy', arm_joint_errors)
                np.save(save_dir / 'arm_joint_error_norms.npy', arm_joint_error_norms)
                np.save(save_dir / 'gripper_pos_errors.npy', gripper_pos_errors)
                np.save(save_dir / 'gripper_pos_error_norms.npy', gripper_pos_error_norms)
                np.save(save_dir / 'gripper_pitch_errors.npy', gripper_pitch_errors)
        
        # Save configuration
        config_dict = {
            'robot_name': self.robot_name,
            'trajectory_name': self.trajectory_name,
            'simulation_dt': self.simulation_dt,
            'control_dt': self.control_dt,
            'simulation_time': self.simulation_time,
            'initial_state': self.initial_state,
            'enable_disturbance': self.enable_disturbance,
            'enable_noise': self.enable_noise,
            'enable_l1_control': self.enable_l1_control,
            'control_delay': self.control_delay
        }
        
        # Add L1 configuration if enabled
        if self.enable_l1_control:
            config_dict.update({
                'l1_version': self.l1_version,
                'l1_start_time': self.l1_start_time,
                'l1_as_coef': self.l1_as_coef,
                'l1_filter_time_constant': self.l1_filter_time_constant
            })
        
        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Save summary statistics
        stats = {
            'total_simulation_steps': len(self.time_data),
            'total_mpc_updates': len(self.mpc_solve_times),
            'avg_mpc_solve_time_ms': float(np.mean(self.mpc_solve_times) * 1000),
            'max_mpc_solve_time_ms': float(np.max(self.mpc_solve_times) * 1000),
            'avg_mpc_cost': float(np.mean(self.mpc_costs)),
            'final_mpc_cost': float(self.mpc_costs[-1]) if self.mpc_costs else 0.0,
            'avg_mpc_iterations': float(np.mean(self.mpc_iterations)) if self.mpc_iterations else 0.0,
            'max_mpc_iterations': int(np.max(self.mpc_iterations)) if self.mpc_iterations else 0
        }
        
        # Add L1 statistics if available
        if self.enable_l1_control and len(self.l1_compute_times) > 0:
            stats.update({
                'total_l1_updates': len(self.l1_compute_times),
                'avg_l1_compute_time_ms': float(np.mean(self.l1_compute_times) * 1000),
                'max_l1_compute_time_ms': float(np.max(self.l1_compute_times) * 1000),
                'l1_start_time': self.l1_start_time,
                'avg_l1_adaptive_magnitude': float(np.mean(np.linalg.norm(self.l1_u_ad_data, axis=1))) if len(self.l1_u_ad_data) > 0 else 0.0
            })
        
        # Add control delay statistics
        if self.control_delay > 0:
            stats.update({
                'control_delay_seconds': self.control_delay,
                'control_delay_steps': self.delay_steps,
                'delayed_control_entries': len(self.delayed_control_data)
            })
        
        with open(save_dir / 'statistics.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        # Save plots as well
        print("Generating and saving plots...")
        self.plot_result(save_dir)
        
        print(f"Results saved to: {save_dir}")
        
        return save_dir


def create_default_config(filename='numeric_sim_config.yaml'):
    """Create a default configuration file"""
    default_config = {
        'robot_name': 's500_uam',
        'trajectory_name': 'hover',
        'dt_traj_opt': 50,
        'use_squash': True,
        'yaml_path': 'config/yaml',
        'simulation_time': 10.0,
        'initial_state': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.2, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'enable_disturbance': False,
        'disturbance_start_time': 5.0,
        'disturbance_force': [0.0, 0.0, 0.0],
        'disturbance_torque': [0.0, 0.0, 0.0],
        'enable_joint_sin_disturbance': False,
        'joint_sin_amplitude': [0.1, 0.05],  # amplitude for joint1, joint2 (Nm)
        'joint_sin_frequency': [2.0, 1.5],   # frequency for joint1, joint2 (Hz)
        'enable_noise': False,
        'state_noise_std': 0.001,
        'save_results': True,
        'plot_results': True,
        'results_dir': 'results/numeric_sim'
    }
    
    with open(filename, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Default configuration created: {filename}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Numeric simulation for MPC controller')
    parser.add_argument('--config', type=str, default='numeric_sim_config_catch_fast.yaml',
                       help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--robot', type=str, default=None,
                       help='Override robot name from config')
    parser.add_argument('--trajectory', type=str, default=None,
                       help='Override trajectory name from config')
    parser.add_argument('--sim-time', type=float, default=None,
                       help='Override simulation time from config')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving results')
    parser.add_argument('--plot-trajectory-only', action='store_true',
                       help='Only plot the planned trajectory without running simulation')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_default_config(args.config)
        return
    
    try:
        # Create simulator
        simulator = NumericSimulator(args.config)
        
        # Apply command line overrides
        if args.robot:
            simulator.robot_name = args.robot
        if args.trajectory:
            simulator.trajectory_name = args.trajectory
        if args.sim_time:
            simulator.simulation_time = args.sim_time
        if args.no_plot:
            simulator.plot_results = False
        if args.no_save:
            simulator.save_results = False
        
        # Check if only plotting trajectory
        if args.plot_trajectory_only:
            print("Plotting planned trajectory only (no simulation)...")
            
            # Create save directory if saving is enabled
            save_dir = None
            if simulator.save_results:
                from datetime import datetime
                results_dir = Path(simulator.results_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = results_dir / f"{simulator.robot_name}_{simulator.trajectory_name}_trajectory_{timestamp}"
                save_dir.mkdir(parents=True, exist_ok=True)
                print(f"Trajectory plot will be saved to: {save_dir}")
            
            # Plot only the planned trajectory
            simulator.plot_planned_trajectory(save_dir)
            print("Trajectory plotting completed!")
            return
        
        # Run simulation
        simulator.simulate()
        
        # Plot results
        # simulator.plot_result()
        
        # Save results
        save_dir = simulator.save_and_plot()
        
        print(f"\nSimulation completed successfully!")
        if save_dir:
            print(f"Results saved to: {save_dir}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise


def test_pid_step_response():
    """
    Test PID controller step response for parameter tuning
    """
    import matplotlib.pyplot as plt
    
    # PID parameters to test (adjusted for better numerical stability)
    test_cases = [
        {'kp': 5.0, 'ki': 0.0, 'kd': 0.0, 'name': 'Default'},
        # {'kp': 2.0, 'ki': 0.05, 'kd': 0.05, 'name': 'Conservative'},
        # {'kp': 8.0, 'ki': 0.2, 'kd': 0.2, 'name': 'Aggressive'},
        {'kp': 0.25, 'ki': 0.2, 'kd': 0.003, 'name': 'PX4 default'},
    ]
    
    dt = 0.001   # 10ms simulation step (reduced for stability)
    duration = 2.0  # 2 seconds test
    time_steps = int(duration / dt)
    
    setpoint = 0.5  # rad/s step input
    step_time = 0.2  # Step occurs at 0.2s
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PID Controller Step Response Test', fontsize=14)
    
    # Real quadrotor pitch dynamics: I * dω/dt = M_pitch + damping
    # For a typical small quadrotor like S500
    I_yy = 0.015  # kg⋅m² (moment of inertia around pitch axis)
    damping_coeff = 0.005  # Nm⋅s/rad (aerodynamic damping)
    
    for i, params in enumerate(test_cases):
        # Initialize PID controller
        pid = PIDController(params['kp'], params['ki'], params['kd'], dt)
        
        # Simulate step response
        time_array = []
        setpoint_array = []
        output_array = []
        response_array = []
        current_value = 0.0  # Start at zero angular velocity
        
        for step in range(time_steps):
            current_time = step * dt
            time_array.append(current_time)
            
            # Apply step input at step_time
            if current_time >= step_time:
                current_setpoint = setpoint
            else:
                current_setpoint = 0.0
            setpoint_array.append(current_setpoint)
            
            # PID output (torque)
            torque_output = pid.update(current_setpoint, current_value)
            
            # Real quadrotor dynamics: I * dω/dt = M_pid - damping * ω
            # Clamp torque_output to prevent numerical issues
            torque_output = np.clip(torque_output, -1.0, 1.0)  # Realistic torque limits
            
            output_array.append(torque_output)
            
            # Check for valid values before integration
            if np.isfinite(torque_output) and np.isfinite(current_value):
                # Angular acceleration: α = (M_pid - damping * ω) / I
                angular_accel = (torque_output - damping_coeff * current_value) / I_yy
                # Euler integration: ω(k+1) = ω(k) + α * dt
                current_value = current_value + angular_accel * dt
                # Clamp current_value to reasonable range
                current_value = np.clip(current_value, -20.0, 20.0)  # rad/s
            else:
                # Reset to zero if numerical issues occur
                current_value = 0.0
                
            response_array.append(current_value)
        
        # Convert to numpy arrays and clean data
        time_array = np.array(time_array)
        setpoint_array = np.array(setpoint_array) 
        output_array = np.array(output_array)
        response_array = np.array(response_array)
        
        # Remove any NaN or inf values
        valid_mask = np.isfinite(time_array) & np.isfinite(setpoint_array) & np.isfinite(output_array) & np.isfinite(response_array)
        time_array = time_array[valid_mask]
        setpoint_array = setpoint_array[valid_mask]
        output_array = output_array[valid_mask]
        response_array = response_array[valid_mask]
        
        if len(time_array) > 0:  # Only plot if we have valid data
            # Plot angular velocity response
            axes[0, 0].plot(time_array, setpoint_array, 'k--', linewidth=2, label='Setpoint' if i == 0 else '')
            axes[0, 0].plot(time_array, response_array, linewidth=2, label=f'{params["name"]} (Kp={params["kp"]}, Ki={params["ki"]}, Kd={params["kd"]})')
            
            # Plot control output (torque)
            axes[0, 1].plot(time_array, output_array, linewidth=2, label=f'{params["name"]}')
        
        # Calculate performance metrics
        step_idx = int(step_time / dt)
        if step_idx < len(response_array):
            # Settling time (within 2% of final value)
            final_value = setpoint
            settling_tolerance = 0.02 * abs(final_value)
            settling_idx = None
            for j in range(step_idx, len(response_array)):
                if abs(response_array[j] - final_value) <= settling_tolerance:
                    # Check if it stays within tolerance for at least 50ms
                    check_duration = int(0.05 / dt)
                    if j + check_duration < len(response_array):
                        within_tolerance = True
                        for k in range(j, min(j + check_duration, len(response_array))):
                            if abs(response_array[k] - final_value) > settling_tolerance:
                                within_tolerance = False
                                break
                        if within_tolerance:
                            settling_idx = j
                            break
            
            settling_time = time_array[settling_idx] - step_time if settling_idx else duration - step_time
            
            # Overshoot
            max_value = np.max(response_array[step_idx:])
            overshoot = max(0, (max_value - final_value) / final_value * 100) if final_value != 0 else 0
            
            # Rise time (10% to 90% of final value)
            val_10 = 0.1 * final_value
            val_90 = 0.9 * final_value
            rise_start_idx = np.where(response_array[step_idx:] >= val_10)[0]
            rise_end_idx = np.where(response_array[step_idx:] >= val_90)[0]
            
            rise_time = 0
            if len(rise_start_idx) > 0 and len(rise_end_idx) > 0:
                rise_time = time_array[step_idx + rise_end_idx[0]] - time_array[step_idx + rise_start_idx[0]]
            
            print(f"{params['name']} PID (Kp={params['kp']}, Ki={params['ki']}, Kd={params['kd']}):")
            print(f"  Settling time: {settling_time:.3f}s")
            print(f"  Overshoot: {overshoot:.1f}%")
            print(f"  Rise time: {rise_time:.3f}s")
            print()
    
    # Format plots
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 0].set_title('Step Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Torque Output (Nm)')
    axes[0, 1].set_title('Control Signal')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Add error plot
    for i, params in enumerate(test_cases):
        pid = PIDController(params['kp'], params['ki'], params['kd'], dt)
        error_array = []
        current_value = 0.0
        
        for step in range(time_steps):
            current_time = step * dt
            
            if current_time >= step_time:
                current_setpoint = setpoint
            else:
                current_setpoint = 0.0
                
            error = current_setpoint - current_value
            error_array.append(error)
            
            torque_output = pid.update(current_setpoint, current_value)
            # Clamp and check for numerical stability  
            torque_output = np.clip(torque_output, -5.0, 5.0)
            if np.isfinite(torque_output) and np.isfinite(current_value):
                angular_accel = (torque_output - damping_coeff * current_value) / I_yy
                current_value = current_value + angular_accel * dt
                current_value = np.clip(current_value, -20.0, 20.0)
            else:
                current_value = 0.0
        
        axes[1, 0].plot(time_array, error_array, linewidth=2, label=f'{params["name"]}')
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (rad/s)')
    axes[1, 0].set_title('Tracking Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Add phase plot (error vs error rate)
    for i, params in enumerate(test_cases):
        pid = PIDController(params['kp'], params['ki'], params['kd'], dt)
        error_array = []
        error_rate_array = []
        current_value = 0.0
        last_error = 0.0
        
        for step in range(time_steps):
            current_time = step * dt
            
            if current_time >= step_time:
                current_setpoint = setpoint
            else:
                current_setpoint = 0.0
                
            error = current_setpoint - current_value
            if step > 0 and dt > 0:
                error_rate = (error - last_error) / dt
                # Clamp error rate to prevent overflow
                error_rate = np.clip(error_rate, -1000.0, 1000.0)
            else:
                error_rate = 0.0
            
            error_array.append(error)
            error_rate_array.append(error_rate)
            
            torque_output = pid.update(current_setpoint, current_value)
            # Clamp and check for numerical stability
            torque_output = np.clip(torque_output, -5.0, 5.0)
            if np.isfinite(torque_output) and np.isfinite(current_value):
                angular_accel = (torque_output - damping_coeff * current_value) / I_yy
                current_value = current_value + angular_accel * dt
                current_value = np.clip(current_value, -20.0, 20.0)
            else:
                current_value = 0.0
            last_error = error
        
        axes[1, 1].plot(error_array, error_rate_array, linewidth=2, label=f'{params["name"]}')
    
    axes[1, 1].set_xlabel('Error (rad/s)')
    axes[1, 1].set_ylabel('Error Rate (rad/s²)')
    axes[1, 1].set_title('Phase Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Check for valid data before plotting
    try:
        plt.tight_layout()
        
        # Save the plot
        save_path = "pid_step_response_test.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PID step response test plot saved to: {save_path}")
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {e}")
        # Try saving without tight_layout
        try:
            save_path = "pid_step_response_test.png"
            plt.savefig(save_path, dpi=300)
            print(f"PID step response test plot saved to: {save_path} (without tight layout)")
            plt.show()
        except Exception as e2:
            print(f"Failed to save plot: {e2}")
            plt.close('all')


if __name__ == "__main__":
    # Check if we're running PID test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-pid":
        test_pid_step_response()
    else:
        main()
