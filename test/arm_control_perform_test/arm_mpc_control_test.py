#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-12-19
Description: 2DOF robotic arm angle tracking optimization problem using Crocoddyl MPC controller
Supports both simulation mode and ROS node mode
'''

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import sys

from scipy import linalg as sLA

# Try to import crocoddyl and related packages (only needed for MPC mode)
try:
    import crocoddyl
    import example_robot_data
    import gepetto
    CROCODDYL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Crocoddyl not available: {e}")
    print("MPC mode will not be available. Only PID mode can be used.")
    CROCODDYL_AVAILABLE = False

# Cascaded P+PID Controller Class (Position P controller + Velocity PID controller)
class CascadedPIDController:
    """
    Cascaded P+PID Controller for 2DOF robotic arm
    Outer loop: Position P controller
    Inner loop: Velocity PID controller
    """
    def __init__(self, kp_pos, kp_vel, ki_vel, kd_vel, dt, joint_names=None, velocity_limits=None):
        """
        Initialize Cascaded P+PID Controller
        
        Args:
            kp_pos: Position proportional gains [joint1, joint2] (outer loop)
            kp_vel: Velocity proportional gains [joint1, joint2] (inner loop)
            ki_vel: Velocity integral gains [joint1, joint2] (inner loop)
            kd_vel: Velocity derivative gains [joint1, joint2] (inner loop)
            dt: Time step
            joint_names: Joint names
            velocity_limits: Maximum allowed velocities [joint1, joint2] (rad/s)
        """
        # Position controller gains (outer loop - P controller)
        self.kp_pos = np.array(kp_pos)
        
        # Velocity controller gains (inner loop - PID controller)
        self.kp_vel = np.array(kp_vel)
        self.ki_vel = np.array(ki_vel)
        self.kd_vel = np.array(kd_vel)
        
        self.dt = dt
        self.joint_names = joint_names or ['joint_1', 'joint_2']
        
        # Velocity constraints
        self.velocity_limits = np.array(velocity_limits) if velocity_limits is not None else np.array([2.0, 2.0])  # Default limits
        self.enable_velocity_constraints = velocity_limits is not None
        
        # Initialize velocity controller state (inner loop)
        self.vel_error_integral = np.zeros(2)
        self.prev_vel_error = np.zeros(2)
        self.prev_velocity = np.zeros(2)
        self.prev_time = None
        
        # Anti-windup limits
        self.vel_integral_limit = 1.0
        self.output_limit = 10
        
        print(f"Cascaded P+PID Controller initialized:")
        print(f"  Position controller gains (Kp_pos): {self.kp_pos}")
        print(f"  Velocity controller gains (Kp_vel): {self.kp_vel}")
        print(f"  Velocity controller gains (Ki_vel): {self.ki_vel}")
        print(f"  Velocity controller gains (Kd_vel): {self.kd_vel}")
        print(f"  dt: {self.dt}")
        print(f"  Joint names: {self.joint_names}")
        print(f"  Velocity limits: {self.velocity_limits} rad/s")
        print(f"  Velocity constraints enabled: {self.enable_velocity_constraints}")
    
    def reset(self):
        """Reset cascaded controller state"""
        self.vel_error_integral = np.zeros(2)
        self.prev_vel_error = np.zeros(2)
        self.prev_velocity = np.zeros(2)
        self.prev_time = None
        print("Cascaded P+PID Controller reset")
    
    def compute_control(self, current_positions, target_positions, current_velocities=None):
        """
        Compute cascaded P+PID control output
        
        Args:
            current_positions: Current joint positions [joint1, joint2]
            target_positions: Target joint positions [joint1, joint2]
            current_velocities: Current joint velocities [joint1, joint2] (required for cascaded control)
            
        Returns:
            control_output: Control torques [joint1, joint2]
        """
        current_positions = np.array(current_positions)
        target_positions = np.array(target_positions)
        
        if current_velocities is None:
            raise ValueError("Current velocities are required for cascaded P+PID control")
        
        current_velocities = np.array(current_velocities)
        
        # Calculate time step
        current_time = time.time()
        if self.prev_time is None:
            dt = self.dt
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # OUTER LOOP: Position P controller
        # Calculate position errors
        position_errors = target_positions - current_positions
        
        # Generate desired velocities using position P controller
        desired_velocities = self.kp_pos * position_errors
        
        # Apply velocity limits to desired velocities
        if self.enable_velocity_constraints:
            desired_velocities = np.clip(desired_velocities, -self.velocity_limits, self.velocity_limits)
        
        # INNER LOOP: Velocity PID controller
        # Calculate velocity errors
        velocity_errors = desired_velocities - current_velocities
        
        # Update velocity integral term with anti-windup
        self.vel_error_integral += velocity_errors * dt
        self.vel_error_integral = np.clip(self.vel_error_integral, -self.vel_integral_limit, self.vel_integral_limit)
        
        # Calculate velocity derivative term
        velocity_derivative = (velocity_errors - self.prev_vel_error) / dt if dt > 0 else np.zeros(2)
        
        # Velocity PID control law
        control_output = (self.kp_vel * velocity_errors + 
                         self.ki_vel * self.vel_error_integral + 
                         self.kd_vel * velocity_derivative)
        
        # Apply output limits
        control_output = np.clip(control_output, -self.output_limit, self.output_limit)
        
        # Update previous states
        self.prev_vel_error = velocity_errors.copy()
        self.prev_velocity = current_velocities.copy()
        
        return control_output
    
    def get_control_debug_info(self):
        """Get debug information for the cascaded controller"""
        return {
            'position_gains': self.kp_pos,
            'velocity_gains': {'kp': self.kp_vel, 'ki': self.ki_vel, 'kd': self.kd_vel},
            'vel_error_integral': self.vel_error_integral.copy(),
            'prev_vel_error': self.prev_vel_error.copy(),
            'prev_velocity': self.prev_velocity.copy(),
            'velocity_limits': self.velocity_limits,
            'velocity_constraints_enabled': self.enable_velocity_constraints
        }
    
    def get_debug_info(self):
        """Get debug information for the cascaded controller (alias for get_control_debug_info)"""
        return self.get_control_debug_info()

# ROS imports (only used when running in ROS mode)
try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. ROS mode will be disabled.")
    
class L1AdaptiveController:
    def __init__(self, dt, robot_model, adaptation_gain=50.0, filter_time_constants=None, 
                 friction_threshold=0.01, enable_static_compensation=True):
        self.dt = dt
        
        self.robot_model = robot_model
        self.robot_model_data = self.robot_model.createData()
        
        self.adaptation_gain = adaptation_gain
        self.filter_time_constants = filter_time_constants
        
        self.state_dim = self.robot_model.nq + self.robot_model.nv
        self.control_dim = self.robot_model.nv
        
        # self.A_s = np.zeros((self.state_dim, self.state_dim))
        self.A_s = np.eye(self.state_dim) * self.adaptation_gain
        
        self.expm_A_s_dt = sLA.expm(self.A_s * self.dt)
        
        self.PHI_diag = (self.expm_A_s_dt - np.identity(self.state_dim)).diagonal() / self.adaptation_gain
        self.PHI_inv_diag = 1.0 / self.PHI_diag
        
        self.M_aug_template = np.zeros((self.state_dim, self.state_dim))
        self.M_aug_template[:self.control_dim, :self.control_dim] = np.diag(np.ones(self.control_dim))
        
        self.init_controller()
        
    def init_controller(self):
        self.current_state = np.zeros(self.state_dim)
        self.z_hat = np.zeros(self.state_dim)
        self.z_tilde = np.zeros(self.state_dim)
        
        self.sig_hat = np.zeros(self.state_dim)
        self.sig_hat_prev = np.zeros(self.state_dim)
        
        self.u_ad = np.zeros(self.control_dim)
    
    def solve_l1_problem(self, current_state, u_baseline):
        # 0. update states
        self.update_z_tilde(current_state)
        # 1. get z_hat
        self.update_z_hat(current_state, u_baseline)
        # 2. get sigma_hat
        self.update_sig_hat(current_state)
        # 3. low pass filter
        self.update_u_ad()
    
    def update_z_tilde(self, current_state):
        self.z_tilde = self.z_hat - current_state  # all states
        
    def update_z_hat(self, current_state, u_b):
        # only for velocity
        z_hat_prev = self.z_hat.copy()
        
        tau_body = u_b + self.u_ad + self.sig_hat[self.control_dim:]
        
        model = self.robot_model
        data = self.robot_model_data
        q = current_state[:self.robot_model.nq]    # state
        v = current_state[self.robot_model.nq:]    # velocity
        
        z_hat_dot_without_disturb = pin.aba(model, data, q, v, tau_body)
        
        z_hat_dot_disturb = self.A_s[self.control_dim:, self.control_dim:] @ self.z_tilde[self.control_dim:]  # get a using the current state
        
        z_hat_dot_vel = z_hat_dot_without_disturb.copy() + z_hat_dot_disturb.copy()
        
        self.z_hat[self.control_dim:] = z_hat_prev[self.control_dim:].copy() + self.dt * z_hat_dot_vel.copy()
    
    
    def update_sig_hat(self, current_state):
        q = current_state[:self.robot_model.nq]    # state
        M = pin.crba(self.robot_model, self.robot_model_data, q)
        
        M_aug = self.M_aug_template.copy()
        M_aug[self.control_dim:, self.control_dim:] = M
        
        mu = np.matmul(self.expm_A_s_dt, self.z_tilde)
        PHI_inv_mul_mu = self.PHI_inv_diag * mu
        
        self.sig_hat = -np.matmul(M_aug, PHI_inv_mul_mu)
        
        # using fixed weight
        # weight = np.array([0, 0, 0.4, 0.09])
        # self.sig_hat = -1 * weight * self.z_tilde.copy()
    
    def update_u_ad(self):
        # low pass filter for sigma_hat
        
        sig_hat_original = self.sig_hat.copy()
        sig_hat_filtered = self.low_pass_filter(self.filter_time_constants, sig_hat_original, self.sig_hat_prev)
        
        self.u_ad = -sig_hat_filtered[self.control_dim:]
        
        self.sig_hat_prev = sig_hat_filtered.copy()
        
    def low_pass_filter(self, time_const, curr_i, prev_i):
        '''
        description: 一阶低通滤波器 
        time_const:  t_c = 1  dt = 0.005 alpha = 0.005/(0.005+1) = 0.005
                     t_c = 0.001 dt = 0.005 alpha = 0.005/(0.005+0.001) = 0.833
                     t_c = 0.005 dt = 0.005 alpha = 0.005/(0.005+0.005) = 0.5
        return {*}
        '''        
        
        alpha       = self.dt / (self.dt + time_const)
        y_filter    = (1 - alpha) * prev_i + alpha * curr_i
        
        return y_filter
        
    
class TwoDOFArmMPCController:
    def __init__(self, urdf_path, mpc_dt=0.01, simulation_dt=None, horizon_length=50, enable_visualization=False, target_change_interval=2.0, control_mode='mpc', velocity_limits=None, enable_velocity_constraints=True, using_l1_adaptive_controller=True, l1_start_time=None):
        """
        Initialize 2DOF robotic arm controller (MPC or PID)
        
        Args:
            urdf_path: URDF file path
            mpc_dt: MPC control time step (for optimization and control)
            simulation_dt: Simulation time step (for dynamics integration, defaults to mpc_dt if None)
            horizon_length: Prediction horizon length 
            enable_visualization: Enable Gepetto visualization
            target_change_interval: Time interval (seconds) for changing target
            control_mode: Control mode ('mpc' or 'pid')
            velocity_limits: Maximum joint velocities [joint1, joint2] (rad/s) for PID mode
            enable_velocity_constraints: Enable velocity constraints for PID mode
            friction_params: Friction parameters dict with keys: 'static_friction', 'dynamic_friction', 'velocity_threshold'
        """
        # Set time steps
        self.mpc_dt = 0.01  # Time step for MPC control
        self.simulation_dt = 0.001  # simulation using 1ms time step
        self.control_rate = 100.0  # Hz
        self.control_interval = 1.0 / self.simulation_dt / self.control_rate
        
        self.horizon_length = horizon_length
        self.enable_visualization = enable_visualization
        self.control_mode = control_mode.lower()
        
        # set numpy print options
        np.set_printoptions(precision=4, suppress=True)
        
        # Initialize friction parameters
        self.friction_params = {
            'static_friction': [0.2, 0.0],    # Static friction coefficients [joint1, joint2] (Nm)
            'dynamic_friction': [0.0, 0.0],   # Dynamic friction coefficients [joint1, joint2] (Nm·s/rad)  
            'velocity_threshold': [0.0, 0.0], # Velocity threshold for static/dynamic transition (rad/s)
            'enable_friction': True,             # Enable friction model in dynamics
            'enable_friction_compensation': False # Enable friction compensation in control
        }
        
        
        self.enable_friction = self.friction_params.get('enable_friction', False)
        self.enable_friction_compensation = self.friction_params.get('enable_friction_compensation', False)
        
        # Load robot model
        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()
        
        # State and control dimensions
        self.state_dim = self.robot.nq + self.robot.nv  # position + velocity
        self.control_dim = self.robot.nv  # joint torques
        
        # L1 adaptive controller configuration
        self.using_l1_adaptive_controller = using_l1_adaptive_controller
        if self.using_l1_adaptive_controller:
            self.l1_config = {
                'adaptation_gain': -1.0,
                'filter_time_constants': 0.0,
                'l1_start_time': 2.0
            }
            
            self.l1_start_step = int(self.l1_config['l1_start_time'] / self.simulation_dt)
            
            self.l1_adaptive_controller = L1AdaptiveController(self.mpc_dt, self.robot, self.l1_config['adaptation_gain'], self.l1_config['filter_time_constants'])
        
        # Target configuration
        self.target_change_interval = target_change_interval  # seconds
        
        # Target mode configuration
        self.use_dynamic_targets = True  # Use dynamic targets or fixed targets
        
        # Fixed target configuration (when use_dynamic_targets = False)
        self.fixed_target_positions = np.array([1.0, 0.6])  # Fixed target positions
        
        # Dynamic target configuration (when use_dynamic_targets = True)
        self.target_sequence = [
            np.array([0.0, 0.0]),   # 0-2s
            np.array([0.0, 0.5]),   # 2-4s
            np.array([0.0, 0.3]),  # 4-6s
            np.array([0.0, 0.0]),   # 6-8s
            np.array([0.0, -0.3]),  # 8-10s
            np.array([0.2, 0.0]),   # 10-12s
            np.array([1.2, 0.6]),   # 12-14s
            np.array([-0.3, 1.0]),  # 14-16s
            np.array([0.6, -0.5]),  # 16-18s
            np.array([0.0, 0.0]),   # 18-20s
        ]
        self.current_target_index = 0
        
        # Current target positions (will be updated based on mode)
        self.target_positions = self.fixed_target_positions.copy()  # Initialize with fixed targets
        
        # Weight parameters (used directly in cost functions)
        self.position_weight = 1.0  # position tracking weight
        self.velocity_weight = 1.0  # velocity tracking weight
        self.control_weight = 1.0  # control weight
        
        # Initialize controller based on mode
        if self.control_mode == 'mpc':
            # Create optimization problem
            self.problem = self.create_optimization_problem()
            # Create solver
            self.solver = crocoddyl.SolverBoxFDDP(self.problem)
            # self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
            
            # Set convergence parameters
            self.solver.th_stop = 1e-6
            
        elif self.control_mode == 'pid':
            # Initialize Cascaded P+PID controller
            # Default gains - can be tuned
            kp_pos = [3.0, 3.0]    # Position proportional gains (outer loop)
            kp_vel = [8.0, 8.0]    # Velocity proportional gains (inner loop)
            ki_vel = [1.0, 1.0]    # Velocity integral gains (inner loop)
            kd_vel = [0.5, 0.5]    # Velocity derivative gains (inner loop)
            
            # Use provided velocity limits or default values
            if velocity_limits is None:
                velocity_limits = [1.5, 1.5]  # Default maximum joint velocities
            
            # Only enable velocity constraints if both enabled and limits are provided
            use_velocity_constraints = enable_velocity_constraints and velocity_limits is not None
            
            self.pid_controller = CascadedPIDController(kp_pos, kp_vel, ki_vel, kd_vel, self.mpc_dt, 
                                                      joint_names=['joint_1', 'joint_2'],
                                                      velocity_limits=velocity_limits if use_velocity_constraints else None)
            
        else:
            raise ValueError(f"Unknown control mode: {control_mode}. Use 'mpc' or 'pid'")
        
        self.create_state_update_model()
        
        # Initialize visualization if enabled
        self.display = None
        if self.enable_visualization:
            self.init_visualization()
        
        # Store trajectory data
        self.time_data = []
        self.position_data = []
        self.velocity_data = []
        self.control_data = []
        self.cost_data = []
        self.target_data = []  # Store target positions over time
        
        self.control_data_u_b = []
        self.control_data_u_ad = []
        
        self.xs = []
        self.us = []
        
        # Debug option for Euler integration comparison
        self._debug_euler_integration = False
        
        self.print_config()
        
    def print_config(self):
        if self.control_mode == 'mpc':
            print(f"Position weight: {self.position_weight}, Velocity weight: {self.velocity_weight}, Control weight: {self.control_weight}")
        
        print(f"Target mode: {'Dynamic' if self.use_dynamic_targets else 'Fixed'}")
        print(f"Initial target positions: {self.target_positions}")
        if self.use_dynamic_targets:
            print(f"Target change interval: {self.target_change_interval} seconds")
            print(f"Number of target sequences: {len(self.target_sequence)}")
        else:
            print(f"Fixed target positions: {self.fixed_target_positions}")
        print(f"Visualization enabled: {self.enable_visualization}")
        print(f"Using unified MPC dynamics model for all control modes")
        print(f"Friction model enabled: {self.enable_friction}")
        print(f"Friction compensation enabled: {self.enable_friction_compensation}")
        if self.enable_friction:
            print(f"  Static friction: {self.friction_params['static_friction']} Nm")
            print(f"  Dynamic friction: {self.friction_params['dynamic_friction']} Nm·s/rad")
            print(f"  Velocity threshold: {self.friction_params['velocity_threshold']} rad/s")
            
        # continue once input is received
        # input("Press Enter to continue...")
        
    def init_visualization(self):
        """Initialize Gepetto visualization"""
        if not CROCODDYL_AVAILABLE:
            print("Warning: Cannot initialize visualization: crocoddyl not available")
            self.enable_visualization = False
            self.display = None
            return
        
        try:
            # Check if gepetto-gui is running
            gepetto.corbaserver.Client()
            
            # Load robot for visualization
            robot_name = 's500_uam_simple'  # Use the robot name that matches your URDF
            robot = example_robot_data.load(robot_name)
            
            # Visualization parameters
            rate = -1
            freq = 1
            cameraTF = [0.0, -3.0, 1.5, 0.7071, 0.0, 0.0, 0.7071]  # Camera position and orientation
            
            # Create display
            self.display = crocoddyl.GepettoDisplay(
                robot, rate, freq, cameraTF, floor=False)
            
            print("Gepetto visualization initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize Gepetto visualization: {e}")
            print("Please make sure gepetto-gui is running: 'gepetto-gui'")
            self.enable_visualization = False
            self.display = None
    
    def create_optimization_problem(self):
        """Create optimization problem"""
        if not CROCODDYL_AVAILABLE:
            raise ValueError("Cannot create optimization problem: crocoddyl not available")
        
        # Create terminal cost model
        terminal_model = self.create_terminal_cost_model()
        
        # Create running cost model
        running_model = self.create_running_cost_model()
        
        # Create problem
        problem = crocoddyl.ShootingProblem(
            np.zeros(self.state_dim),  # initial state
            [running_model] * self.horizon_length,
            terminal_model
        )
        
        return problem
    
    def create_state_update_model(self):
        """Create state update model for simulation"""
        if not CROCODDYL_AVAILABLE:
            raise ValueError("Cannot create state update model: crocoddyl not available")
        
        # Create differential model for state update
        state = crocoddyl.StateMultibody(self.robot)
        actuation = crocoddyl.ActuationModelFull(state)
        
        # Create differential action model
        differential_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, crocoddyl.CostModelSum(state, self.control_dim)
        )
        
        # Create integrated model using RK4 with MPC time step (for MPC optimization)
        self.state_update_model = crocoddyl.IntegratedActionModelRK4(differential_model, self.mpc_dt)
        
        # Create data for the integrated model
        self.state_update_data = self.state_update_model.createData()
        
        # Create separate state update model for simulation with simulation time step
        self.simulation_update_model = crocoddyl.IntegratedActionModelRK4(differential_model, self.simulation_dt)
        self.simulation_update_data = self.simulation_update_model.createData()
        
        print("State update models created successfully")
        print(f"  MPC model uses dt = {self.mpc_dt} s")
        print(f"  Simulation model uses dt = {self.simulation_dt} s")
    
    def create_running_cost_model(self):
        """Create running cost model"""
        # Create state model
        state = crocoddyl.StateMultibody(self.robot)
        
        # Create action model (Euler integration)
        actuation = crocoddyl.ActuationModelFull(state)
        action_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, crocoddyl.CostModelSum(state, self.control_dim)
        )
        
        # Create weight vectors for position and velocity separately
        position_weights = np.zeros(self.state_dim)
        position_weights[:self.robot.nq] = self.position_weight
        
        velocity_weights = np.zeros(self.state_dim)
        velocity_weights[self.robot.nq:] = self.velocity_weight
        
        # Add position tracking cost
        position_residual = crocoddyl.ResidualModelState(state, self.get_target_state(), self.control_dim)
        position_activation = crocoddyl.ActivationModelWeightedQuad(position_weights)
        position_cost = crocoddyl.CostModelResidual(state, position_activation, position_residual)
        action_model.costs.addCost("position_tracking", position_cost, 1.0)
        
        # Add velocity tracking cost
        velocity_residual = crocoddyl.ResidualModelState(state, self.get_target_state(), self.control_dim)
        velocity_activation = crocoddyl.ActivationModelWeightedQuad(velocity_weights)
        velocity_cost = crocoddyl.CostModelResidual(state, velocity_activation, velocity_residual)
        action_model.costs.addCost("velocity_tracking", velocity_cost, 1.0)
        
        # Add control cost
        control_cost = crocoddyl.CostModelResidual(
            state, 
            crocoddyl.ResidualModelControl(state, self.control_dim)
        )
        action_model.costs.addCost("control", control_cost, self.control_weight)
        
        # Create integrated model using MPC time step
        integrated_model = crocoddyl.IntegratedActionModelEuler(action_model, self.mpc_dt)
        
        return integrated_model
    
    def create_terminal_cost_model(self):
        """Create terminal cost model"""
        # Create state model
        state = crocoddyl.StateMultibody(self.robot)
        
        # Create action model
        actuation = crocoddyl.ActuationModelFull(state)
        action_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, crocoddyl.CostModelSum(state, self.control_dim)
        )
        
        # Create weight vectors for position and velocity separately (higher weights for terminal cost)
        position_weights = np.zeros(self.state_dim)
        position_weights[:self.robot.nq] = self.position_weight * 100.0
        
        velocity_weights = np.zeros(self.state_dim)
        velocity_weights[self.robot.nq:] = self.velocity_weight * 100.0
        
        # Add terminal position tracking cost (higher weight)
        position_residual = crocoddyl.ResidualModelState(state, self.get_target_state(), self.control_dim)
        position_activation = crocoddyl.ActivationModelWeightedQuad(position_weights)
        terminal_position_cost = crocoddyl.CostModelResidual(state, position_activation, position_residual)
        action_model.costs.addCost("terminal_position", terminal_position_cost, 1.0)
        
        # Add terminal velocity tracking cost (higher weight)
        velocity_residual = crocoddyl.ResidualModelState(state, self.get_target_state(), self.control_dim)
        velocity_activation = crocoddyl.ActivationModelWeightedQuad(velocity_weights)
        terminal_velocity_cost = crocoddyl.CostModelResidual(state, velocity_activation, velocity_residual)
        action_model.costs.addCost("terminal_velocity", terminal_velocity_cost, 1.0)
        
        # Create integrated model using MPC time step
        integrated_model = crocoddyl.IntegratedActionModelEuler(action_model, self.mpc_dt)
        
        return integrated_model
    
    def get_target_state(self):
        """Get target state (position and velocity)"""
        target_state = np.zeros(self.state_dim)
        target_state[:self.robot.nq] = self.target_positions  # target position
        target_state[self.robot.nq:] = np.zeros(self.robot.nv)  # target velocity (zero)
        return target_state
    
    def compute_control(self, current_state, max_iterations=100):
        """
        Compute control output based on current mode (MPC or PID)
        
        Args:
            current_state: Current state [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
            max_iterations: Maximum iterations (for MPC)
            
        Returns:
            control_output: Control torques [joint1, joint2]
            additional_info: Additional information (varies by mode)
        """
        if self.control_mode == 'mpc':
            return self.solve_mpc(current_state, max_iterations)
        elif self.control_mode == 'pid':
            return self.solve_pid(current_state)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
    
    def solve_mpc(self, current_state, max_iterations=100):
        """
        Solve MPC problem
        
        Args:
            current_state: Current state
            max_iterations: Maximum iterations
            
        Returns:
            optimal_control: Optimal control sequence
            optimal_states: Optimal state sequence
            solve_time: Time taken to solve
            iterations: Number of iterations
        """
        import time
        
        # Update problem initial state
        self.problem.x0 = current_state
        
        # Record solve time
        start_time = time.time()
        
        # Solve optimization problem
        self.solver.solve(self.xs, self.us, max_iterations, False, 1e-4)
        
        self.xs = self.solver.xs
        self.us = self.solver.us
        
        # Calculate solve time
        solve_time = time.time() - start_time
        
        # Get optimal solution
        optimal_states = self.solver.xs
        optimal_controls = self.solver.us
        
        # Get number of iterations
        iterations = self.solver.iter
        
        return optimal_controls, optimal_states, solve_time, iterations
    
    def solve_pid(self, current_state):
        """
        Solve PID control problem
        
        Args:
            current_state: Current state [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
            
        Returns:
            control_output: Control torques [joint1, joint2]
            debug_info: Debug information
            solve_time: Time taken to compute (negligible for PID)
            iterations: Always 1 for PID
        """
        import time
        
        # Extract current positions and velocities
        current_positions = current_state[:2]  # [joint1_pos, joint2_pos]
        current_velocities = current_state[2:]  # [joint1_vel, joint2_vel]
        
        # Record compute time
        start_time = time.time()
        
        # Compute PID control
        control_output = self.pid_controller.compute_control(
            current_positions, 
            self.target_positions, 
            current_velocities
        )
        
        # Calculate compute time
        solve_time = time.time() - start_time
        
        # Get debug information
        debug_info = self.pid_controller.get_debug_info()
        
        # For PID, we return a single control vector and empty states
        # to maintain compatibility with MPC interface
        return [control_output], [], solve_time, 1
    
    def display_robot_state(self, state):
        """Display robot state in Gepetto visualization"""
        if self.display is not None and self.enable_visualization:
            try:
                # For a 2-joint robot with fixed base (nq=2, nv=2)
                # State is [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
                q = np.array(state[:2], dtype=np.float64)  # Joint positions as numpy array
                
                # Debug: Print display information
                print(f"Display q shape: {q.shape}, value: {q}")
                
                # Display robot configuration
                self.display.display(q)
                
            except Exception as e:
                print(f"Warning: Could not display robot state: {e}")
                print(f"State: {state}, q: {q if 'q' in locals() else 'undefined'}")

    def display_trajectory(self, states):
        """Display complete trajectory in Gepetto visualization"""
        if self.display is not None and self.enable_visualization:
            try:
                print("Displaying complete trajectory in Gepetto...")
                for i, state in enumerate(states):
                    # For a 2-joint robot with fixed base
                    q = np.array(state[:2], dtype=np.float64)  # Joint positions as numpy array
                    
                    # Display robot configuration
                    self.display.display(q)
                    
                    # Add delay for visualization
                    time.sleep(0.1)
                
                print("Trajectory display completed")
                
            except Exception as e:
                print(f"Warning: Could not display trajectory: {e}")

    def simulate_system(self, initial_state, simulation_time=5.0):
        """
        Simulate system operation
        
        Args:
            initial_state: Initial state
            simulation_time: Simulation time
        """
        current_state = initial_state.copy()
        current_time = 0.0
        current_step = 0
        
        print(f"Starting simulation, total time: {simulation_time} seconds")
        
        while current_time < simulation_time:
            # Record current state
            self.time_data.append(current_time)
            self.position_data.append(current_state[:self.robot.nq].copy())
            self.velocity_data.append(current_state[self.robot.nq:].copy())
            
            # Update target position
            self.update_target_position(current_time)
            
            # Record target position
            self.target_data.append(self.target_positions.copy())
            
            # Display robot state in visualization
            self.display_robot_state(current_state)
            
            # Solve control problem (MPC or PID or L1 adaptive controller)
            if current_step % self.control_interval == 0:
                # get baseline control (MPC or PID)
                optimal_controls, optimal_states, solve_time, iterations = self.compute_control(current_state)
                u_b = optimal_controls[0]
                
                # get adaptive control (L1 adaptive controller)
                if self.using_l1_adaptive_controller and current_step >= self.l1_start_step:
                    self.l1_adaptive_controller.solve_l1_problem(current_state, u_b)
                    
                    print('--------------------------------step: {}--------------------------------'.format(current_step))
                    print('current_state: ', current_state)
                    print('u_b          : ', u_b)
                    print('z_hat        : ', self.l1_adaptive_controller.z_hat)
                    print('z_tilde      : ', self.l1_adaptive_controller.z_tilde)
                    print('sig_hat      : ', self.l1_adaptive_controller.sig_hat)
                    print('u_ad         : ', self.l1_adaptive_controller.u_ad)
                    
                    u_ad = self.l1_adaptive_controller.u_ad.copy()
                else:
                    u_ad = np.zeros(2)
            
            optimal_controls[0] = u_b + u_ad
            # Record control input
            self.control_data.append(optimal_controls[0])
            self.control_data_u_b.append(u_b)
            self.control_data_u_ad.append(u_ad)
            
            # Record cost and solver info (handle both MPC and PID modes)
            if self.control_mode == 'mpc':
                self.cost_data.append(self.solver.cost)
                # Print solver performance
                # print(f"MPC Solve Time: {solve_time*1000:.1f}ms, Iterations: {iterations}, Cost: {self.solver.cost:.6f}")
            else:  # PID mode
                # For PID, we don't have a cost from solver, so we'll use a placeholder or calculate tracking error
                tracking_error = np.linalg.norm(current_state[:self.robot.nq] - self.target_positions)
                self.cost_data.append(tracking_error)
                # Print PID performance
                print(f"PID Compute Time: {solve_time*1000:.1f}ms, Iterations: {iterations}, Tracking Error: {tracking_error:.6f}")
            
            # Apply first control input
            control_input = optimal_controls[0]  # This should be a 2D vector
            
            
            # Ensure control_input is the right shape and type
            control_input = np.array(control_input, dtype=np.float64)
            
            # Apply friction compensation to control input (if enabled)
            current_velocities = current_state[self.robot.nq:]  # Extract velocities from state
            compensated_control_input = self.apply_friction_compensation_to_control(control_input, current_velocities)
            
            # Compute actual friction torque for debugging
            friction_torque = self.compute_friction_torque(current_velocities)
            friction_compensation = self.compute_static_friction_compensation(current_velocities) if self.enable_friction_compensation else np.zeros(2)
            
            # Debug: Print control information
            # if self.enable_friction:
            #     if self.enable_friction_compensation:
            #         print(f"Control: {control_input} -> Compensation: {friction_compensation} -> Final: {compensated_control_input}, Friction in dynamics: {friction_torque}")
            #     else:
            #         print(f"Control: {control_input} (No compensation), Friction in dynamics: {friction_torque}")
            # else:
            #     print(f"Control: {control_input} (No friction model)")
            
            # Update state using the compensated control input
            next_state = self.calculate_next_state(current_state, compensated_control_input)
            
            print('--------------------------------step: {}--------------------------------'.format(current_step))
            print('current_state            : ', current_state)
            print('compensated_control_input: ', compensated_control_input)
            print('next_state               : ', next_state)
            
            # Update state and time (using simulation time step)
            current_state = next_state
            current_time += self.simulation_dt
            current_step += 1
            
            # Print progress
            if int(current_time * 10) % 10 == 0:
                if self.control_mode == 'mpc':
                    print(f"Time: {current_time:.1f}s, Position: {current_state[:self.robot.nq]}, Cost: {self.solver.cost:.6f}")
                else:  # PID mode
                    tracking_error = np.linalg.norm(current_state[:self.robot.nq] - self.target_positions)
                    print(f"Time: {current_time:.1f}s, Position: {current_state[:self.robot.nq]}, Tracking Error: {tracking_error:.6f}")
            
            # Add small delay for visualization
            if self.enable_visualization:
                time.sleep(0.1)  # 100ms delay for visualization
        
        print("Simulation completed")
    
    def plot_results(self):
        """Plot results"""
        if not self.time_data:
            print("No data to plot")
            return
        
        # Convert to numpy arrays
        time_array = np.array(self.time_data)
        position_array = np.array(self.position_data)
        velocity_array = np.array(self.velocity_data)
        control_array = np.array(self.control_data)
        cost_array = np.array(self.cost_data)
        target_array = np.array(self.target_data)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        control_mode_title = 'MPC' if self.control_mode == 'mpc' else 'PID'
        fig.suptitle(f'2DOF Robotic Arm {control_mode_title} Angle Tracking Results', fontsize=16)
        
        # Position tracking with dynamic targets
        axes[0, 0].plot(time_array, position_array[:, 0], 'b-', label='Joint 1 Actual Position', linewidth=2)
        axes[0, 0].plot(time_array, position_array[:, 1], 'r-', label='Joint 2 Actual Position', linewidth=2)
        axes[0, 0].plot(time_array, target_array[:, 0], 'b--', label='Joint 1 Target Position', linewidth=2, alpha=0.7)
        axes[0, 0].plot(time_array, target_array[:, 1], 'r--', label='Joint 2 Target Position', linewidth=2, alpha=0.7)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (rad)')
        axes[0, 0].set_title('Joint Position Tracking (Dynamic Targets)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Velocity
        axes[0, 1].plot(time_array, velocity_array[:, 0], 'b-', label='Joint 1 Velocity', linewidth=2)
        axes[0, 1].plot(time_array, velocity_array[:, 1], 'r-', label='Joint 2 Velocity', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (rad/s)')
        axes[0, 1].set_title('Joint Velocity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Control input
        axes[1, 0].plot(time_array, control_array[:, 0], 'b-', label='Joint 1 Control', linewidth=2)
        axes[1, 0].plot(time_array, control_array[:, 1], 'r-', label='Joint 2 Control', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Input (Nm)')
        axes[1, 0].set_title('Control Input')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Cost function or tracking error
        axes[1, 1].plot(time_array, cost_array, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        if self.control_mode == 'mpc':
            axes[1, 1].set_ylabel('Cost')
            axes[1, 1].set_title('Optimization Cost')
        else:  # PID mode
            axes[1, 1].set_ylabel('Tracking Error (rad)')
            axes[1, 1].set_title('Position Tracking Error')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        control_mode_str = self.control_mode.upper()
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', f'arm_{control_mode_str.lower()}_results_{timestamp}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
        
        plt.show()
    
    def save_data(self):
        """Save data to CSV file"""
        if not self.time_data:
            print("No data to save")
            return
        
        # Convert to numpy arrays
        time_array = np.array(self.time_data)
        position_array = np.array(self.position_data)
        velocity_array = np.array(self.velocity_data)
        control_array = np.array(self.control_data)
        cost_array = np.array(self.cost_data)
        target_array = np.array(self.target_data)
        
        # Create data directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        control_mode_str = self.control_mode.upper()
        data_path = os.path.join(data_dir, f'arm_{control_mode_str.lower()}_data_{timestamp}.csv')
        
        # Create data matrix
        data_matrix = np.column_stack((
            time_array,
            position_array[:, 0],  # joint_1 position
            position_array[:, 1],  # joint_2 position
            velocity_array[:, 0],  # joint_1 velocity
            velocity_array[:, 1],  # joint_2 velocity
            control_array[:, 0],   # joint_1 control
            control_array[:, 1],   # joint_2 control
            target_array[:, 0],    # joint_1 target
            target_array[:, 1],    # joint_2 target
            cost_array
        ))
        
        # Save as CSV
        cost_label = "cost" if self.control_mode == 'mpc' else "tracking_error"
        header = f"time,joint1_pos,joint2_pos,joint1_vel,joint2_vel,joint1_control,joint2_control,joint1_target,joint2_target,{cost_label}"
        np.savetxt(data_path, data_matrix, delimiter=',', header=header, comments='')
        print(f"Data saved to: {data_path}")
        
    def calculate_next_state(self, current_state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Calculate next state using the robot dynamics with friction effects
        
        Args:
            current_state: Current state [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
            control: Control input [joint1_torque, joint2_torque]
            
        Returns:
            next_state: Next state
        """
        # Extract current velocities for friction calculation
        current_velocities = current_state[self.robot.nq:]
        
        # Compute friction torque
        friction_torque = self.compute_friction_torque(current_velocities)
        
        # Apply friction to control input
        control_with_friction = control.copy() + friction_torque
        
        # Use crocoddyl state update model for consistent dynamics
        # Use simulation model with simulation time step for state integration
        
        # Compute next state using the simulation model with friction-modified control
        self.simulation_update_model.calc(self.simulation_update_data, current_state.copy(), control_with_friction.copy())
        next_state_mpc = self.simulation_update_data.xnext
        
        return next_state_mpc
    
    def enable_debug_euler_comparison(self, enable=True):
        """
        Enable/disable debug comparison between MPC and Euler integration
        
        Args:
            enable: True to enable debug output, False to disable
        """
        self._debug_euler_integration = enable
        status = "enabled" if enable else "disabled"
        print(f"Debug Euler integration comparison {status}")

    def compute_friction_torque(self, current_velocities):
        """
        Compute realistic friction torque including static and dynamic friction
        
        Args:
            current_velocities: Current joint velocities [joint1_vel, joint2_vel]
            
        Returns:
            friction_torque: Friction torques [joint1, joint2] (opposing motion)
        """
        if not self.enable_friction:
            return np.zeros(2)
        
        current_velocities = np.array(current_velocities)
        friction_torque = np.zeros(2)
        
        static_friction = np.array(self.friction_params['static_friction'])
        dynamic_friction = np.array(self.friction_params['dynamic_friction'])
        velocity_threshold = np.array(self.friction_params['velocity_threshold'])
        
        for i in range(2):
            abs_velocity = abs(current_velocities[i])
            
            if abs_velocity < velocity_threshold[i]:
                # Static friction region - Coulomb friction model
                # Static friction opposes any tendency to move
                if abs_velocity > 0:
                    friction_torque[i] = -np.sign(current_velocities[i]) * static_friction[i]
                else:
                    # At exactly zero velocity, friction depends on applied torque
                    # For simulation, we'll apply a small static friction
                    friction_torque[i] = 0.0
            else:
                # Dynamic friction region - Viscous friction model
                # Dynamic friction = viscous friction (proportional to velocity)
                friction_torque[i] = -np.sign(current_velocities[i]) * (
                    static_friction[i] + dynamic_friction[i] * abs_velocity
                )
        
        return friction_torque
    
    def compute_static_friction_compensation(self, current_velocities):
        """
        Legacy method - now calls the improved friction model
        
        Args:
            current_velocities: Current joint velocities [joint1_vel, joint2_vel]
            
        Returns:
            friction_compensation: Static friction compensation torques [joint1, joint2]
        """
        # Return negative of friction torque to compensate for friction
        return -self.compute_friction_torque(current_velocities)

    def apply_friction_compensation_to_control(self, control_output, current_velocities):
        """
        Apply friction compensation to control output (optional)
        
        Args:
            control_output: Original control output [joint1_torque, joint2_torque]
            current_velocities: Current joint velocities [joint1_vel, joint2_vel]
            
        Returns:
            compensated_control: Control output with optional friction compensation
        """
        if not self.enable_friction_compensation:
            # No compensation applied
            return control_output
        
        # Get friction compensation (negative of friction torque to counteract it)
        friction_compensation = self.compute_static_friction_compensation(current_velocities)
        
        # Add friction compensation to control output
        compensated_control = control_output + friction_compensation
        
        return compensated_control

    def update_target_position(self, current_time):
        """
        Update target position based on current time and target mode
        
        Args:
            current_time: Current simulation time
        """
        # Determine new target based on mode
        if self.use_dynamic_targets:
            # Dynamic target mode: cycle through target sequence
            target_index = int(current_time / self.target_change_interval)
            
            # Cycle through target sequence
            if target_index < len(self.target_sequence):
                new_target = self.target_sequence[target_index]
            else:
                # If we exceed the sequence, cycle back to the beginning
                new_target = self.target_sequence[target_index % len(self.target_sequence)]
            
            self.current_target_index = target_index
        else:
            # Fixed target mode: use fixed target positions
            new_target = self.fixed_target_positions
            self.current_target_index = 0
        
        # Only update if target has changed
        if not np.array_equal(self.target_positions, new_target):
            self.target_positions = new_target.copy()
            print(f"Time {current_time:.2f}s: Target changed to {self.target_positions}")
            
            # Recreate optimization problem with new target (only for MPC mode)
            if self.control_mode == 'mpc':
                self.problem = self.create_optimization_problem()
                self.solver = crocoddyl.SolverBoxFDDP(self.problem)
                # self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
                self.solver.setCallbacks([])
            elif self.control_mode == 'pid':
                # For PID mode, just reset the PID controller when target changes
                self.pid_controller.reset()


class RosArmControlNode:
    """
    ROS node for 2DOF robotic arm control (MPC or PID)
    Subscribes to joint states and publishes control commands
    """
    
    def __init__(self, mpc_controller):
        """
        Initialize ROS control node
        
        Args:
            mpc_controller: TwoDOFArmMPCController instance
        """
        self.mpc_controller = mpc_controller
        self.control_mode = mpc_controller.control_mode
        self.current_state = np.zeros(mpc_controller.state_dim)
        self.state_received = False
        self.last_target_update_time = 0.0
        
        # Initialize ROS node
        rospy.init_node(f'arm_{self.control_mode}_ros_node', anonymous=False)
        
        # Get ROS parameters
        self.control_rate = rospy.get_param('~control_rate', 100.0)  # Hz
        self.use_simulation = rospy.get_param('~use_simulation', True)
        
        # Set up subscribers
        if self.use_simulation:
            # Use simulation joint states topic
            # rospy.Subscriber('/arm_controller/joint_states', JointState, self.joint_state_callback)
            rospy.Subscriber('/arm_controller/joint_states', JointState, self.joint_state_callback)
        else:
            # Use real robot joint states topic
            rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        
        # Set up publishers
        self.control_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)
        
        # Add publishers for control input and target state arrays
        self.control_input_pub = rospy.Publisher(f'/{self.control_mode}/control_input', Float64MultiArray, queue_size=10)
        self.target_state_pub = rospy.Publisher(f'/{self.control_mode}/target_state', Float64MultiArray, queue_size=10)
        
        # Add publishers for solver information
        self.solver_info_pub = rospy.Publisher(f'/{self.control_mode}/solver_info', Float64MultiArray, queue_size=10)
        self.predicted_states_pub = rospy.Publisher(f'/{self.control_mode}/predicted_states', Float64MultiArray, queue_size=10)
        self.predicted_controls_pub = rospy.Publisher(f'/{self.control_mode}/predicted_controls', Float64MultiArray, queue_size=10)
        self.cost_pub = rospy.Publisher(f'/{self.control_mode}/cost', Float64, queue_size=10)
        self.solve_time_pub = rospy.Publisher(f'/{self.control_mode}/solve_time', Float64, queue_size=10)
        self.iterations_pub = rospy.Publisher(f'/{self.control_mode}/iterations', Float64, queue_size=10)
        
        # Add individual target publishers for easier monitoring
        self.target_joint1_pub = rospy.Publisher(f'/{self.control_mode}/target_joint1', Float64, queue_size=10)
        self.target_joint2_pub = rospy.Publisher(f'/{self.control_mode}/target_joint2', Float64, queue_size=10)
        
        # Individual joint publishers for simulation
        if self.use_simulation:
            self.joint1_pub = rospy.Publisher('/arm_controller/joint_1_controller/command', Float64, queue_size=10)
            self.joint2_pub = rospy.Publisher('/arm_controller/joint_2_controller/command', Float64, queue_size=10)
        
        # Control rate
        self.rate = rospy.Rate(self.control_rate)
        
        print(f"ROS {self.control_mode.upper()} node initialized")
        print(f"Control mode: {self.control_mode}")
        print(f"Control rate: {self.control_rate} Hz")
        print(f"Simulation mode: {self.use_simulation}")
        print(f"Joint states topic: {'/arm_controller/joint_states' if self.use_simulation else '/joint_states'}")
        print(f"Control topic: /desired_joint_states")
        print(f"Control input topic: /{self.control_mode}/control_input")
        print(f"Target state topic: /{self.control_mode}/target_state")
        print(f"Solver info topic: /{self.control_mode}/solver_info")
        print(f"Predicted states topic: /{self.control_mode}/predicted_states")
        print(f"Predicted controls topic: /{self.control_mode}/predicted_controls")
        print(f"Cost topic: /{self.control_mode}/cost")
        print(f"Solve time topic: /{self.control_mode}/solve_time")
        print(f"Iterations topic: /{self.control_mode}/iterations")
        print(f"Target joint1 topic: /{self.control_mode}/target_joint1")
        print(f"Target joint2 topic: /{self.control_mode}/target_joint2")
        print(f"Joint data extraction: Using joint names (joint_1, joint_2) for robust data handling")
    
    
    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        
        Args:
            msg: JointState message
        """
        try:
            # Extract joint positions and velocities by name
            joint_data = {
                name: {
                    'position': position,
                    'velocity': velocity
                }
                for name, position, velocity in zip(msg.name, msg.position, msg.velocity)
            }
            # Update state: [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
            self.current_state[0] = joint_data['joint_1']['position']  # joint_1 position
            self.current_state[1] = joint_data['joint_2']['position']  # joint_2 position
            self.current_state[2] = joint_data['joint_1']['velocity']  # joint_1 velocity
            self.current_state[3] = joint_data['joint_2']['velocity']  # joint_2 velocity
            self.state_received = True
        except Exception as e:
            rospy.logerr(f"Error in joint state callback: {e}")
    
    def run(self):
        """
        Main ROS node loop
        """
        print(f"Starting ROS {self.control_mode.upper()} control loop...")
        
        while not rospy.is_shutdown():
            if not self.state_received:
                rospy.logwarn_throttle(1.0, "Waiting for joint state data...")
                self.rate.sleep()
                continue
            
            try:
                # Update target position based on current time (only for dynamic targets)
                current_time = rospy.get_time()
                # if self.mpc_controller.use_dynamic_targets:
                self.mpc_controller.update_target_position(current_time)
                
                # Solve control problem (MPC or PID)
                optimal_controls, optimal_states, solve_time, iterations = self.mpc_controller.compute_control(self.current_state)
                
                optimal_states_array = np.array(optimal_states)
                
                if optimal_controls is not None and len(optimal_controls) > 0:
                    # Extract control command (first control in sequence)
                    control_input = np.array(optimal_controls[0], dtype=np.float64)
                    
                    # Apply friction compensation to control input (if enabled)
                    current_velocities = self.current_state[self.mpc_controller.robot.nq:]  # Extract velocities from state
                    compensated_control_input = self.mpc_controller.apply_friction_compensation_to_control(control_input, current_velocities)
                    
                    input_scale = compensated_control_input * 1
                    
                    # add limit to control input
                    # input_scale = np.clip(input_scale, -0.2, 0.2)
                    
                    # Publish JointState control command
                    cmd_msg = JointState()
                    cmd_msg.header.stamp = rospy.Time.now()
                    cmd_msg.name = ['joint_1', 'joint_2']  # Use correct joint names
                    cmd_msg.effort = input_scale.tolist()
                    self.control_pub.publish(cmd_msg)
                    
                    # Debug: Print control info (only every 50th message to avoid spam)
                    if hasattr(self, '_control_msg_count'):
                        self._control_msg_count += 1
                    else:
                        self._control_msg_count = 0
                        
                    if self._control_msg_count % 50 == 0:
                        rospy.loginfo(f"Original control - joint_1: {control_input[0]:.3f}, joint_2: {control_input[1]:.3f}")
                        if self.mpc_controller.enable_friction_compensation:
                            rospy.loginfo(f"Friction compensated control - joint_1: {compensated_control_input[0]:.3f}, joint_2: {compensated_control_input[1]:.3f}")
                        else:
                            rospy.loginfo(f"Control (no friction compensation) - joint_1: {compensated_control_input[0]:.3f}, joint_2: {compensated_control_input[1]:.3f}")
                        rospy.loginfo(f"Target positions: {self.mpc_controller.target_positions}")
                        rospy.loginfo(f"Friction model: {self.mpc_controller.enable_friction}, Compensation: {self.mpc_controller.enable_friction_compensation}")
                    
                    # For simulation, also publish individual joint commands
                    if self.use_simulation:
                        # compensated_control_input[0] corresponds to joint_1, compensated_control_input[1] corresponds to joint_2
                        self.joint1_pub.publish(Float64(compensated_control_input[0]))  # joint_1 control
                        self.joint2_pub.publish(Float64(compensated_control_input[1]))  # joint_2 control
                    
                    # Publish control input array (both original and compensated)
                    control_input_msg = Float64MultiArray()
                    control_input_msg.data = compensated_control_input.tolist()
                    self.control_input_pub.publish(control_input_msg)
                    
                    # Publish target state array (for both MPC and PID)
                    target_state_msg = Float64MultiArray()
                    target_state_msg.data = self.mpc_controller.target_positions.tolist()
                    self.target_state_pub.publish(target_state_msg)
                    
                    # Publish target positions as individual topics for easier monitoring
                    if hasattr(self, 'target_joint1_pub') and hasattr(self, 'target_joint2_pub'):
                        self.target_joint1_pub.publish(Float64(self.mpc_controller.target_positions[0]))
                        self.target_joint2_pub.publish(Float64(self.mpc_controller.target_positions[1]))
                    
                    # Publish solver information
                    self.publish_solver_info(optimal_states, optimal_controls, solve_time, iterations)
                    
                    # Log control information
                    if self.control_mode == 'mpc':
                        rospy.loginfo_throttle(1.0, f"MPC Control: {compensated_control_input}, Target: {self.mpc_controller.target_positions}, Cost: {self.mpc_controller.solver.cost:.6f}, Solve Time: {solve_time*1000:.1f}ms, Iterations: {iterations}")
                    else:  # PID mode
                        cost = np.sum((self.mpc_controller.target_positions - self.current_state[:2])**2)
                        rospy.loginfo_throttle(1.0, f"PID Control: {compensated_control_input}, Target: {self.mpc_controller.target_positions}, Cost: {cost:.6f}, Solve Time: {solve_time*1000:.1f}ms")
                else:
                    rospy.logwarn(f"{self.control_mode.upper()} solver returned no control commands")
                    
            except Exception as e:
                                    rospy.logerr(f"Error in {self.control_mode.upper()} control loop: {e}")
            
            self.rate.sleep()
    
    def publish_solver_info(self, optimal_states, optimal_controls, solve_time, iterations):
        """
        Publish solver information (MPC or PID)
        
        Args:
            optimal_states: Optimal state sequence (empty for PID)
            optimal_controls: Optimal control sequence
            solve_time: Time taken to solve
            iterations: Number of iterations
        """
        try:
            # Get cost value based on control mode
            if self.control_mode == 'mpc':
                cost = self.mpc_controller.solver.cost
            else:  # PID mode
                # Calculate a simple cost based on position error
                current_positions = self.current_state[:2]
                position_errors = self.mpc_controller.target_positions - current_positions
                cost = np.sum(position_errors**2)  # Sum of squared errors
            
            # Publish solver information as array [cost, solve_time, iterations]
            solver_info_msg = Float64MultiArray()
            solver_info_msg.data = [cost, solve_time, float(iterations)]
            self.solver_info_pub.publish(solver_info_msg)
            
            # Publish individual values
            self.cost_pub.publish(Float64(cost))
            self.solve_time_pub.publish(Float64(solve_time))
            self.iterations_pub.publish(Float64(iterations))
            
            # Publish predicted states (flatten the state sequence)
            predicted_states_msg = Float64MultiArray()
            states_array = np.array(optimal_states)
            # Flatten: [horizon, state_dim] -> [horizon * state_dim]
            predicted_states_msg.data = states_array.flatten().tolist()
            self.predicted_states_pub.publish(predicted_states_msg)
            
            # Publish predicted controls (flatten the control sequence)
            predicted_controls_msg = Float64MultiArray()
            controls_array = np.array(optimal_controls)
            # Flatten: [horizon, control_dim] -> [horizon * control_dim]
            predicted_controls_msg.data = controls_array.flatten().tolist()
            self.predicted_controls_pub.publish(predicted_controls_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing MPC info: {e}")


def main():
    """Main function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='2DOF Robotic Arm MPC Control Test')
    parser.add_argument('--visualization', action='store_true', default=False,
                      help='Enable Gepetto visualization')
    parser.add_argument('--simulation-time', type=float, default=5.0,
                      help='Simulation time in seconds')
    parser.add_argument('--mpc-dt', type=float, default=0.01,
                      help='MPC control time step in seconds')
    parser.add_argument('--simulation-dt', type=float, default=0.01,
                      help='Simulation time step in seconds (defaults to mpc-dt if not provided)')
    parser.add_argument('--target-change-interval', type=float, default=1000.0,
                      help='Target change interval in seconds')
    parser.add_argument('--ros', action='store_true', default=False,
                      help='Run as ROS node')
    parser.add_argument('--dynamic-targets', action='store_true', default=False,
                      help='Use dynamic changing targets')
    parser.add_argument('--fixed-targets', action='store_true', default=True,
                      help='Use fixed target positions')
    parser.add_argument('--fixed-joint1', type=float, default=1.0,
                      help='Fixed target position for joint 1 (when using fixed targets)')
    parser.add_argument('--fixed-joint2', type=float, default=0.6,
                      help='Fixed target position for joint 2 (when using fixed targets)')
    parser.add_argument('--control-mode', type=str, default='mpc', choices=['mpc', 'pid'],
                      help='Control mode: mpc or pid')
    parser.add_argument('--kp-pos', type=float, nargs=2, default=[3.0, 3.0],
                      help='Position proportional gains for joint 1 and joint 2 (outer loop)')
    parser.add_argument('--kp-vel', type=float, nargs=2, default=[8.0, 8.0],
                      help='Velocity proportional gains for joint 1 and joint 2 (inner loop)')
    parser.add_argument('--ki-vel', type=float, nargs=2, default=[1.0, 1.0],
                      help='Velocity integral gains for joint 1 and joint 2 (inner loop)')
    parser.add_argument('--kd-vel', type=float, nargs=2, default=[0.5, 0.5],
                      help='Velocity derivative gains for joint 1 and joint 2 (inner loop)')
    # Keep legacy PID parameters for backward compatibility
    parser.add_argument('--pid-kp', type=float, nargs=2, default=None,
                      help='Legacy: Use --kp-pos instead')
    parser.add_argument('--pid-ki', type=float, nargs=2, default=None,
                      help='Legacy: Use --ki-vel instead')
    parser.add_argument('--pid-kd', type=float, nargs=2, default=None,
                      help='Legacy: Use --kd-vel instead')
    parser.add_argument('--velocity-limits', type=float, nargs=2, default=[1.5, 1.5],
                      help='Maximum joint velocities for PID control (rad/s) for joint 1 and joint 2')
    parser.add_argument('--enable-velocity-constraints', action='store_true', default=True,
                      help='Enable velocity constraints for PID control')
    parser.add_argument('--disable-velocity-constraints', action='store_true', default=False,
                      help='Disable velocity constraints for PID control')
    # Friction parameters
    parser.add_argument('--static-friction', type=float, nargs=2, default=[0.15, 0.12],
                      help='Static friction coefficients for joint 1 and joint 2 (Nm)')
    parser.add_argument('--dynamic-friction', type=float, nargs=2, default=[0.08, 0.06],
                      help='Dynamic friction coefficients for joint 1 and joint 2 (Nm·s/rad)')
    parser.add_argument('--friction-velocity-threshold', type=float, nargs=2, default=[0.01, 0.01],
                      help='Velocity threshold for static/dynamic friction transition (rad/s)')
    parser.add_argument('--enable-friction', action='store_true', default=True,
                      help='Enable friction model')
    parser.add_argument('--disable-friction', action='store_true', default=False,
                      help='Disable friction model')
    parser.add_argument('--enable-friction-compensation', action='store_true', default=False,
                      help='Enable friction compensation in control')
    parser.add_argument('--disable-friction-compensation', action='store_true', default=False,
                      help='Disable friction compensation in control')
    
    args = parser.parse_args()
    
    # Check if ROS mode is requested but ROS is not available
    if args.ros and not ROS_AVAILABLE:
        print("Error: ROS mode requested but ROS is not available")
        print("Please install ROS and required packages")
        return
    
    # URDF file path
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'urdf', 's500_uam_arm_effort.urdf')
    
    # Check if URDF file exists
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file does not exist: {urdf_path}")
        return
    
    # Create controller (MPC or PID)
    velocity_constraints_enabled = args.enable_velocity_constraints and not args.disable_velocity_constraints
    friction_enabled = args.enable_friction and not args.disable_friction
    friction_compensation_enabled = args.enable_friction_compensation and not args.disable_friction_compensation
    
    mpc_controller = TwoDOFArmMPCController(
        urdf_path=urdf_path,
        mpc_dt=args.mpc_dt,
        simulation_dt=args.simulation_dt,
        enable_visualization=args.visualization,  # enable visualization
        target_change_interval=args.target_change_interval,
        control_mode=args.control_mode,
        velocity_limits=args.velocity_limits if velocity_constraints_enabled else None,
        enable_velocity_constraints=velocity_constraints_enabled
    )
    
    # Set cascaded P+PID gains and velocity constraints if using PID mode
    if args.control_mode == 'pid':
        # Handle legacy parameters for backward compatibility
        if args.pid_kp is not None:
            print("Warning: --pid-kp is deprecated. Use --kp-pos instead.")
            kp_pos = args.pid_kp
        else:
            kp_pos = args.kp_pos
            
        if args.pid_ki is not None:
            print("Warning: --pid-ki is deprecated. Use --ki-vel instead.")
            ki_vel = args.pid_ki
        else:
            ki_vel = args.ki_vel
            
        if args.pid_kd is not None:
            print("Warning: --pid-kd is deprecated. Use --kd-vel instead.")
            kd_vel = args.pid_kd
        else:
            kd_vel = args.kd_vel
        
        # Set cascaded controller gains
        mpc_controller.pid_controller.kp_pos = np.array(kp_pos)
        mpc_controller.pid_controller.kp_vel = np.array(args.kp_vel)
        mpc_controller.pid_controller.ki_vel = np.array(ki_vel)
        mpc_controller.pid_controller.kd_vel = np.array(kd_vel)
        
        # Set velocity constraints
        velocity_constraints_enabled = args.enable_velocity_constraints and not args.disable_velocity_constraints
        mpc_controller.pid_controller.velocity_limits = np.array(args.velocity_limits)
        mpc_controller.pid_controller.enable_velocity_constraints = velocity_constraints_enabled
        
        print(f"Cascaded P+PID gains set:")
        print(f"  Position gains (Kp_pos): {kp_pos}")
        print(f"  Velocity gains (Kp_vel): {args.kp_vel}")
        print(f"  Velocity gains (Ki_vel): {ki_vel}")
        print(f"  Velocity gains (Kd_vel): {kd_vel}")
        print(f"Velocity constraints - Enabled: {velocity_constraints_enabled}, Limits: {args.velocity_limits} rad/s")
    
    # Set target mode and positions based on command line arguments
    if args.fixed_targets:
        # Fixed target mode
        mpc_controller.use_dynamic_targets = False
        mpc_controller.fixed_target_positions = np.array([args.fixed_joint1, args.fixed_joint2])
        mpc_controller.target_positions = mpc_controller.fixed_target_positions.copy()
        print(f"Using fixed targets: {mpc_controller.target_positions}")
    else:
        # Dynamic target mode
        mpc_controller.use_dynamic_targets = True
        print(f"Using dynamic targets with {len(mpc_controller.target_sequence)} sequences")
        print(f"Target change interval: {mpc_controller.target_change_interval} seconds")
    
    if args.ros:
        # Run as ROS node
        print(f"Starting ROS {args.control_mode.upper()} node...")
        ros_node = RosArmControlNode(mpc_controller)
        ros_node.run()
    else:
        # Run simulation mode
        # Set initial state [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        
        print(f"Initial state: {initial_state}")
        print(f"Target positions: {mpc_controller.target_positions}")
        print(f"Visualization enabled: {args.visualization}")
        
        # Run simulation
        simulation_time = args.simulation_time  # seconds
        mpc_controller.simulate_system(initial_state, simulation_time)
        
        # Display complete trajectory if visualization is enabled
        if args.visualization and mpc_controller.enable_visualization:
            print("\nDisplaying complete trajectory...")
            mpc_controller.display_trajectory(mpc_controller.position_data)
        
        # Plot results
        mpc_controller.plot_results()
        
        # Save data
        mpc_controller.save_data()
        
        print("Test completed!")


if __name__ == "__main__":
    main()
