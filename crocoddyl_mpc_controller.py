#!/usr/bin/env python3
"""
Pure Crocoddyl MPC Controller for Trajectory Tracking
Author: Assistant
Date: 2025-10-28
Description: A pure Crocoddyl implementation of MPC controller without eagle_mpc dependency
"""

import numpy as np
import crocoddyl
import pinocchio as pin
import yaml
import os
from scipy.spatial.transform import Rotation as R


class CrocoddylMPCController:
    """
    Pure Crocoddyl MPC Controller for trajectory tracking
    """
    
    def __init__(self, robot_model, platform_params, mpc_config, dt_mpc=0.02):
        """
        Initialize Crocoddyl MPC Controller
        
        Args:
            robot_model: Pinocchio robot model
            platform_params: Platform parameters (thrust coefficients, etc.)
            mpc_config: MPC configuration dictionary
            dt_mpc: MPC time step (seconds)
        """
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.platform_params = platform_params
        self.dt_mpc = dt_mpc
        
        # MPC parameters
        self.horizon = int(mpc_config.get('horizon', 20))
        self.max_iterations = int(mpc_config.get('max_iterations', 100))
        self.convergence_tolerance = float(mpc_config.get('convergence_tolerance', 1e-6))
        
        # State and control dimensions
        self.state_dim = robot_model.nq + robot_model.nv
        
        # Initialize state first to get the correct tangent space dimension
        self.state = crocoddyl.StateMultibody(self.robot_model)
        
        # Create actuation model first to get correct control dimension
        self.actuation = self._create_actuation_model()
        
        # Use actuation model's control dimension (nu) instead of robot.nv
        self.control_dim = self.actuation.nu
        
        # Get the correct tangent space dimension (ndx) for weights
        # For quaternion states: nq=9, nv=8, but ndx=16 (quaternion tangent space is 3D, not 4D)
        tangent_dim = self.state.ndx
        
        # Cost weights (adjust dimensions based on actual dimensions)
        default_state_weights = [100, 100, 100, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 16 elements for ndx=16
        default_control_weights = [1, 1, 1, 1, 1, 1]  # 6 controls for actuation.nu=6
        default_terminal_weights = [1000, 1000, 1000, 100, 100, 100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # 16 elements for ndx=16
        
        # Ensure weights match actual dimensions
        state_weights = mpc_config.get('state_weights', default_state_weights)
        control_weights = mpc_config.get('control_weights', default_control_weights)
        terminal_weights = mpc_config.get('terminal_weights', default_terminal_weights)
        
        # Adjust weights to match actual dimensions
        self.state_weights = np.array(state_weights[:tangent_dim] + [1.0] * max(0, tangent_dim - len(state_weights)))
        self.control_weights = np.array(control_weights[:self.control_dim] + [1.0] * max(0, self.control_dim - len(control_weights)))
        self.terminal_weights = np.array(terminal_weights[:tangent_dim] + [1.0] * max(0, tangent_dim - len(terminal_weights)))
        
        # Initialize trajectory reference
        self.reference_trajectory = []
        self.current_reference_index = 0
        
        # Initialize MPC problem
        self.problem = None
        self.solver = None
        
        print(f"Crocoddyl MPC Controller initialized:")
        print(f"  State dimension (nq+nv): {self.state_dim}")
        print(f"  Tangent space dimension (ndx): {tangent_dim}")
        print(f"  Control dimension (robot.nv): {self.control_dim}")
        print(f"  Actuation control dimension (nu): {self.actuation.nu}")
        print(f"  Horizon: {self.horizon}")
        print(f"  MPC time step: {self.dt_mpc}")
        print(f"  State weights shape: {self.state_weights.shape}")
        print(f"  Control weights shape: {self.control_weights.shape}")
        print(f"  Terminal weights shape: {self.terminal_weights.shape}")
    
    def _create_actuation_model(self):
        """Create actuation model based on platform parameters"""
        # Create multirotor actuation model
        n_rotors = self.platform_params.n_rotors
        tau_f = self.platform_params.tau_f
        
        # Create actuation model
        try:
            # Try to use the new FloatingBaseThrusters model
            # This requires creating individual thruster objects
            thrusters = []
            for i in range(n_rotors):
                # Create thruster with torque coefficient only
                # According to C++ signature: Thruster(double ctorque)
                torque_coeff = float(tau_f[5, i]) if tau_f.shape[0] > 5 else 0.1
                thruster = crocoddyl.Thruster(torque_coeff)
                thrusters.append(thruster)
            
            actuation = crocoddyl.ActuationModelFloatingBaseThrusters(
                self.state, thrusters
            )
            print("Using ActuationModelFloatingBaseThrusters")
        except (AttributeError, TypeError) as e:
            print(f"FloatingBaseThrusters failed: {e}")
            try:
                # Fallback to deprecated MultiCopterBase without n_rotors
                actuation = crocoddyl.ActuationModelMultiCopterBase(
                    self.state, tau_f
                )
                print("Using deprecated ActuationModelMultiCopterBase")
            except (AttributeError, TypeError) as e:
                print(f"MultiCopterBase failed: {e}")
                # Final fallback to full actuation model
                print("Warning: Multicopter actuation models not available, using full actuation")
                actuation = crocoddyl.ActuationModelFull(self.state)
        
        return actuation
    
    def set_reference_trajectory(self, reference_states, reference_controls=None):
        """
        Set reference trajectory for MPC
        
        Args:
            reference_states: List of reference states
            reference_controls: List of reference controls (optional)
        """
        self.reference_trajectory = reference_states
        self.reference_controls = reference_controls if reference_controls is not None else []
        print(f"Reference trajectory set with {len(reference_states)} states")
    
    def _create_running_model(self, ref_state, ref_control=None):
        """Create running model for a single time step"""
        # Create cost model first with correct control dimension
        cost_model = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        
        # State regulation cost
        state_residual = crocoddyl.ResidualModelState(self.state, ref_state, self.actuation.nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(self.state_weights)
        state_cost = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        cost_model.addCost("state_reg", state_cost, 1.0)
        
        # Control regulation cost
        if ref_control is not None:
            # Ensure ref_control has correct dimension
            if len(ref_control) != self.actuation.nu:
                ref_control_resized = np.zeros(self.actuation.nu)
                ref_control_resized[:min(len(ref_control), self.actuation.nu)] = ref_control[:min(len(ref_control), self.actuation.nu)]
                control_residual = crocoddyl.ResidualModelControl(self.state, ref_control_resized)
            else:
                control_residual = crocoddyl.ResidualModelControl(self.state, ref_control)
        else:
            control_residual = crocoddyl.ResidualModelControl(self.state, self.actuation.nu)
        control_activation = crocoddyl.ActivationModelWeightedQuad(self.control_weights)
        control_cost = crocoddyl.CostModelResidual(self.state, control_activation, control_residual)
        cost_model.addCost("control_reg", control_cost, 1.0)
        
        # Create differential action model with all required arguments
        diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_model
        )
        
        # Create integrated action model
        int_model = crocoddyl.IntegratedActionModelEuler(diff_model, self.dt_mpc)
        
        return int_model
    
    def _create_terminal_model(self, ref_state):
        """Create terminal model"""
        # Create cost model first with correct control dimension
        cost_model = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        
        # Terminal state cost
        state_residual = crocoddyl.ResidualModelState(self.state, ref_state, self.actuation.nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(self.terminal_weights)
        state_cost = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        cost_model.addCost("terminal_state", state_cost, 1.0)
        
        # Create differential action model with all required arguments
        diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_model
        )
        
        # Create integrated action model
        int_model = crocoddyl.IntegratedActionModelEuler(diff_model, self.dt_mpc)
        
        return int_model
    
    def update_problem(self, current_state, reference_index=0):
        """
        Update MPC problem with current state and reference
        
        Args:
            current_state: Current system state
            reference_index: Index in reference trajectory
        """
        self.current_reference_index = reference_index
        
        # Create running models
        running_models = []
        for i in range(self.horizon):
            ref_idx = min(reference_index + i, len(self.reference_trajectory) - 1)
            ref_state = self.reference_trajectory[ref_idx]
            
            # Get reference control if available
            ref_control = None
            if self.reference_controls and ref_idx < len(self.reference_controls):
                ref_control = self.reference_controls[ref_idx]
            
            running_model = self._create_running_model(ref_state, ref_control)
            running_models.append(running_model)
        
        # Create terminal model
        terminal_ref_idx = min(reference_index + self.horizon, len(self.reference_trajectory) - 1)
        terminal_ref_state = self.reference_trajectory[terminal_ref_idx]
        terminal_model = self._create_terminal_model(terminal_ref_state)
        
        # Create shooting problem
        self.problem = crocoddyl.ShootingProblem(current_state, running_models, terminal_model)
        
        # Create solver
        self.solver = crocoddyl.SolverFDDP(self.problem)
        self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
        
        # Set solver parameters
        self.solver.th_stop = self.convergence_tolerance
        
        # Initialize with warm start if available
        if hasattr(self, 'xs_warm') and hasattr(self, 'us_warm'):
            if len(self.xs_warm) == len(running_models) + 1 and len(self.us_warm) == len(running_models):
                self.solver.xs = self.xs_warm
                self.solver.us = self.us_warm
    
    def solve(self, current_state, reference_index=0, warm_start=True):
        """
        Solve MPC optimization problem
        
        Args:
            current_state: Current system state
            reference_index: Index in reference trajectory
            warm_start: Whether to use warm start
            
        Returns:
            control_input: Optimal control input
            solve_info: Solver information dictionary
        """
        # Update problem
        self.update_problem(current_state, reference_index)
        
        # Set solver parameters
        self.solver.th_stop = self.convergence_tolerance
        
        # Solve (Crocoddyl solve method doesn't take maxiter as parameter)
        solved = self.solver.solve()
        
        # Extract solution
        if solved or self.solver.iter > 0:
            control_input = self.solver.us[0].copy()
            
            # Store warm start for next iteration
            if warm_start and len(self.solver.xs) > 1:
                # Shift solution for warm start
                # Convert to list first, then concatenate
                xs_list = list(self.solver.xs)
                us_list = list(self.solver.us)
                self.xs_warm = xs_list[1:] + [xs_list[-1]]
                self.us_warm = us_list[1:] + [us_list[-1]]
        else:
            # Fallback: use zero control or previous control
            control_input = np.zeros(self.control_dim)
            if hasattr(self, 'last_control'):
                control_input = self.last_control.copy()
        
        # Store last control
        self.last_control = control_input.copy()
        
        # Prepare solve info
        solve_info = {
            'solved': solved,
            'iterations': self.solver.iter,
            'cost': self.solver.cost,
            'convergence': self.solver.th_stop,
            'solve_time': 0.0  # Would need timing implementation
        }
        
        return control_input, solve_info
    
    def get_predicted_trajectory(self):
        """
        Get predicted state and control trajectories
        
        Returns:
            xs: Predicted state trajectory
            us: Predicted control trajectory
        """
        if self.solver is not None:
            return self.solver.xs.copy(), self.solver.us.copy()
        else:
            return [], []


class PlatformParams:
    """Platform parameters class"""
    
    def __init__(self, config_dict):
        """Initialize from configuration dictionary"""
        self.n_rotors = config_dict.get('n_rotors', 4)
        self.cf = config_dict.get('cf', 1.0)
        self.cm = config_dict.get('cm', 0.1)
        self.max_thrust = config_dict.get('max_thrust', 10.0)
        self.min_thrust = config_dict.get('min_thrust', 0.0)
        
        # Create tau_f matrix (thrust to force/torque mapping)
        self.tau_f = self._create_tau_f_matrix(config_dict)
    
    def _create_tau_f_matrix(self, config_dict):
        """Create tau_f matrix from configuration"""
        n_rotors = self.n_rotors
        cf = self.cf
        cm = self.cm
        
        # Initialize tau_f matrix (6 x n_rotors)
        tau_f = np.zeros((6, n_rotors))
        
        # Get rotor positions and directions from config
        if 'rotors' in config_dict:
            rotors = config_dict['rotors']
            for i, rotor in enumerate(rotors):
                if i >= n_rotors:
                    break
                
                # Position
                pos = rotor.get('position', [0, 0, 0])
                # Direction (thrust direction)
                direction = rotor.get('direction', 1)  # 1 or -1
                
                # Force mapping (thrust in z-direction)
                tau_f[2, i] = cf * direction  # Fz
                
                # Torque mapping
                tau_f[3, i] = cf * pos[1] * direction  # Mx = Fz * y
                tau_f[4, i] = -cf * pos[0] * direction  # My = -Fz * x
                tau_f[5, i] = cm * direction  # Mz
        else:
            # Default quadrotor configuration
            arm_length = config_dict.get('arm_length', 0.25)
            
            # Standard quadrotor layout
            positions = [
                [arm_length, 0, 0],      # Front
                [0, arm_length, 0],      # Right  
                [-arm_length, 0, 0],     # Back
                [0, -arm_length, 0]      # Left
            ]
            directions = [1, -1, 1, -1]  # Alternating spin directions
            
            for i in range(min(n_rotors, 4)):
                pos = positions[i]
                direction = directions[i]
                
                # Force mapping
                tau_f[2, i] = cf  # Fz (all rotors contribute to lift)
                
                # Torque mapping
                tau_f[3, i] = cf * pos[1]  # Mx = Fz * y
                tau_f[4, i] = -cf * pos[0]  # My = -Fz * x
                tau_f[5, i] = cm * direction  # Mz (yaw torque)
        
        return tau_f


def load_mpc_config(config_path):
    """Load MPC configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('mpc_controller', {})


def load_platform_config(config_path):
    """Load platform configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('platform', {})
