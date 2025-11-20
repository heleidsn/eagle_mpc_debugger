#!/usr/bin/env python3
"""
S500 Quadrotor Trajectory Planning Script
Using Crocoddyl and Pinocchio for trajectory optimization

Features:
- Load S500 quadrotor geometry information from YAML file
- Load Pinocchio model from URDF file
- Perform trajectory optimization using Crocoddyl

Author: Assistant
Date: 2025-10-07
"""

import eagle_mpc
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import time
import pinocchio as pin
import crocoddyl
import rospkg
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class S500TrajectoryPlanner:
    """S500 Quadrotor Trajectory Planner"""
    
    def __init__(self, s500_yaml_path: str = None, urdf_path: str = None):
        """
        Initialize S500 trajectory planner
        
        Args:
            s500_yaml_path: Path to S500 configuration YAML file
            urdf_path: Path to URDF model file
        """
        # Get ROS package path
        try:
            rospack = rospkg.RosPack()
            self.package_path = rospack.get_path('eagle_mpc_debugger')
        except:
            self.package_path = os.path.dirname(os.path.abspath(__file__))
            print("Warning: ROS package not found, using current directory")
        
        # Set default paths
        if s500_yaml_path is None:
            s500_yaml_path = os.path.join(self.package_path, 'config/yaml/multicopter/s500.yaml')
        if urdf_path is None:
            urdf_path = os.path.join(self.package_path, 'models/urdf/s500_simple.urdf')
            
        self.s500_yaml_path = s500_yaml_path
        self.urdf_path = urdf_path
        
        # Initialize attributes
        self.s500_config = None
        self.robot_model = None
        self.robot_data = None
        self.state = None
        self.actuation = None
        self.problem = None
        self.solver = None
        self.dt = None  # Store actual time step used in optimization
        
        # Load configuration and models
        self.load_s500_config()
        self.load_pinocchio_model()
        self.create_actuation_model()
        
    def load_s500_config(self):
        """Load S500 configuration from YAML file"""
        try:
            with open(self.s500_yaml_path, 'r') as f:
                self.s500_config = yaml.safe_load(f)
            print(f"✓ Successfully loaded S500 configuration file: {self.s500_yaml_path}")
            
            # Print key configuration information
            platform = self.s500_config['platform']
            print(f"  - Number of rotors: {platform['n_rotors']}")
            print(f"  - Thrust coefficient cf: {platform['cf']}")
            print(f"  - Moment coefficient cm: {platform['cm']}")
            print(f"  - Maximum thrust: {platform['max_thrust']} N")
            print(f"  - Minimum thrust: {platform['min_thrust']} N")
            
        except Exception as e:
            print(f"✗ Failed to load S500 configuration file: {e}")
            raise
            
    def load_pinocchio_model(self):
        """Load Pinocchio model from URDF file"""
        try:
            print(f"Loading URDF file: {self.urdf_path}")
            
            # Build Pinocchio model (using free flyer joint model)
            self.robot_model = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
            self.robot_data = self.robot_model.createData()
            
            # Create state model
            self.state = crocoddyl.StateMultibody(self.robot_model)
            
            print(f"✓ Successfully loaded Pinocchio model")
            print(f"  - Number of joints: {self.robot_model.nq}")
            print(f"  - Velocity dimension: {self.robot_model.nv}")
            print(f"  - State dimension: {self.state.ndx}")
            print(f"  - Base link mass: {self.robot_model.inertias[1].mass:.3f} kg")
            
            # Print inertia information
            base_inertia = self.robot_model.inertias[1]
            print(f"  - Inertia matrix diagonal: [{base_inertia.inertia[0,0]:.6f}, {base_inertia.inertia[1,1]:.6f}, {base_inertia.inertia[2,2]:.6f}]")
            
        except Exception as e:
            print(f"✗ Failed to load Pinocchio model: {e}")
            raise
            
    def create_actuation_model(self):
        """Create actuation model"""
        try:
            platform = self.s500_config['platform']
            n_rotors = platform['n_rotors']
            cf = platform['cf']
            cm = platform['cm']
            rotors = platform['$rotors']
            
            # Build tau_f matrix (thrust to force/moment mapping matrix)
            # tau_f shape: (6, n_rotors) - mapping from n_rotors thrust inputs to 6 force/moment outputs
            tau_f = np.zeros((6, n_rotors))
            
            print("Rotor configuration:")
            for i, rotor in enumerate(rotors):
                pos = np.array(rotor['translation'])
                spin_dir = rotor['spin_direction'][0]
                
                print(f"  Rotor{i+1}: position={pos}, spin_direction={'CCW' if spin_dir < 0 else 'CW'}")
                
                # Force mapping (all rotors generate upward thrust)
                tau_f[0, i] = 0.0      # x-direction force
                tau_f[1, i] = 0.0      # y-direction force  
                tau_f[2, i] = 1.0      # z-direction thrust (upward positive)
                
                # Moment mapping (using rotor position to calculate moment arm)
                tau_f[3, i] = pos[1]   # moment about x-axis (roll) = y * Fz
                tau_f[4, i] = -pos[0]  # moment about y-axis (pitch) = -x * Fz
                tau_f[5, i] = spin_dir * cm / cf  # moment about z-axis (yaw) = reaction moment coefficient ratio
            
            # Create multicopter actuation model
            self.actuation = crocoddyl.ActuationModelMultiCopterBase(self.state, tau_f)
            
            print(f"✓ Successfully created actuation model")
            print(f"  - Control input dimension: {self.actuation.nu}")
            print(f"  - tau_f matrix shape: {tau_f.shape}")
            
        except Exception as e:
            print(f"✗ Failed to create actuation model: {e}")
            raise
            
    def create_cost_model(self, target_state: np.ndarray = None, 
                         control_weight: float = 1e-5,
                         state_weight: float = 1,
                         is_terminal: bool = False,
                         is_waypoint: bool = False,
                         waypoint_multiplier: float = 10000.0) -> crocoddyl.CostModelSum:
        """
        Create cost model
        
        Args:
            target_state: Target state
            control_weight: Control input weight
            state_weight: State weight
            is_terminal: Whether this is a terminal cost
            is_waypoint: Whether this is a waypoint stage
            waypoint_multiplier: Waypoint weight multiplier
            
        Returns:
            Cost model
        """
        # Create cost model
        # Keep control dimension consistent to avoid dimension mismatch issues
        control_dim = self.actuation.nu
        cost_model = crocoddyl.CostModelSum(self.state, control_dim)
        
        # Default target state (hovering 1m above origin)
        if target_state is None:
            target_state = np.zeros(self.state.ndx)
            target_state[2] = 1.0  # z = 1m
            target_state[6] = 1.0  # quaternion w component
        
        # Apply waypoint weight enhancement
        effective_state_weight = float(state_weight)
        effective_control_weight = float(control_weight)
        if is_waypoint:
            original_state_weight = effective_state_weight
            original_control_weight = effective_control_weight
            effective_state_weight = float(effective_state_weight * waypoint_multiplier)
            effective_control_weight = float(effective_control_weight * waypoint_multiplier)
            print(f"🎯 Waypoint权重增强:")
            print(f"   State权重: {original_state_weight} -> {effective_state_weight} (倍数: {waypoint_multiplier})")
            print(f"   Control权重: {original_control_weight} -> {effective_control_weight} (倍数: {waypoint_multiplier})")
        
        # State cost
        state_activation = crocoddyl.ActivationModelQuad(self.state.ndx)
        state_residual = crocoddyl.ResidualModelState(self.state, target_state, control_dim)
        cost_model.addCost("state_reg", 
                          crocoddyl.CostModelResidual(self.state, state_activation, state_residual), 
                          effective_state_weight)
        
        # Control input cost (only for non-terminal models)
        if not is_terminal:
            # Hover thrust (gravity compensation)
            mass = self.robot_model.inertias[1].mass
            hover_thrust = mass * 9.81 / self.actuation.nu  # Evenly distributed among rotors
            control_ref = np.full(self.actuation.nu, hover_thrust)
            
            control_activation = crocoddyl.ActivationModelQuad(self.actuation.nu)
            control_residual = crocoddyl.ResidualModelControl(self.state, control_ref)
            cost_model.addCost("control_reg", 
                              crocoddyl.CostModelResidual(self.state, control_activation, control_residual), 
                              effective_control_weight)
        
        return cost_model
        
    def create_trajectory_problem(self, 
                                waypoints: List[np.ndarray],
                                durations: List[float],
                                dt: float = 0.02,
                                waypoint_multiplier: float = 1000.0,
                                use_thrust_constraints: bool = True) -> None:
        """
        Create trajectory optimization problem
        
        Args:
            waypoints: List of waypoints, each point is [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            durations: Duration of each trajectory segment (seconds)
            dt: Time step (seconds)
            waypoint_multiplier: Waypoint weight multiplier for enhanced precision
            use_thrust_constraints: Whether to add thrust constraints to the problem
        """
        try:
            if len(waypoints) != len(durations) + 1:
                raise ValueError("Number of waypoints should be one more than number of durations")
            
            # Store the actual time step used
            self.dt = dt
            
            # Initial state
            initial_state = waypoints[0]
            
            # Store waypoint times for plotting
            self._waypoint_times = [0.0]  # Start with initial waypoint at t=0
            self._waypoint_positions = [waypoints[0][:3]]  # Store positions for 3D plotting
            
            running_models = []
            current_time = 0.0
            
            for i, duration in enumerate(durations):
                # Target state for current segment
                target_state = waypoints[i + 1]
                current_time += duration
                
                # Record waypoint time and position
                self._waypoint_times.append(current_time)
                self._waypoint_positions.append(target_state[:3])
                
                # Calculate number of time steps
                n_steps = max(1, int(duration / dt))
                
                print(f"Creating trajectory segment {i+1}: duration={duration:.2f}s, steps={n_steps}")
                
                # Create cost models for this segment
                # First step (waypoint) gets enhanced weight
                waypoint_cost_model = self.create_cost_model(
                    target_state=target_state,
                    is_waypoint=True,
                    waypoint_multiplier=waypoint_multiplier
                )
                
                # Intermediate steps get normal weight
                normal_cost_model = self.create_cost_model(target_state=target_state)
                
                # Create differential action models
                waypoint_diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self.state, self.actuation, waypoint_cost_model
                )
                normal_diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self.state, self.actuation, normal_cost_model
                )
                
                # Create integrated action models
                waypoint_int_model = crocoddyl.IntegratedActionModelEuler(waypoint_diff_model, dt)
                normal_int_model = crocoddyl.IntegratedActionModelEuler(normal_diff_model, dt)
                
                # Add thrust constraints if enabled
                if use_thrust_constraints:
                    platform = self.s500_config['platform']
                    min_thrust = platform['min_thrust']
                    max_thrust = platform['max_thrust']
                    
                    # Set control bounds for both models
                    u_lb = np.full(self.actuation.nu, min_thrust)
                    u_ub = np.full(self.actuation.nu, max_thrust)
                    
                    waypoint_int_model.u_lb = u_lb
                    waypoint_int_model.u_ub = u_ub
                    normal_int_model.u_lb = u_lb
                    normal_int_model.u_ub = u_ub
                    
                    if i == 0:  # Print only once
                        print(f"🔒 Applied thrust constraints: [{min_thrust:.3f}, {max_thrust:.3f}] N per rotor")
                
                # Add to running models list
                # First step uses waypoint model (enhanced weight)
                running_models.append(waypoint_int_model)
                # Remaining steps use normal model
                for _ in range(n_steps - 1):
                    running_models.append(normal_int_model)
            
            # Create terminal model
            terminal_target = waypoints[-1]
            terminal_cost = self.create_cost_model(target_state=terminal_target, 
                                                 state_weight=10.0,  # Higher weight for terminal state
                                                 is_terminal=True,
                                                 is_waypoint=True,  # Terminal is also a waypoint
                                                 waypoint_multiplier=waypoint_multiplier)
            
            terminal_diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, terminal_cost
            )
            terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff_model, 0.0)
            
            # Create shooting problem
            self.problem = crocoddyl.ShootingProblem(initial_state, running_models, terminal_model)
            
            total_time = sum(durations)
            print(f"✓ Successfully created trajectory optimization problem")
            print(f"  - Number of waypoints: {len(waypoints)}")
            print(f"  - Number of running nodes: {len(running_models)}")
            print(f"  - Time step: {dt}s")
            print(f"  - Total time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Failed to create trajectory optimization problem: {e}")
            raise
            
    def solve_trajectory(self, max_iter: int = 100, verbose: bool = True) -> bool:
        """
        Solve trajectory optimization problem
        
        Args:
            max_iter: Maximum number of iterations
            verbose: Whether to display detailed information
            
        Returns:
            Whether converged
        """
        try:
            if self.problem is None:
                raise RuntimeError("Please create trajectory optimization problem first")
            
            # Create solver
            # self.solver = crocoddyl.SolverBoxFDDP(self.problem)
            # self.solver = eagle_mpc.SolverSbFDDP(self.problem, self.problem.squash)
            self.solver = crocoddyl.SolverBoxDDP(self.problem)
            
            # Set solver parameters
            self.solver.convergence_init = 1e-12
            self.solver.convergence_stop = 1e-12
            
            # Set callback functions
            callbacks = []
            if verbose:
                callbacks.append(crocoddyl.CallbackVerbose())
            
            # Add logging
            logger = crocoddyl.CallbackLogger()
            callbacks.append(logger)
            
            self.solver.setCallbacks(callbacks)
            
            print(f"Starting trajectory optimization...")
            print(f"Maximum iterations: {max_iter}")
            
            # Record start time
            start_time = time.time()
            
            # Solve
            converged = self.solver.solve([], [], max_iter)
            
            # Record end time
            end_time = time.time()
            solve_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            print(f"✓ Trajectory optimization completed")
            print(f"  - Solve time: {solve_time:.2f} ms")
            print(f"  - Convergence status: {'Converged' if converged else 'Not converged'}")
            print(f"  - Final cost: {self.solver.cost:.6f}")
            print(f"  - Iterations: {self.solver.iter}")
            
            return converged
            
        except Exception as e:
            print(f"✗ Failed to solve trajectory optimization problem: {e}")
            raise
            
    def get_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get optimized trajectory
        
        Returns:
            (states, controls): State trajectory and control trajectory
        """
        if self.solver is None:
            raise RuntimeError("Please solve trajectory optimization problem first")
            
        return self.solver.xs, self.solver.us
        
    def _identify_waypoint_indices(self) -> List[int]:
        """
        Identify waypoint indices in the trajectory
        
        Returns:
            List of waypoint indices
        """
        if not hasattr(self, '_waypoint_times') or self.dt is None:
            return []
            
        waypoint_indices = []
        for wp_time in self._waypoint_times:
            wp_idx = int(wp_time / self.dt)
            waypoint_indices.append(wp_idx)
            
        return waypoint_indices
        
    def plot_trajectory(self, save_path: Optional[str] = None, show_waypoints: bool = True):
        """
        Plot trajectory results
        
        Args:
            save_path: Save path (optional)
            show_waypoints: Whether to show waypoint annotations (default True)
        """
        if self.solver is None:
            print("✗ Please solve trajectory optimization problem first")
            return
            
        states, controls = self.get_trajectory()
        
        # Check if dt is available
        if self.dt is None:
            print("⚠️ Warning: dt not set, using default value 0.02s")
            dt = 0.02
        else:
            dt = self.dt
            
        # Time axis
        time_states = np.arange(len(states)) * dt
        time_controls = np.arange(len(controls)) * dt
        
        # Identify waypoint indices
        waypoint_indices = []
        if show_waypoints:
            waypoint_indices = self._identify_waypoint_indices()
            print(f"🎯 Found {len(waypoint_indices)} waypoints for plotting (dt={dt:.3f}s)")
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('S500 Trajectory Optimization Results', fontsize=16)
        
        # Debug: Print state structure information
        if len(states) > 0:
            print(f"🔍 State vector debugging:")
            print(f"   State dimension: {len(states[0])}")
            print(f"   First state: {states[0]}")
            print(f"   Robot model nq (position): {self.robot_model.nq}")
            print(f"   Robot model nv (velocity): {self.robot_model.nv}")
            print(f"   State ndx: {self.state.ndx}")
        
        # Extract state data - For Free Flyer: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        positions = np.array([x[:3] for x in states])  # x, y, z
        orientations = np.array([x[3:7] for x in states])  # quaternion qx, qy, qz, qw
        velocities = np.array([x[7:10] for x in states])  # linear velocity vx, vy, vz
        angular_velocities = np.array([x[10:13] for x in states])  # angular velocity wx, wy, wz
        
        # Debug: Print extracted position data
        print(f"🔍 Position data debugging:")
        print(f"   Positions shape: {positions.shape}")
        print(f"   Position range - X: [{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}]")
        print(f"   Position range - Y: [{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}]")
        print(f"   Position range - Z: [{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}]")
        
        # Plot position
        axes[0, 0].plot(time_states, positions[:, 0], 'r-', label='x', linewidth=2)
        axes[0, 0].plot(time_states, positions[:, 1], 'g-', label='y', linewidth=2)
        axes[0, 0].plot(time_states, positions[:, 2], 'b-', label='z', linewidth=2)
        
        # Add waypoint annotations
        if show_waypoints and waypoint_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if wp_idx < len(time_states):
                    axes[0, 0].axvline(x=time_states[wp_idx], color='orange', linestyle='--', alpha=0.7)
                    axes[0, 0].text(time_states[wp_idx], axes[0, 0].get_ylim()[1]*0.9, 
                                   f'WP{i+1}', rotation=90, ha='right', va='top',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Position Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot linear velocity
        axes[0, 1].plot(time_states, velocities[:, 0], 'r-', label='vx', linewidth=2)
        axes[0, 1].plot(time_states, velocities[:, 1], 'g-', label='vy', linewidth=2)
        axes[0, 1].plot(time_states, velocities[:, 2], 'b-', label='vz', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Linear Velocity (m/s)')
        axes[0, 1].set_title('Linear Velocity Trajectory')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot angular velocity
        axes[0, 2].plot(time_states, angular_velocities[:, 0], 'r-', label='ωx', linewidth=2)
        axes[0, 2].plot(time_states, angular_velocities[:, 1], 'g-', label='ωy', linewidth=2)
        axes[0, 2].plot(time_states, angular_velocities[:, 2], 'b-', label='ωz', linewidth=2)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 2].set_title('Angular Velocity Trajectory')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot control inputs
        if len(controls) > 0:
            controls_array = np.array(controls)
            colors = ['r', 'g', 'b', 'orange']
            for i in range(min(4, controls_array.shape[1])):  # Show maximum 4 rotors
                axes[1, 0].plot(time_controls, controls_array[:, i], 
                               color=colors[i], label=f'Rotor{i+1}', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Thrust (N)')
            axes[1, 0].set_title('Control Inputs (Rotor Thrusts)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot quaternion
        axes[1, 1].plot(time_states, orientations[:, 0], 'r-', label='qx', linewidth=2)
        axes[1, 1].plot(time_states, orientations[:, 1], 'g-', label='qy', linewidth=2)
        axes[1, 1].plot(time_states, orientations[:, 2], 'b-', label='qz', linewidth=2)
        axes[1, 1].plot(time_states, orientations[:, 3], 'orange', label='qw', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Quaternion')
        axes[1, 1].set_title('Orientation (Quaternion)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 3D trajectory
        ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=3)
        ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                     color='g', s=100, label='Start')
        ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                     color='r', s=100, label='End')
        
        # Add waypoint annotations to 3D plot
        if show_waypoints and hasattr(self, '_waypoint_positions'):
            for i, wp_pos in enumerate(self._waypoint_positions):
                ax_3d.scatter(wp_pos[0], wp_pos[1], wp_pos[2], 
                             color='orange', s=150, marker='*', 
                             label='Waypoints' if i == 0 else "", alpha=0.8)
                ax_3d.text(wp_pos[0], wp_pos[1], wp_pos[2] + 0.1, 
                          f'WP{i+1}', fontsize=10, ha='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Trajectory')
        ax_3d.legend()
        
        # Set equal aspect ratio for better visualization
        # Get the range of each axis
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()
        
        # Set minimum display range to avoid showing numerical noise
        min_display_range = 0.1  # 10cm minimum display range
        x_range = max(x_range, min_display_range)
        y_range = max(y_range, min_display_range)
        z_range = max(z_range, min_display_range)
        
        max_range = max(x_range, y_range, z_range)
        
        # Set equal limits around the center
        x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
        y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
        z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
        
        if max_range > 0:
            ax_3d.set_xlim([x_center - max_range/2, x_center + max_range/2])
            ax_3d.set_ylim([y_center - max_range/2, y_center + max_range/2])
            ax_3d.set_zlim([z_center - max_range/2, z_center + max_range/2])
        
        print(f"🔍 3D Plot axis ranges:")
        print(f"   Original - X: {positions[:, 0].max() - positions[:, 0].min():.6f}, Y: {positions[:, 1].max() - positions[:, 1].min():.6f}, Z: {positions[:, 2].max() - positions[:, 2].min():.6f}")
        print(f"   Display - X: {x_range:.6f}, Y: {y_range:.6f}, Z: {z_range:.6f}")
        print(f"   Max range: {max_range:.6f}")
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Trajectory plot saved to: {save_path}")
        
        plt.show()
        
    def save_trajectory(self, save_path: str):
        """
        Save optimized trajectory data
        
        Args:
            save_path: Save path
        """
        if self.solver is None:
            print("✗ Please solve trajectory optimization problem first")
            return
            
        states, controls = self.get_trajectory()
        
        # Save as numpy arrays
        np.savez(save_path, 
                states=np.array(states),
                controls=np.array(controls),
                cost=self.solver.cost,
                iterations=self.solver.iter,
                s500_config=self.s500_config)
        
        print(f"✓ Trajectory data saved to: {save_path}")


def create_simple_waypoints() -> Tuple[List[np.ndarray], List[float]]:
    """
    Create simple test waypoints
    
    Returns:
        (waypoints, durations): Waypoints and durations
    """
    # State vector: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    waypoints = []
    
    # Starting point: ground level
    start = np.zeros(13)
    start[6] = 1.0  # qw = 1 (unit quaternion)
    waypoints.append(start)
    
    # First waypoint: ascend to 1m
    wp1 = start.copy()
    wp1[2] = 1.0  # z = 1m
    waypoints.append(wp1)
    
    # Second waypoint: move forward 2m
    wp2 = wp1.copy()
    wp2[0] = 2.0  # x = 2m
    waypoints.append(wp2)
    
    # Third waypoint: move right 2m
    wp3 = wp2.copy()
    wp3[1] = 2.0  # y = 2m
    waypoints.append(wp3)
    
    # Fourth waypoint: return to above origin
    wp4 = wp1.copy()
    waypoints.append(wp4)
    
    # Durations (seconds)
    durations = [2.0, 3.0, 3.0, 3.0]  # Duration of each trajectory segment
    # durations = [2.0]
    
    return waypoints, durations


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='S500 Quadrotor Trajectory Planning')
    parser.add_argument('--s500-yaml', type=str, help='Path to S500 configuration YAML file')
    parser.add_argument('--urdf', type=str, help='Path to URDF model file')
    parser.add_argument('--max-iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (seconds)')
    parser.add_argument('--save-dir', type=str, help='Results save directory')
    parser.add_argument('--no-thrust-constraints', action='store_true', help='Disable thrust constraints')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("S500 Quadrotor Trajectory Planning Script")
    print("Using Crocoddyl and Pinocchio for trajectory optimization")
    print("=" * 80)
    
    try:
        # Create trajectory planner
        planner = S500TrajectoryPlanner(s500_yaml_path=args.s500_yaml, 
                                       urdf_path=args.urdf)
        
        # Create test waypoints
        waypoints, durations = create_simple_waypoints()
        
        print(f"\nPath planning:")
        for i, (wp, dur) in enumerate(zip(waypoints, durations + [0])):
            pos = wp[:3]
            print(f"  Waypoint{i}: position=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]", end="")
            if i < len(durations):
                print(f", duration={dur:.1f}s")
            else:
                print(" (endpoint)")
        
        # Create trajectory optimization problem
        use_constraints = not args.no_thrust_constraints
        print(f"🔒 Thrust constraints: {'Enabled' if use_constraints else 'Disabled'}")
        planner.create_trajectory_problem(waypoints, durations, dt=args.dt, 
                                        use_thrust_constraints=use_constraints)
        
        # Solve
        converged = planner.solve_trajectory(max_iter=args.max_iter, verbose=True)
        
        if converged:
            print("\n" + "=" * 80)
            print("Trajectory optimization completed successfully!")
            print("=" * 80)
            
            # Set save directory
            if args.save_dir:
                save_dir = args.save_dir
            else:
                save_dir = os.path.join(planner.package_path, 'results', 's500_trajectory_optimization')
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot results
            plot_path = os.path.join(save_dir, 's500_trajectory_optimization.png')
            planner.plot_trajectory(save_path=plot_path)
            
            # Save trajectory data
            data_path = os.path.join(save_dir, 's500_trajectory_optimization.npz')
            planner.save_trajectory(data_path)
            
        else:
            print("\n" + "=" * 80)
            print("Trajectory optimization did not converge, please check parameter settings")
            print("Suggestions:")
            print("  - Increase maximum iterations (--max-iter)")
            print("  - Adjust time step (--dt)")
            print("  - Check if waypoints are reasonable")
            print("=" * 80)
            
    except Exception as e:
        print(f"\n✗ Program execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
