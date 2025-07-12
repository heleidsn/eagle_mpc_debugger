#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-12-19
Description: 2DOF robotic arm angle tracking optimization problem using Crocoddyl MPC controller
Supports both simulation mode and ROS node mode
'''

import numpy as np
import crocoddyl
import pinocchio as pin
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import example_robot_data
import gepetto
import sys
import time

# PID Controller Class
class PIDController:
    """
    PID Controller for 2DOF robotic arm
    """
    def __init__(self, kp, ki, kd, dt, joint_names=None):
        """
        Initialize PID Controller
        
        Args:
            kp: Proportional gains [joint1, joint2]
            ki: Integral gains [joint1, joint2] 
            kd: Derivative gains [joint1, joint2]
            dt: Time step
            joint_names: Joint names
        """
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        self.joint_names = joint_names or ['joint_1', 'joint_2']
        
        # Initialize error integrals and previous errors
        self.error_integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_time = None
        
        # Anti-windup limits
        self.integral_limit = 1.0
        self.output_limit = 0.2
        
        print(f"PID Controller initialized:")
        print(f"  Kp: {self.kp}")
        print(f"  Ki: {self.ki}")
        print(f"  Kd: {self.kd}")
        print(f"  dt: {self.dt}")
        print(f"  Joint names: {self.joint_names}")
    
    def reset(self):
        """Reset PID controller state"""
        self.error_integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_time = None
        print("PID Controller reset")
    
    def compute_control(self, current_positions, target_positions, current_velocities=None):
        """
        Compute PID control output
        
        Args:
            current_positions: Current joint positions [joint1, joint2]
            target_positions: Target joint positions [joint1, joint2]
            current_velocities: Current joint velocities [joint1, joint2] (optional)
            
        Returns:
            control_output: Control torques [joint1, joint2]
        """
        current_positions = np.array(current_positions)
        target_positions = np.array(target_positions)
        
        # Calculate position errors
        position_errors = target_positions - current_positions
        
        # Calculate time step
        current_time = time.time()
        if self.prev_time is None:
            dt = self.dt
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # Update integral term with anti-windup
        self.error_integral += position_errors * dt
        self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)
        
        # Calculate derivative term
        if current_velocities is not None:
            # Use actual velocity feedback if available
            velocity_errors = -np.array(current_velocities)  # Negative because we want to reduce velocity
        else:
            # Use finite difference approximation
            velocity_errors = (position_errors - self.prev_error) / dt if dt > 0 else np.zeros(2)
        
        # PID control law
        control_output = (self.kp * position_errors + 
                         self.ki * self.error_integral + 
                         self.kd * velocity_errors)
        
        # Apply output limits
        control_output = np.clip(control_output, -self.output_limit, self.output_limit)
        
        # Update previous error
        self.prev_error = position_errors.copy()
        
        return control_output
    
    def get_debug_info(self):
        """Get debug information for the PID controller"""
        return {
            'error_integral': self.error_integral.copy(),
            'prev_error': self.prev_error.copy(),
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd
        }

# ROS imports (only used when running in ROS mode)
try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. ROS mode will be disabled.")

class TwoDOFArmMPCController:
    def __init__(self, urdf_path, dt=0.01, horizon_length=50, enable_visualization=False, target_change_interval=2.0, control_mode='mpc'):
        """
        Initialize 2DOF robotic arm controller (MPC or PID)
        
        Args:
            urdf_path: URDF file path
            dt: Time step
            horizon_length: Prediction horizon length 
            enable_visualization: Enable Gepetto visualization
            target_change_interval: Time interval (seconds) for changing target
            control_mode: Control mode ('mpc' or 'pid')
        """
        self.dt = 0.05
        self.horizon_length = 40
        self.enable_visualization = enable_visualization
        self.control_mode = control_mode.lower()
        
        # Load robot model
        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()
        
        # State and control dimensions
        self.state_dim = self.robot.nq + self.robot.nv  # position + velocity
        self.control_dim = self.robot.nv  # joint torques
        
        # Target configuration
        self.target_change_interval = target_change_interval  # seconds
        
        # Target mode configuration
        self.use_dynamic_targets = True  # Use dynamic targets or fixed targets
        
        # Fixed target configuration (when use_dynamic_targets = False)
        self.fixed_target_positions = np.array([0.3, 0.6])  # Fixed target positions
        
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
        self.position_weight = 10000.0  # position tracking weight
        self.velocity_weight = 10.0  # velocity tracking weight
        self.control_weight = 0.1  # control weight
        
        # Create optimization problem
        self.problem = self.create_optimization_problem()
        
        # Initialize controller based on mode
        if self.control_mode == 'mpc':
            # Create solver
            self.solver = crocoddyl.SolverBoxFDDP(self.problem)
            self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
            
            # Set convergence parameters
            self.solver.th_stop = 1e-12
            
            # Create state update model for simulation
            self.create_state_update_model()
            
        elif self.control_mode == 'pid':
            # Initialize PID controller
            # Default PID gains - can be tuned
            kp = [10.0, 10.0]  # Proportional gains
            ki = [2.0, 2.0]    # Integral gains  
            kd = [1.0, 1.0]    # Derivative gains
            
            self.pid_controller = PIDController(kp, ki, kd, self.dt)
            
        else:
            raise ValueError(f"Unknown control mode: {control_mode}. Use 'mpc' or 'pid'")
        
        # Create state update model for simulation (only for MPC mode)
        if self.control_mode == 'mpc':
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
        
        self.xs = []
        self.us = []
        
        print(f"2DOF robotic arm {self.control_mode.upper()} controller initialized")
        print(f"Control mode: {self.control_mode}")
        print(f"State dimension: {self.state_dim}, Control dimension: {self.control_dim}")
        print(f"Robot nq: {self.robot.nq}, nv: {self.robot.nv}")
        print(f"Robot joint names: {[self.robot.names[i] for i in range(self.robot.nq)]}")
        print(f"Robot joint types: {[self.robot.joints[i].shortname() for i in range(1, self.robot.nq+1)]}")
        
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
    
    def init_visualization(self):
        """Initialize Gepetto visualization"""
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
        # Create differential model for state update
        state = crocoddyl.StateMultibody(self.robot)
        actuation = crocoddyl.ActuationModelFull(state)
        
        # Create differential action model
        differential_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, crocoddyl.CostModelSum(state, self.control_dim)
        )
        
        # Create integrated model using RK4
        self.state_update_model = crocoddyl.IntegratedActionModelRK4(differential_model, self.dt)
        
        # Create data for the integrated model
        self.state_update_data = self.state_update_model.createData()
        
        print("State update model created successfully")
    
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
        
        # Create integrated model
        integrated_model = crocoddyl.IntegratedActionModelEuler(action_model, self.dt)
        
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
        
        # Create integrated model
        integrated_model = crocoddyl.IntegratedActionModelEuler(action_model, self.dt)
        
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
            
            # Solve control problem (MPC or PID)
            optimal_controls, optimal_states, solve_time, iterations = self.compute_control(current_state)
            
            # Debug: Print control information
            print(f"Optimal controls shape: {len(optimal_controls)}")
            print(f"First control shape: {optimal_controls[0].shape if hasattr(optimal_controls[0], 'shape') else type(optimal_controls[0])}")
            print(f"First control value: {optimal_controls[0]}")
            
            # Record control input
            self.control_data.append(optimal_controls[0].copy())
            
            # Record cost and solver info
            self.cost_data.append(self.solver.cost)
            
            # Print solver performance
            print(f"MPC Solve Time: {solve_time*1000:.1f}ms, Iterations: {iterations}, Cost: {self.solver.cost:.6f}")
            
            # Apply first control input
            control_input = optimal_controls[0]  # This should be a 2D vector
            
            # Ensure control_input is the right shape and type
            control_input = np.array(control_input, dtype=np.float64)
            if control_input.size != 2:
                print(f"Warning: control_input size is {control_input.size}, expected 2")
                control_input = np.array([control_input[0], 0.0]) if control_input.size > 0 else np.array([0.0, 0.0])
            
            # Debug: Print control information
            print(f"Control input shape: {control_input.shape}, value: {control_input}")
            
            # Update state using the new calculate_next_state method
            next_state = self.calculate_next_state(current_state, control_input)
            
            # Update state and time
            current_state = next_state
            current_time += self.dt
            
            # Print progress
            if int(current_time * 10) % 10 == 0:
                print(f"Time: {current_time:.1f}s, Position: {current_state[:self.robot.nq]}, Cost: {self.solver.cost:.6f}")
            
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
        fig.suptitle('2DOF Robotic Arm MPC Angle Tracking Results (Dynamic Targets)', fontsize=16)
        
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
        
        # Cost function
        axes[1, 1].plot(time_array, cost_array, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Cost')
        axes[1, 1].set_title('Optimization Cost')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', f'arm_mpc_results_{timestamp}.png')
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
        data_path = os.path.join(data_dir, f'arm_mpc_data_{timestamp}.csv')
        
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
        header = "time,joint1_pos,joint2_pos,joint1_vel,joint2_vel,joint1_control,joint2_control,joint1_target,joint2_target,cost"
        np.savetxt(data_path, data_matrix, delimiter=',', header=header, comments='')
        print(f"Data saved to: {data_path}")

    def calculate_next_state(self, current_state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Calculate next state using pre-created Crocoddyl integrated model
        
        Args:
            current_state: Current state [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
            control: Control input [joint1_torque, joint2_torque]
            
        Returns:
            next_state: Next state
        """
        # Set current state and control
        self.state_update_data.x = current_state
        self.state_update_data.u = np.array(control, dtype=np.float64).flatten()
        
        # Compute next state using the pre-created integrated model
        self.state_update_model.calc(self.state_update_data, current_state, self.state_update_data.u)
        next_state = self.state_update_data.xnext
        
        return np.copy(next_state)

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
                self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
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
        self.control_rate = rospy.get_param('~control_rate', 500.0)  # Hz
        self.use_simulation = rospy.get_param('~use_simulation', False)
        
        # Set up subscribers
        if self.use_simulation:
            # Use simulation joint states topic
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
            self.joint1_pub = rospy.Publisher('/arm_controller/joint_1_position_controller/command', Float64, queue_size=10)
            self.joint2_pub = rospy.Publisher('/arm_controller/joint_2_position_controller/command', Float64, queue_size=10)
        
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
                    input_scale = control_input * 0.5
                    
                    # add limit to control input
                    input_scale = np.clip(input_scale, -0.2, 0.2)
                    
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
                        rospy.loginfo(f"Control command - joint_1: {input_scale[0]:.3f}, joint_2: {input_scale[1]:.3f}")
                        rospy.loginfo(f"Target positions: {self.mpc_controller.target_positions}")
                    
                    # For simulation, also publish individual joint commands
                    if self.use_simulation:
                        # control_input[0] corresponds to joint_1, control_input[1] corresponds to joint_2
                        self.joint1_pub.publish(Float64(control_input[0]))  # joint_1 control
                        self.joint2_pub.publish(Float64(control_input[1]))  # joint_2 control
                    
                    # Publish control input array
                    control_input_msg = Float64MultiArray()
                    control_input_msg.data = control_input.tolist()
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
                        rospy.loginfo_throttle(1.0, f"MPC Control: {control_input}, Target: {self.mpc_controller.target_positions}, Cost: {self.mpc_controller.solver.cost:.6f}, Solve Time: {solve_time*1000:.1f}ms, Iterations: {iterations}")
                    else:  # PID mode
                        cost = np.sum((self.mpc_controller.target_positions - self.current_state[:2])**2)
                        rospy.loginfo_throttle(1.0, f"PID Control: {control_input}, Target: {self.mpc_controller.target_positions}, Cost: {cost:.6f}, Solve Time: {solve_time*1000:.1f}ms")
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
    parser.add_argument('--simulation-time', type=float, default=20.0,
                      help='Simulation time in seconds')
    parser.add_argument('--target-change-interval', type=float, default=1000.0,
                      help='Target change interval in seconds')
    parser.add_argument('--ros', action='store_true', default=True,
                      help='Run as ROS node')
    parser.add_argument('--dynamic-targets', action='store_true', default=False,
                      help='Use dynamic changing targets')
    parser.add_argument('--fixed-targets', action='store_true', default=True,
                      help='Use fixed target positions')
    parser.add_argument('--fixed-joint1', type=float, default=0.3,
                      help='Fixed target position for joint 1 (when using fixed targets)')
    parser.add_argument('--fixed-joint2', type=float, default=0.6,
                      help='Fixed target position for joint 2 (when using fixed targets)')
    parser.add_argument('--control-mode', type=str, default='mpc', choices=['mpc', 'pid'],
                      help='Control mode: mpc or pid')
    parser.add_argument('--pid-kp', type=float, nargs=2, default=[20.0, 20.0],
                      help='PID proportional gains for joint 1 and joint 2')
    parser.add_argument('--pid-ki', type=float, nargs=2, default=[1.0, 1.0],
                      help='PID integral gains for joint 1 and joint 2')
    parser.add_argument('--pid-kd', type=float, nargs=2, default=[0.0, 0.0],
                      help='PID derivative gains for joint 1 and joint 2')
    
    args = parser.parse_args()
    
    # Check if ROS mode is requested but ROS is not available
    if args.ros and not ROS_AVAILABLE:
        print("Error: ROS mode requested but ROS is not available")
        print("Please install ROS and required packages")
        return
    
    # URDF file path
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'urdf', 's500_uam_arm_test.urdf')
    
    # Check if URDF file exists
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file does not exist: {urdf_path}")
        return
    
    # Create controller (MPC or PID)
    mpc_controller = TwoDOFArmMPCController(
        urdf_path=urdf_path,
        enable_visualization=args.visualization,  # enable visualization
        target_change_interval=args.target_change_interval,
        control_mode=args.control_mode
    )
    
    # Set PID gains if using PID mode
    if args.control_mode == 'pid':
        mpc_controller.pid_controller.kp = np.array(args.pid_kp)
        mpc_controller.pid_controller.ki = np.array(args.pid_ki)
        mpc_controller.pid_controller.kd = np.array(args.pid_kd)
        print(f"PID gains set - Kp: {args.pid_kp}, Ki: {args.pid_ki}, Kd: {args.pid_kd}")
    
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
