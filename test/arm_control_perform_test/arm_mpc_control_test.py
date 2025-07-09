#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-12-19
Description: 2DOF robotic arm angle tracking optimization problem using Crocoddyl MPC controller
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

class TwoDOFArmMPCController:
    def __init__(self, urdf_path, dt=0.01, horizon_length=50, enable_visualization=False, target_change_interval=2.0):
        """
        Initialize 2DOF robotic arm MPC controller
        
        Args:
            urdf_path: URDF file path
            dt: Time step
            horizon_length: Prediction horizon length
            enable_visualization: Enable Gepetto visualization
            target_change_interval: Time interval (seconds) for changing target
        """
        self.dt = dt
        self.horizon_length = horizon_length
        self.enable_visualization = enable_visualization
        
        # Load robot model
        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()
        
        # State and control dimensions
        self.state_dim = self.robot.nq + self.robot.nv  # position + velocity
        self.control_dim = self.robot.nv  # joint torques
        
        # Target positions (radians) - will be updated dynamically
        self.target_positions = np.array([0.5, 0.3])  # initial target angles for joint_1, joint_2
        
        # Dynamic target configuration
        self.target_change_interval = target_change_interval  # seconds
        self.target_sequence = [
            np.array([0.5, 0.3]),   # 0-2s
            np.array([1.0, 0.8]),   # 2-4s
            np.array([-0.5, 0.5]),  # 4-6s
            np.array([0.0, 1.2]),   # 6-8s
            np.array([0.8, -0.3]),  # 8-10s
            np.array([0.2, 0.0]),   # 10-12s
            np.array([1.2, 0.6]),   # 12-14s
            np.array([-0.3, 1.0]),  # 14-16s
            np.array([0.6, -0.5]),  # 16-18s
            np.array([0.1, 0.7]),   # 18-20s
        ]
        self.current_target_index = 0
        
        # Weight parameters (used directly in cost functions)
        self.state_weight = 1.0  # state tracking weight
        self.control_weight = 0.1  # control weight
        
        # Create optimization problem
        self.problem = self.create_optimization_problem()
        
        # Create solver
        self.solver = crocoddyl.SolverBoxFDDP(self.problem)
        self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
        
        # Create state update model for simulation
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
        
        print(f"2DOF robotic arm MPC controller initialized")
        print(f"State dimension: {self.state_dim}, Control dimension: {self.control_dim}")
        print(f"Robot nq: {self.robot.nq}, nv: {self.robot.nv}")
        print(f"Robot joint names: {[self.robot.names[i] for i in range(self.robot.nq)]}")
        print(f"Robot joint types: {[self.robot.joints[i].shortname() for i in range(1, self.robot.nq+1)]}")
        print(f"Initial target positions: {self.target_positions}")
        print(f"Target change interval: {self.target_change_interval} seconds")
        print(f"Number of target sequences: {len(self.target_sequence)}")
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
        
        # Add state tracking cost
        state_tracking_cost = crocoddyl.CostModelResidual(
            state, 
            crocoddyl.ResidualModelState(state, self.get_target_state(), self.control_dim)
        )
        action_model.costs.addCost("state_tracking", state_tracking_cost, self.state_weight)
        
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
        
        # Add terminal state tracking cost (higher weight)
        terminal_state_cost = crocoddyl.CostModelResidual(
            state, 
            crocoddyl.ResidualModelState(state, self.get_target_state(), self.control_dim)
        )
        action_model.costs.addCost("terminal_state", terminal_state_cost, self.state_weight * 10.0)
        
        # Create integrated model
        integrated_model = crocoddyl.IntegratedActionModelEuler(action_model, self.dt)
        
        return integrated_model
    
    def get_target_state(self):
        """Get target state (position and velocity)"""
        target_state = np.zeros(self.state_dim)
        target_state[:self.robot.nq] = self.target_positions  # target position
        target_state[self.robot.nq:] = np.zeros(self.robot.nv)  # target velocity (zero)
        return target_state
    
    def solve_mpc(self, current_state, max_iterations=100):
        """
        Solve MPC problem
        
        Args:
            current_state: Current state
            max_iterations: Maximum iterations
            
        Returns:
            optimal_control: Optimal control sequence
            optimal_states: Optimal state sequence
        """
        # Update problem initial state
        self.problem.x0 = current_state
        
        # Solve optimization problem
        self.solver.solve([], [], max_iterations)
        
        # Get optimal solution
        optimal_states = self.solver.xs
        optimal_controls = self.solver.us
        
        return optimal_controls, optimal_states
    
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
            
            # Solve MPC problem
            optimal_controls, optimal_states = self.solve_mpc(current_state)
            
            # Debug: Print control information
            print(f"Optimal controls shape: {len(optimal_controls)}")
            print(f"First control shape: {optimal_controls[0].shape if hasattr(optimal_controls[0], 'shape') else type(optimal_controls[0])}")
            print(f"First control value: {optimal_controls[0]}")
            
            # Record control input
            self.control_data.append(optimal_controls[0].copy())
            
            # Record cost
            self.cost_data.append(self.solver.cost)
            
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
        Update target position based on current time
        
        Args:
            current_time: Current simulation time
        """
        target_index = int(current_time / self.target_change_interval)
        
        # Cycle through target sequence
        if target_index < len(self.target_sequence):
            new_target = self.target_sequence[target_index]
        else:
            # If we exceed the sequence, cycle back to the beginning
            new_target = self.target_sequence[target_index % len(self.target_sequence)]
        
        # Only update if target has changed
        if not np.array_equal(self.target_positions, new_target):
            self.target_positions = new_target.copy()
            self.current_target_index = target_index
            print(f"Time {current_time:.2f}s: Target changed to {self.target_positions}")
            
            # Recreate optimization problem with new target
            self.problem = self.create_optimization_problem()
            self.solver = crocoddyl.SolverBoxFDDP(self.problem)
            self.solver.setCallbacks([crocoddyl.CallbackVerbose()])

def main():
    """Main function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='2DOF Robotic Arm MPC Control Test')
    parser.add_argument('--visualization', action='store_true', default=False,
                      help='Enable Gepetto visualization')
    parser.add_argument('--simulation-time', type=float, default=20.0,
                      help='Simulation time in seconds')
    parser.add_argument('--horizon-length', type=int, default=30,
                      help='MPC horizon length')
    parser.add_argument('--dt', type=float, default=0.01,
                      help='Time step')
    parser.add_argument('--target-change-interval', type=float, default=4.0,
                      help='Target change interval in seconds')
    
    args = parser.parse_args()
    
    # URDF file path
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'models', 'urdf', 's500_uam_simple.urdf')
    
    # Check if URDF file exists
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file does not exist: {urdf_path}")
        return
    
    # Create MPC controller
    mpc_controller = TwoDOFArmMPCController(
        urdf_path=urdf_path,
        dt=args.dt,                    # time step
        horizon_length=args.horizon_length,  # prediction horizon length
        enable_visualization=args.visualization,  # enable visualization
        target_change_interval=args.target_change_interval
    )
    
    # Set initial state [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Set target positions
    mpc_controller.target_positions = np.array([0.5, 0.3])  # radians
    
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
