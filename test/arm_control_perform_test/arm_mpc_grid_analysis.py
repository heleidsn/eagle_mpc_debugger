#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Analyze MPC outputs for different joint angles in a grid pattern
'''

import numpy as np
import matplotlib.pyplot as plt
from utils.create_problem import get_opt_traj, create_mpc_controller
from utils.u_convert import thrustToForceTorqueAll_array
from scipy.spatial.transform import Rotation as R
import time
import pinocchio as pin
from mpl_toolkits.mplot3d import Axes3D

def analyze_mpc_grid(robot_name, trajectory_name, dt_traj_opt, useSquash, yaml_path, use_3d=False):
    """Analyze MPC outputs for different joint angles in a grid pattern
    
    Args:
        robot_name: Name of the robot
        trajectory_name: Name of the trajectory
        dt_traj_opt: Trajectory optimization time step
        useSquash: Whether to use squashing for control inputs
        yaml_path: Path to yaml configuration files
        use_3d: Whether to use 3D plotting (default: False)
    """
    
    # Load trajectory and create MPC controller
    trajectory, traj_state_ref, _, trajectory_obj = get_opt_traj(
        robot_name,
        trajectory_name, 
        dt_traj_opt, 
        useSquash,
        yaml_path
    )
    
    # Create MPC controller
    mpc_name = "rail"
    mpc_yaml = '{}/mpc/{}_mpc.yaml'.format(yaml_path, robot_name)    
    
    initial_state = trajectory_obj.initial_state
    hover_state_ref = [initial_state] * len(traj_state_ref)
    
    # mpc_controller = create_mpc_controller(
    #     mpc_name,
    #     trajectory_obj,
    #     traj_state_ref,
    #     dt_traj_opt,
    #     mpc_yaml
    # )
    
    mpc_controller = create_mpc_controller(
        mpc_name,
        trajectory_obj,
        hover_state_ref,  # Use hover reference state
        dt_traj_opt,
        mpc_yaml
    )
    
    
    # Get robot model
    model = mpc_controller.robot_model
    data = model.createData()
    
    # Create grid of joint angles
    joint1_range = np.linspace(-1.0, 1.0, 11)  # 11 points from -1 to 1
    joint2_range = np.linspace(-1.0, 1.0, 11)  # 11 points from -1 to 1
    joint1_grid, joint2_grid = np.meshgrid(joint1_range, joint2_range)
    
    # Initialize arrays to store results
    control_joint1 = np.zeros_like(joint1_grid)
    control_joint2 = np.zeros_like(joint2_grid)
    solving_times = np.zeros_like(joint1_grid)
    costs = np.zeros_like(joint1_grid)
    gripper_positions = []
    
    # Get initial state from trajectory object
    initial_state = trajectory_obj.initial_state
    
    # Analyze each grid point
    for i in range(len(joint1_range)):
        for j in range(len(joint2_range)):
            # Create state vector using initial state as base
            current_state = initial_state.copy()
            current_state[7] = joint1_grid[i,j]  # Update joint1 angle
            current_state[8] = joint2_grid[i,j]  # Update joint2 angle
            
            # Set initial state for MPC
            mpc_controller.problem.x0 = current_state
            
            # Solve MPC
            time_start = time.time()
            mpc_controller.solver.solve(
                mpc_controller.solver.xs,
                mpc_controller.solver.us,
                mpc_controller.iters
            )
            time_end = time.time()
            
            # Store results
            control_command = mpc_controller.solver.us_squash[0]
            control_joint1[i,j] = control_command[-2]  # Joint 1 torque
            control_joint2[i,j] = control_command[-1]  # Joint 2 torque
            solving_times[i,j] = time_end - time_start
            costs[i,j] = mpc_controller.logger.costs[-1]
            
            # Calculate gripper position
            q = current_state[:model.nq]
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            
            try:
                gripper_frame_id = model.getFrameId("gripper_link")
                gripper_pose = data.oMf[gripper_frame_id]
                gripper_positions.append({
                    'joint1': joint1_grid[i,j],
                    'joint2': joint2_grid[i,j],
                    'position': gripper_pose.translation
                })
            except:
                print(f"Could not find gripper_link frame at joint1={joint1_grid[i,j]}, joint2={joint2_grid[i,j]}")
    
    # Plot results
    plot_grid_analysis_results(joint1_grid, joint2_grid, control_joint1, control_joint2, 
                             solving_times, costs, gripper_positions, use_3d)

def plot_grid_analysis_results(joint1_grid, joint2_grid, control_joint1, control_joint2, 
                             solving_times, costs, gripper_positions, use_3d=False):
    """Plot grid analysis results
    
    Args:
        joint1_grid: Grid of joint1 angles
        joint2_grid: Grid of joint2 angles
        control_joint1: Control commands for joint1
        control_joint2: Control commands for joint2
        solving_times: MPC solving times
        costs: MPC costs
        gripper_positions: List of gripper positions
        use_3d: Whether to use 3D plotting (default: False)
    """
    
    if use_3d:
        # Create figure with subplots for 3D plotting
        fig = plt.figure(figsize=(20, 15))
        
        # Plot joint 1 control commands
        ax1 = fig.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(joint1_grid, joint2_grid, control_joint1, cmap='viridis')
        ax1.set_xlabel('Joint 1 Angle (rad)')
        ax1.set_ylabel('Joint 2 Angle (rad)')
        ax1.set_zlabel('Joint 1 Torque (Nm)')
        ax1.set_title('Joint 1 Control Commands')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # Plot joint 2 control commands
        ax2 = fig.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(joint1_grid, joint2_grid, control_joint2, cmap='viridis')
        ax2.set_xlabel('Joint 1 Angle (rad)')
        ax2.set_ylabel('Joint 2 Angle (rad)')
        ax2.set_zlabel('Joint 2 Torque (Nm)')
        ax2.set_title('Joint 2 Control Commands')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        # Plot solving times
        ax3 = fig.add_subplot(223, projection='3d')
        surf3 = ax3.plot_surface(joint1_grid, joint2_grid, solving_times*1000, cmap='viridis')
        ax3.set_xlabel('Joint 1 Angle (rad)')
        ax3.set_ylabel('Joint 2 Angle (rad)')
        ax3.set_zlabel('Solving Time (ms)')
        ax3.set_title('MPC Solving Time')
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
        
        # Plot costs
        ax4 = fig.add_subplot(224, projection='3d')
        surf4 = ax4.plot_surface(joint1_grid, joint2_grid, costs, cmap='viridis')
        ax4.set_xlabel('Joint 1 Angle (rad)')
        ax4.set_ylabel('Joint 2 Angle (rad)')
        ax4.set_zlabel('Cost')
        ax4.set_title('MPC Cost')
        fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        # Create additional figure for gripper positions
        fig2 = plt.figure(figsize=(10, 10))
        ax5 = fig2.add_subplot(111, projection='3d')
        
        # Extract gripper positions
        x_pos = [pos['position'][0] for pos in gripper_positions]
        y_pos = [pos['position'][1] for pos in gripper_positions]
        z_pos = [pos['position'][2] for pos in gripper_positions]
        
        # Plot gripper positions
        scatter = ax5.scatter(x_pos, y_pos, z_pos, c=range(len(gripper_positions)), cmap='viridis')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_zlabel('Z (m)')
        ax5.set_title('Gripper Positions for Different Joint Angles')
        fig2.colorbar(scatter, label='Sample Index')
        
        plt.tight_layout()
    else:
        # Create figure with subplots for 2D plotting
        fig = plt.figure(figsize=(20, 15))
        
        # Get the joint angles for plotting
        joint1_angles = joint1_grid[0,:]  # First row contains all joint1 angles
        joint2_angles = joint2_grid[:,0]  # First column contains all joint2 angles
        
        # Find the index where joint2 is closest to 0
        joint2_zero_idx = np.abs(joint2_angles - 0).argmin()
        joint1_zero_idx = np.abs(joint1_angles - 0).argmin()
        
        # Plot joint 1 control commands vs joint1 angle (joint2 fixed at 0)
        ax1 = fig.add_subplot(221)
        ax1.plot(joint1_angles, control_joint1[joint2_zero_idx,:], 'b-', label='Joint 2 = 0')
        ax1.set_xlabel('Joint 1 Angle (rad)')
        ax1.set_ylabel('Joint 1 Torque (Nm)')
        ax1.set_title('Joint 1 Control Commands (Joint 2 = 0)')
        ax1.grid(True)
        
        # Plot joint 2 control commands vs joint2 angle (joint1 fixed at 0)
        ax2 = fig.add_subplot(222)
        ax2.plot(joint2_angles, control_joint2[:,joint1_zero_idx], 'r-', label='Joint 1 = 0')
        ax2.set_xlabel('Joint 2 Angle (rad)')
        ax2.set_ylabel('Joint 2 Torque (Nm)')
        ax2.set_title('Joint 2 Control Commands (Joint 1 = 0)')
        ax2.grid(True)
        
        # Plot solving times
        ax3 = fig.add_subplot(223)
        ax3.plot(joint1_angles, solving_times[joint2_zero_idx,:]*1000, 'b-', label='Joint 2 = 0')
        ax3.set_xlabel('Joint 1 Angle (rad)')
        ax3.set_ylabel('Solving Time (ms)')
        ax3.set_title('MPC Solving Time (Joint 2 = 0)')
        ax3.grid(True)
        
        # Plot costs
        ax4 = fig.add_subplot(224)
        ax4.plot(joint1_angles, costs[joint2_zero_idx,:], 'b-', label='Joint 2 = 0')
        ax4.set_xlabel('Joint 1 Angle (rad)')
        ax4.set_ylabel('Cost')
        ax4.set_title('MPC Cost (Joint 2 = 0)')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Create additional figure for gripper positions
        fig2 = plt.figure(figsize=(10, 10))
        ax5 = fig2.add_subplot(111)
        
        # Extract gripper positions for joint2 = 0
        x_pos = [pos['position'][0] for pos in gripper_positions if abs(pos['joint2']) < 1e-6]
        y_pos = [pos['position'][1] for pos in gripper_positions if abs(pos['joint2']) < 1e-6]
        
        # Plot gripper positions
        scatter = ax5.scatter(x_pos, y_pos, c=range(len(x_pos)), cmap='viridis')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title('Gripper Positions (Joint 2 = 0)')
        fig2.colorbar(scatter, label='Sample Index')
        ax5.grid(True)
        
        plt.tight_layout()
    
    plt.show()

def main():
    # Settings
    robot_name = 's500_uam'
    trajectory_name = 'catch_vicon'
    dt_traj_opt = 10  # ms
    useSquash = True
    yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_debugger/config/yaml'
    use_3d = False  # Default to 2D plotting
    
    print(f"Analyzing MPC outputs for {robot_name}")
    print(f"Parameters:")
    print(f"  dt_traj_opt: {dt_traj_opt} ms")
    print(f"  useSquash: {useSquash}")
    print(f"  use_3d: {use_3d}")
    
    analyze_mpc_grid(robot_name, trajectory_name, dt_traj_opt, useSquash, yaml_path, use_3d)

if __name__ == '__main__':
    main() 