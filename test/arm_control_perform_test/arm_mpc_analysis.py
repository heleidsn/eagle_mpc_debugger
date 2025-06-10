#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Analyze arm control under different states using MPC while keeping the drone stationary
'''

import numpy as np
import matplotlib.pyplot as plt
from utils.create_problem import get_opt_traj, create_mpc_controller
from utils.u_convert import thrustToForceTorqueAll_array
from scipy.spatial.transform import Rotation as R
import time
import pinocchio as pin

def analyze_arm_control(robot_name, trajectory_name, dt_traj_opt, useSquash, yaml_path):
    """Analyze arm control under different states using MPC"""
    
    # Load trajectory and create MPC controller
    trajectory, traj_state_ref, _, trajectory_obj = get_opt_traj(
        robot_name,
        trajectory_name, 
        dt_traj_opt, 
        useSquash,
        yaml_path
    )
    
    initial_state = trajectory_obj.initial_state
    hover_state_ref = [initial_state] * len(traj_state_ref)
    
    # Create MPC controller
    mpc_name = "rail"
    mpc_yaml = '{}/mpc/{}_mpc.yaml'.format(yaml_path, robot_name)    
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
    
    # # Get robot model
    model = mpc_controller.robot_model
    data = model.createData()
    
    # # Get initial state
    
    # # Get control command of initial state
    # hover_mpc.problem.x0 = initial_state
    # hover_mpc.solver.solve(
    #     hover_mpc.solver.xs,
    #     hover_mpc.solver.us,
    #     hover_mpc.iters
    # )
    
    # control_command_initial = np.array(hover_mpc.solver.us_squash)
    
    # Define different arm states to test
    arm_states = [
        # State 1: Arm fully extended
        {'joint1': 0.0, 'joint2': 0.0},
        # State 2: Arm bent at joint1
        {'joint1': 1.0, 'joint2': 0.0},
        # State 3: Arm bent at joint2
        {'joint1': 0.0, 'joint2': 1.0},
        # State 4: Arm bent at both joints
        {'joint1': 1.0, 'joint2': 1.0},
        # State 5: Arm in extreme position
        {'joint1': -1.57, 'joint2': 1.57}
    ]
    
    # Initialize arrays to store results
    control_commands = []
    solving_times = []
    costs = []
    
    # Keep drone position and orientation constant
    drone_pos = traj_state_ref[0][:3]  # Initial position
    drone_quat = traj_state_ref[0][3:7]  # Initial orientation
    
    # Analyze each arm state
    for state in arm_states:
        # Create state vector
        current_state = np.zeros_like(traj_state_ref[0])
        current_state[:3] = drone_pos
        current_state[3:7] = drone_quat
        current_state[7] = state['joint1']
        current_state[8] = state['joint2']
        
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
        control_commands.append(mpc_controller.solver.us_squash[0])
        solving_times.append(time_end - time_start)
        costs.append(mpc_controller.logger.costs[-1])
        
        # Print results
        print(f"\nArm State: joint1={state['joint1']:.2f}, joint2={state['joint2']:.2f}")
        print(f"Solving time: {(time_end - time_start)*1000:.2f} ms")
        print(f"Final cost: {mpc_controller.logger.costs[-1]:.2f}")
        print(f"Control command: {mpc_controller.solver.us_squash[0]}")
        
        # Calculate and print gripper position
        q = current_state[:model.nq]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        try:
            gripper_frame_id = model.getFrameId("gripper_link")
            gripper_pose = data.oMf[gripper_frame_id]
            print(f"Gripper position: {gripper_pose.translation}")
        except:
            print("Could not find gripper_link frame")
    
    # Plot results
    plot_analysis_results(arm_states, control_commands, solving_times, costs)

def plot_analysis_results(arm_states, control_commands, solving_times, costs):
    """Plot analysis results"""
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot control commands
    ax1 = plt.subplot(221)
    control_commands = np.array(control_commands)
    x = range(len(arm_states))
    width = 0.35
    
    ax1.bar(x, control_commands[:, -2], width, label='Joint 1 Torque')
    ax1.bar([i + width for i in x], control_commands[:, -1], width, label='Joint 2 Torque')
    ax1.set_xlabel('Arm State')
    ax1.set_ylabel('Control Command (Nm)')
    ax1.set_title('Control Commands for Different Arm States')
    ax1.set_xticks([i + width/2 for i in x])
    ax1.set_xticklabels([f"State {i+1}" for i in range(len(arm_states))])
    ax1.legend()
    
    # Plot solving times
    ax2 = plt.subplot(222)
    ax2.bar(x, [t*1000 for t in solving_times])
    ax2.set_xlabel('Arm State')
    ax2.set_ylabel('Solving Time (ms)')
    ax2.set_title('MPC Solving Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"State {i+1}" for i in range(len(arm_states))])
    
    # Plot costs
    ax3 = plt.subplot(223)
    ax3.bar(x, costs)
    ax3.set_xlabel('Arm State')
    ax3.set_ylabel('Cost')
    ax3.set_title('MPC Cost')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"State {i+1}" for i in range(len(arm_states))])
    
    # Plot state information
    ax4 = plt.subplot(224)
    joint1_angles = [state['joint1'] for state in arm_states]
    joint2_angles = [state['joint2'] for state in arm_states]
    ax4.bar(x, joint1_angles, width, label='Joint 1')
    ax4.bar([i + width for i in x], joint2_angles, width, label='Joint 2')
    ax4.set_xlabel('Arm State')
    ax4.set_ylabel('Joint Angle (rad)')
    ax4.set_title('Joint Angles for Different States')
    ax4.set_xticks([i + width/2 for i in x])
    ax4.set_xticklabels([f"State {i+1}" for i in range(len(arm_states))])
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Settings
    robot_name = 's500_uam'
    trajectory_name = 'catch_vicon'
    dt_traj_opt = 10  # ms
    useSquash = True
    yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_debugger/config/yaml'
    
    print(f"Analyzing arm control for {robot_name}")
    print(f"Parameters:")
    print(f"  dt_traj_opt: {dt_traj_opt} ms")
    print(f"  useSquash: {useSquash}")
    
    analyze_arm_control(robot_name, trajectory_name, dt_traj_opt, useSquash, yaml_path)

if __name__ == '__main__':
    main() 