'''
Author: Lei He
Date: 2025-02-24 10:31:39
LastEditTime: 2025-06-16 14:29:20
Description: Run planning to generate planning results and save them to file
Github: https://github.com/heleidsn
'''
import numpy as np
import matplotlib.pyplot as plt
from utils.create_problem import get_opt_traj, create_mpc_controller
from utils.u_convert import thrustToForceTorqueAll_array
from tf.transformations import quaternion_matrix
from pathlib import Path
# import yaml
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path
import argparse

import example_robot_data
import crocoddyl
import time
# import gepetto
import pinocchio as pin

import tkinter as tk

def plot_trajectory(trajectory, trajectory_obj, traj_state_ref, control_force_torque, dt_traj_opt, state_array,  save_dir=None):
    """Plot optimized trajectory results.
    
    Args:
        trajectory: Trajectory object containing optimized data
        traj_state_ref: Reference state trajectory
        control_force_torque: Control inputs in force/torque format
        dt_traj_opt: Time step for trajectory optimization
        save_dir: Directory to save plots (optional)
    """
    # Create time vector
    n_points_control = len(control_force_torque)
    n_points_state = len(traj_state_ref)
    time_control = np.arange(n_points_control) * dt_traj_opt / 1000  # Convert to seconds
    time_state = np.arange(n_points_state) * dt_traj_opt / 1000  # Convert to seconds
    
    # Get robot model from trajectory
    model = trajectory_obj.robot_model
    data = model.createData()
    
    # Get number of joints
    joint_num = model.nq - 7  # Subtract 7 for base position (3) and orientation (4)
    # Get number of rotors from the model
    n_rotors = trajectory_obj.platform_params.n_rotors
    
    # Define key points from configuration
    start_pose_drone = [-1.5, 0, 1.5]
    start_joint_angle = [-1.2, -0.6]
    
    final_pose_drone = [1.5, 0, 1.5]
    final_joint_angle = [-1.2, -0.6]
    
    grasp_time = 3.0
    grasp_duration = 0.5
    
    object_pose = [0, 0, 0.8]
    
    grasp_pose_drone = None
    grasp_joint_angle = None
    
    
    # State labels (convert radians to degrees for roll, pitch, yaw)
    state_labels = ['x (m)', 'y (m)', 'z (m)', 
                   'roll (deg)', 'pitch (deg)', 'yaw (deg)',
                   'joint1 (deg)', 'joint2 (deg)', 'joint3 (deg)']
    
    # Set style parameters
    plt.rcParams['figure.figsize'] = [10, 6]  # Default figure size
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Common figure parameters
    COL_WIDTH = 5.5  # Width per column of subplots
    ROW_HEIGHT = 3.5  # Height per row of subplots
    
    # --------------------------------1. plot cost curve--------------------------------
    fig_cost = plt.figure(figsize=(COL_WIDTH, ROW_HEIGHT))
    plt.plot(trajectory_obj.logger.costs, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.title('Cost Convergence during DDP Optimization')
    plt.grid(True)
    plt.tight_layout()
    
    # --------------------------------2. plot  gripper states--------------------------------
    if joint_num > 0:
        gripper_positions = []
        gripper_orientations = []
        
        # Get gripper frame ID
        try:
            gripper_frame_id = model.getFrameId("gripper_link")
        except Exception as e:
            print("\nError: Could not find gripper_link in the robot model")
            print("Please check the URDF file for the correct gripper link name")
            print(f"Error details: {str(e)}")
            return
        
        for i in range(len(state_array)):
            # Get base position and orientation
            base_pos = state_array[i, :3]
            quat = state_array[i, 3:7]
            rot = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
            R_mat = rot.as_matrix()
            
            # Get joint angles
            joint_angles = state_array[i, 7:9]
        
            
            # Update robot configuration
            q = np.zeros(model.nq)
            q[0:3] = base_pos
            q[3:7] = quat
            q[7:9] = joint_angles
            
            # Forward kinematics
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            
            # Get gripper pose
            gripper_pose = data.oMf[gripper_frame_id]
            gripper_pos = gripper_pose.translation
            gripper_rot = R.from_matrix(gripper_pose.rotation)
            
            # Get Euler angles with different sequences
            gripper_euler_xyz = gripper_rot.as_euler('xyz', degrees=True)
            gripper_euler_zyx = gripper_rot.as_euler('zyx', degrees=True)
            
            # Choose the better representation based on pitch angle
            if abs(gripper_euler_xyz[1]) > 85:  # If pitch is near ±90 degrees
                gripper_euler = gripper_euler_zyx
            else:
                gripper_euler = gripper_euler_xyz
            
            # Normalize angles to [-180, 180] range
            gripper_euler = np.mod(gripper_euler + 180, 360) - 180
            
            gripper_positions.append(gripper_pos.copy())
            gripper_orientations.append(gripper_euler.copy())
        
        gripper_positions = np.array(gripper_positions)
        gripper_orientations = np.array(gripper_orientations)
        
        # Calculate gripper velocities
        dt = dt_traj_opt / 1000.0  # Convert to seconds
        gripper_linear_vel = np.zeros_like(gripper_positions)
        gripper_angular_vel = np.zeros_like(gripper_orientations)
        
        # Calculate velocities using central difference
        for i in range(1, len(gripper_positions)-1):
            # Linear velocity
            gripper_linear_vel[i] = (gripper_positions[i+1] - gripper_positions[i-1]) / (2 * dt)
            
            # Angular velocity calculation using quaternions
            quat_prev = R.from_euler('xyz', gripper_orientations[i-1], degrees=True).as_quat()
            quat_next = R.from_euler('xyz', gripper_orientations[i+1], degrees=True).as_quat()
            
            # Calculate relative rotation
            quat_diff = R.from_quat(quat_next) * R.from_quat(quat_prev).inv()
            angle_axis = quat_diff.as_rotvec()
            
            # Convert to angular velocity (in degrees/s)
            gripper_angular_vel[i] = np.degrees(angle_axis) / (2 * dt)
            
            # Check for gimbal lock and handle large angular velocities
            for j in range(3):
                if abs(gripper_angular_vel[i, j]) > 180/dt:  # If angular velocity is too large
                    # Use alternative calculation for this axis
                    angle_diff = gripper_orientations[i+1, j] - gripper_orientations[i-1, j]
                    angle_diff = np.mod(angle_diff + 180, 360) - 180
                    gripper_angular_vel[i, j] = angle_diff / (2 * dt)
        
        # Handle endpoints using forward/backward difference
        gripper_linear_vel[0] = (gripper_positions[1] - gripper_positions[0]) / dt
        gripper_linear_vel[-1] = (gripper_positions[-1] - gripper_positions[-2]) / dt
        
        # Handle angular velocity endpoints
        for i in [0, -1]:
            quat_curr = R.from_euler('xyz', gripper_orientations[i], degrees=True).as_quat()
            quat_next = R.from_euler('xyz', gripper_orientations[i+1 if i == 0 else i-1], degrees=True).as_quat()
            
            # Calculate relative rotation
            quat_diff = R.from_quat(quat_next) * R.from_quat(quat_curr).inv()
            angle_axis = quat_diff.as_rotvec()
            
            # Convert to angular velocity (in degrees/s)
            gripper_angular_vel[i] = np.degrees(angle_axis) / dt
            
            # Check for gimbal lock and handle large angular velocities
            for j in range(3):
                if abs(gripper_angular_vel[i, j]) > 180/dt:  # If angular velocity is too large
                    # Use alternative calculation for this axis
                    angle_diff = gripper_orientations[i+1 if i == 0 else i-1, j] - gripper_orientations[i, j]
                    angle_diff = np.mod(angle_diff + 180, 360) - 180
                    gripper_angular_vel[i, j] = angle_diff / dt

        # Create figure for gripper states
        fig_gripper = plt.figure(figsize=(COL_WIDTH * 3, ROW_HEIGHT * 2))
        fig_gripper.suptitle('Gripper State Trajectory', fontsize=12)
        
        # Plot gripper states
        # First row: Position and velocity
        for i in range(3):  # x, y, z positions and velocities
            ax = plt.subplot(2, 3, i + 1)
            # Add background color for grasping phase only if joint_num > 0
            if joint_num > 0:
                ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
            ax.plot(time_state, gripper_positions[:, i], 'b-', label='Position')
            ax.plot(time_state, gripper_linear_vel[:, i], 'r--', label='Velocity')
            
            # Add markers for object pose
            ax.plot(grasp_time, object_pose[i], 'ro', label='Object Pose', markersize=6)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Gripper {state_labels[i]}')
            ax.grid(True)
            ax.legend(fontsize='small', loc='best')
            ax.set_title(f'Gripper {state_labels[i]}')
        
        # Second row: Orientation and angular velocity
        for i in range(3):  # roll, pitch, yaw angles and angular velocities
            ax = plt.subplot(2, 3, i + 4)
            
            # Add background color for grasping phase only if joint_num > 0
            if joint_num > 0:
                ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
            
            # Get angles and make them continuous
            angles = gripper_orientations[:, i].copy()
            
            # Make angles continuous by detecting jumps
            for j in range(1, len(angles)):
                diff = angles[j] - angles[j-1]
                if diff > 180:
                    angles[j:] -= 360
                elif diff < -180:
                    angles[j:] += 360
            
            # Plot continuous angles
            ax.plot(time_state, angles, 'b-', label='Angle')
            
            # Calculate angular velocity using quaternions for better stability
            angular_vel = np.zeros_like(angles)
            
            # Convert angles to quaternions
            angles_reshaped = np.zeros((len(angles), 3))  # Create array of shape (n, 3)
            angles_reshaped[:, i] = angles  # Fill the corresponding axis column
            quats = R.from_euler('xyz', angles_reshaped, degrees=True).as_quat()
            
            # Calculate angular velocity using quaternion differences
            for j in range(1, len(angles)-1):
                # Get consecutive quaternions
                q1 = quats[j-1]
                q2 = quats[j+1]
                
                # Calculate relative rotation
                q_diff = R.from_quat(q2) * R.from_quat(q1).inv()
                angle_axis = q_diff.as_rotvec()
                
                # Convert to angular velocity (in degrees/s)
                angular_vel[j] = np.degrees(angle_axis[i]) / (2 * dt)
                
                # Limit angular velocity to reasonable range
                if abs(angular_vel[j]) > 180/dt:
                    # If too large, use simple difference with angle wrapping
                    angle_diff = angles[j+1] - angles[j-1]
                    angle_diff = np.mod(angle_diff + 180, 360) - 180
                    angular_vel[j] = angle_diff / (2 * dt)
            
            # Handle endpoints
            # First point
            q1 = quats[0]
            q2 = quats[1]
            q_diff = R.from_quat(q2) * R.from_quat(q1).inv()
            angle_axis = q_diff.as_rotvec()
            angular_vel[0] = np.degrees(angle_axis[i]) / dt
            if abs(angular_vel[0]) > 180/dt:
                angle_diff = angles[1] - angles[0]
                angle_diff = np.mod(angle_diff + 180, 360) - 180
                angular_vel[0] = angle_diff / dt
                
            # Last point
            q1 = quats[-2]
            q2 = quats[-1]
            q_diff = R.from_quat(q2) * R.from_quat(q1).inv()
            angle_axis = q_diff.as_rotvec()
            angular_vel[-1] = np.degrees(angle_axis[i]) / dt
            if abs(angular_vel[-1]) > 180/dt:
                angle_diff = angles[-1] - angles[-2]
                angle_diff = np.mod(angle_diff + 180, 360) - 180
                angular_vel[-1] = angle_diff / dt
            
            # Apply additional smoothing to angular velocity
            window_size = 3
            angular_vel = np.convolve(angular_vel, np.ones(window_size)/window_size, mode='same')
            
            # Limit angular velocity to reasonable range (e.g., ±180 degrees/s)
            max_angular_vel = 180.0  # degrees per second
            angular_vel = np.clip(angular_vel, -max_angular_vel, max_angular_vel)
            
            ax.plot(time_state, angular_vel, 'r--', label='Angular Velocity')
            
            # Add markers for gripper orientation at key points
            ax.plot(grasp_time, 0, 'ro', label='Grasp Point', markersize=6)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Gripper {state_labels[i+3]}')
            ax.grid(True)
            ax.legend(fontsize='small', loc='best')
            ax.set_title(f'Gripper {state_labels[i+3]}')
        
        plt.tight_layout()
    
    # --------------------------------3. plot  drone states--------------------------------
    state_ref_world_frame = traj_state_ref.copy()
    state_num = trajectory_obj.robot_model.nv
    
    # Calculate subplot layout based on joint number
    n_cols = max(3, joint_num)  # Maximum 3 columns
    n_rows = (joint_num + n_cols - 1) // n_cols  # Ceiling division
    total_rows = 2 + n_rows  # 2 rows for drone states + n_rows for joint states
    
    # Create figure for drone and joint states
    fig_drone = plt.figure(figsize=(COL_WIDTH * 3, ROW_HEIGHT * total_rows))
    fig_drone.suptitle('Drone and Joint States', fontsize=12)
    
    # Plot drone states
    for i in range(6):  # Plot position and orientation states
        ax = plt.subplot(total_rows, 3, i + 1)
        # Add background color for grasping phase only if joint_num > 0
        if joint_num > 0:
            ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
        state_data = [s[i] for s in traj_state_ref[:n_points_state]]
        vel_data = [s[i+state_num] for s in traj_state_ref[:n_points_state]]
        vel_world_data = [s[i+state_num] for s in state_ref_world_frame[:n_points_state]]
        
        # Convert roll, pitch, yaw from radians to degrees
        if i in [3, 4, 5]:  # Indices for roll, pitch, yaw
            state_data = np.degrees(state_data)
            vel_data = np.degrees(vel_data)
        
        # Plot main trajectory
        ax.plot(time_state, state_data, 'b-', label='Position')
        ax.plot(time_state, vel_data, 'r--', label='Velocity (body)')
        
        # Add markers for key points
        if i < 3:  # Position states
            ax.plot(time_state[0], start_pose_drone[i], 'go', label='Initial Point', markersize=6)
            ax.plot(time_state[-1], final_pose_drone[i], 'mo', label='Final Point', markersize=6)
            if grasp_pose_drone is not None:
                ax.plot(time_state[len(time_state)//2], grasp_pose_drone[i], 'ro', label='Grasp Point', markersize=6)
        elif i < 6:  # Orientation states
            ax.plot(time_state[0], 0, 'go', label='Initial Point', markersize=6)
            ax.plot(time_state[-1], 0, 'mo', label='Final Point', markersize=6)
            if grasp_pose_drone is not None:
                ax.plot(time_state[len(time_state)//2], 0, 'ro', label='Grasp Point', markersize=6)
        
        if i in [0, 1, 2]:  # Position states
            ax.plot(time_state, vel_world_data, 'g--', label='Velocity (world)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_labels[i])
        ax.grid(True)
        ax.legend(fontsize='small', loc='best')
        ax.set_title(state_labels[i])
    
    # Plot joint states
    for i in range(joint_num):
        ax = plt.subplot(total_rows, 3, 6 + i + 1)
        # Add background color for grasping phase only if joint_num > 0
        if joint_num > 0:
            ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
        joint_angle_data = [s[i+6] for s in traj_state_ref[:n_points_state]]  # Joint angles start at index 6
        joint_vel_data = [s[i+state_num+6] for s in traj_state_ref[:n_points_state]]  # Joint velocities start at state_num+7
        
        # Convert to degrees
        joint_angle_data = np.degrees(joint_angle_data)
        joint_vel_data = np.degrees(joint_vel_data)
        
        # Plot joint data
        ax.plot(time_state, joint_angle_data, 'b-', label='Joint Angle')
        ax.plot(time_state, joint_vel_data, 'r--', label='Joint Velocity')
        
        # Add markers for key points
        ax.plot(time_state[0], np.degrees(start_joint_angle[i]), 'go', label='Initial Point', markersize=6)
        ax.plot(time_state[-1], np.degrees(final_joint_angle[i]), 'mo', label='Final Point', markersize=6)
        if grasp_joint_angle is not None:
            ax.plot(grasp_time, np.degrees(grasp_joint_angle[i]), 'ro', label='Grasp Point', markersize=6)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Joint {i+1} (deg)')
        ax.grid(True)
        ax.legend(fontsize='small', loc='best')
        ax.set_title(f'Joint {i+1} State')
    
    plt.tight_layout()
    
    # --------------------------------4. plot  control inputs--------------------------------
    # Control inputs have 2 rows and 2 columns
    fig_controls = plt.figure(figsize=(COL_WIDTH * 2, ROW_HEIGHT * 2))
    fig_controls.suptitle('Control Inputs', fontsize=12)
    
    # Plot rotor thrusts (first subplot)
    ax = plt.subplot(2, 2, 1)
    # Add background color for grasping phase only if joint_num > 0
    if joint_num > 0:
        ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
    for i in range(n_rotors):  # Plot all rotor thrusts
        rotor_data = [u[i] for u in trajectory.us_squash[:n_points_control]]
        ax.plot(time_control, rotor_data, label=f'Rotor {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Rotor Thrusts (N)')
    
    # Plot joint controls (second subplot)
    ax = plt.subplot(2, 2, 2)
    # Add background color for grasping phase only if joint_num > 0
    if joint_num > 0:
        ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
    for i in range(joint_num):  # Plot all joint controls
        joint_data = [u[i+n_rotors] for u in trajectory.us_squash[:n_points_control]]
        ax.plot(time_control, joint_data, label=f'Joint {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Control')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Joint Controls (Nm)')
    
    # Plot all force components in one subplot
    ax = plt.subplot(2, 2, 3)
    # Add background color for grasping phase only if joint_num > 0
    if joint_num > 0:
        ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
    force_labels = ['F_x', 'F_y', 'F_z']
    for i in range(3):
        force_data = [u[i] for u in control_force_torque[:n_points_control]]
        ax.plot(time_control, force_data, label=force_labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Collective Force (body frame)')
    
    # Plot all torque components in one subplot
    ax = plt.subplot(2, 2, 4)
    # Add background color for grasping phase only if joint_num > 0
    if joint_num > 0:
        ax.axvspan(grasp_time, grasp_time + grasp_duration, color='yellow', alpha=0.2, label='Grasping Phase')
    torque_labels = ['τ_x', 'τ_y', 'τ_z']
    for i in range(3):
        torque_data = [u[i+3] for u in control_force_torque[:n_points_control]]
        ax.plot(time_control, torque_data, label=torque_labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Torque (body frame)')
    
    plt.tight_layout()
    
    # --------------------------------5. save plots--------------------------------
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_controls.savefig(save_dir / 'control_inputs.png', dpi=300, bbox_inches='tight')
        fig_drone.savefig(save_dir / 'drone_trajectory.png', dpi=300, bbox_inches='tight')
        fig_gripper.savefig(save_dir / 'gripper_trajectory.png', dpi=300, bbox_inches='tight')
        
        # Save trajectory data
        np.save(save_dir / 'traj_state_ref.npy', traj_state_ref[:n_points_state])
        np.save(save_dir / 'control_force_torque.npy', control_force_torque[:n_points_control])
        np.save(save_dir / 'time_control.npy', time_control)
        np.save(save_dir / 'time_state.npy', time_state)
        np.save(save_dir / 'gripper_positions.npy', gripper_positions)
        np.save(save_dir / 'gripper_orientations.npy', gripper_orientations)
    
    plt.show(block=False)
    plt.pause(0.1)  # Give a small pause to ensure plots are displayed

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trajectory planning for drone control')
    parser.add_argument('--robot', type=str, default='s500_uam',
                      choices=['s500', 's500_uam', 'hexacopter370_flying_arm_3'],
                      help='Robot model to use')
    parser.add_argument('--trajectory', type=str, default='catch_vicon_real',
                      help='Trajectory name')
    parser.add_argument('--dt', type=int, default=50,
                      help='Time step for trajectory optimization (ms)')
    parser.add_argument('--use-squash', action='store_true', default=True,
                      help='Use squash function for control inputs')
    parser.add_argument('--gepetto-vis', action='store_true', default=True,
                      help='Enable Gepetto visualization')
    parser.add_argument('--save', action='store_true', default=False,
                      help='Save results to file')
    parser.add_argument('--config-path', type=str, default='config/yaml',
                      help='Path to MPC configuration files')
    
    args = parser.parse_args()
    
    # Settings
    mpc_yaml_path = args.config_path
    
    robot_name = args.robot
    trajectory_name = args.trajectory
    dt_traj_opt = args.dt
    useSquash = args.use_squash
    
    gepetto_vis = args.gepetto_vis
    
    save_file = args.save
    save_dir = None
    
    root = tk.Tk()
    screen_dpi = root.winfo_fpixels('1i')
    root.destroy()
    plt.rcParams['figure.dpi'] = screen_dpi
    
    task_name = robot_name + '_' + trajectory_name
    print(f"Running trajectory optimization for task: {task_name}")
    print(f"Parameters:")
    print(f"  dt_traj_opt: {dt_traj_opt} ms")
    print(f"  useSquash: {useSquash}")
    print(f"  gepetto_vis: {gepetto_vis}")
    print(f"  save_file: {save_file}")
    print(f"  config_path: {mpc_yaml_path}")
    
    # Run trajectory optimization
    trajectory, traj_state_ref, _, trajectory_obj = get_opt_traj(
        robot_name,
        trajectory_name, 
        dt_traj_opt, 
        useSquash,
        mpc_yaml_path
    )
    
    # Get tau_f from MPC yaml file
    tau_f = trajectory_obj.platform_params.tau_f
    
    # Convert control plan to force/torque
    control_plan_rotor = np.array(trajectory.us_squash)
    control_force_torque = thrustToForceTorqueAll_array(
        control_plan_rotor, 
        tau_f
    )
    
    # Transfer traj_state_ref to state_array
    state_array = np.array(traj_state_ref)
    
    # get task name from config
    if save_file:
        # save state_array to file
        file_path = Path(__file__).resolve()
        dir_path = file_path.parent
        
        save_dir = str(dir_path) + '/results/' + task_name + '/'
        # create save_dir if not exist
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(save_dir + 'state_array.npy', state_array)
        
        # save control plan and control force torque to file
        np.save(save_dir + 'control_plan.npy', control_plan_rotor)
        np.save(save_dir + 'control_force_torque.npy', control_force_torque)
    
    # transfer quaternion to euler angle
    quat = state_array[:, 3:7]
    rotation = R.from_quat(quat)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    state_array_new = np.hstack((state_array[:,:3], euler_angles, state_array[:,7:]))
    
    # Plot results
    plot_trajectory(
        trajectory,
        trajectory_obj,
        state_array_new,
        control_force_torque,
        dt_traj_opt,
        state_array,
        save_dir
    )
    
    if gepetto_vis:
        # Check if gepetto-gui is running
        try:
            gepetto.corbaserver.Client()
        except Exception as e:
            print("\nError: gepetto-gui is not running!")
            print("Please start gepetto-gui first by running 'gepetto-gui' in a terminal.")
            print("Then run this program again.")
            plt.show(block=True)  # Keep plots open after program ends
            return

        if robot_name == 's500_uam':
            robot_name = 's500_uam_simple'
        robot = example_robot_data.load(robot_name)

        rate = -1
        freq = 1

        # Camera position: [x, y, z, qw, qx, qy, qz]
        # Position the camera at (3, -3, 2) looking at the center of the movement range
        # Using quaternion for 45-degree rotation around z-axis and 30-degree around x-axis
        cameraTF = [3, -3, 2, 0.8536, 0.1464, 0.1464, 0.3536]  # todo: looks not working
        
        display = crocoddyl.GepettoDisplay(
            robot, rate, freq, cameraTF, floor=False)
        
        print("\nGepetto visualization is running. Press Ctrl+C to stop.")
        try:
            while True:
                display.displayFromSolver(trajectory)
                time.sleep(1)  
        except KeyboardInterrupt:
            print("\nStopping visualization...")
            plt.show(block=True)  # Keep plots open after program ends

if __name__ == '__main__':
    main() 