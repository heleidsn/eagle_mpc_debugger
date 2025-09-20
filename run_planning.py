'''
Author: Lei He
Date: 2025-02-24 10:31:39
LastEditTime: 2025-09-20 15:58:45
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
import gepetto
import pinocchio as pin

import tkinter as tk

def parse_stage_info_from_trajectory(trajectory_obj, dt_traj_opt):
    """Parse stage information from trajectory object.
    
    Args:
        trajectory_obj: Trajectory object containing stage information
        dt_traj_opt: Time step for trajectory optimization in ms
        
    Returns:
        Dictionary containing stage information with start/end times
    """
    try:
        # Try to read the temporary YAML file that was used for trajectory creation
        import rospkg
        import yaml
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('eagle_mpc_debugger')
        temp_yaml_path = os.path.join(package_path, 'config/yaml/trajectories/temp_trajectory.yaml')
        
        if os.path.exists(temp_yaml_path):
            with open(temp_yaml_path, 'r') as f:
                yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            
            if 'trajectory' in yaml_data and 'stages' in yaml_data['trajectory']:
                stages = yaml_data['trajectory']['stages']
                dt_seconds = dt_traj_opt / 1000.0  # Convert to seconds
                
                stage_info = {'stages': []}
                current_time = 0.0
                
                for stage in stages:
                    stage_name = stage.get('name', 'unknown')
                    duration_ms = stage.get('duration', 0)
                    duration_s = duration_ms / 1000.0
                    
                    # Only add stages with non-zero duration
                    if duration_s > 0:
                        stage_info['stages'].append({
                            'name': stage_name,
                            'start_time': current_time,
                            'end_time': current_time + duration_s,
                            'duration': duration_s
                        })
                        current_time += duration_s
                
                return stage_info
    except Exception as e:
        print(f"Warning: Could not parse stage information: {e}")
    
    return None

def add_stage_backgrounds(ax, time_vector, stage_info, alpha=0.15, show_labels=True, show_legend=True):
    """Add background colors for different trajectory stages.
    
    Args:
        ax: Matplotlib axis object
        time_vector: Time vector for the trajectory
        stage_info: Dictionary containing stage information
        alpha: Transparency of the background colors
        show_labels: Whether to show stage names as text labels
        show_legend: Whether to add legend entries for stages
    """
    if stage_info is None:
        return
    
    # Define colors for different stages
    stage_colors = {
        'approach': 'lightblue',
        'pre_grasp': 'lightgreen', 
        'grasp': 'yellow',
        'after_grasp': 'lightcoral',
        'move_away': 'lightsalmon',
        'hover_after_grasp': 'lightpink',
        'take_off': 'lightsteelblue',
        'nav_wp1': 'lightcyan',
        'wp_1': 'lightgoldenrodyellow',
        'nav_wp2': 'lightseagreen',
        'hover': 'lavender',
        'land': 'lightgray'
    }
    
    # Track added legend entries to avoid duplicates
    added_to_legend = set()
    
    # Add background spans for each stage
    for stage in stage_info['stages']:
        stage_name = stage['name'].lower()
        start_time = stage['start_time']
        end_time = stage['end_time']
        
        # Skip zero-duration stages
        if end_time <= start_time:
            continue
            
        # Get color for the stage (use default if not found)
        color = stage_colors.get(stage_name, 'lightgray')
        
        # Add background span with legend entry if needed
        if show_legend and stage_name not in added_to_legend:
            ax.axvspan(start_time, end_time, color=color, alpha=alpha, 
                      label=f'{stage["name"]}')
            added_to_legend.add(stage_name)
        else:
            ax.axvspan(start_time, end_time, color=color, alpha=alpha)

def add_stage_labels(ax, stage_info):
    """Add text labels for trajectory stages (call after plotting data).
    
    Args:
        ax: Matplotlib axis object
        stage_info: Dictionary containing stage information
    """
    if stage_info is None:
        return
        
    # Add text labels for each stage
    for stage in stage_info['stages']:
        start_time = stage['start_time']
        end_time = stage['end_time']
        
        # Skip zero-duration stages and very short stages
        if end_time <= start_time or (end_time - start_time) < 0.1:
            continue
            
        center_time = (start_time + end_time) / 2
        
        # Get y-axis limits to position text appropriately
        ylim = ax.get_ylim()
        text_y = ylim[1] - 0.08 * (ylim[1] - ylim[0])  # 8% from top
        
        # Add text with background box for better readability
        ax.text(center_time, text_y, stage['name'], 
               ha='center', va='top', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

def create_stage_legend(stage_info):
    """Create a unified legend for stage colors.
    
    Args:
        stage_info: Dictionary containing stage information
        
    Returns:
        List of legend handles and labels
    """
    if stage_info is None:
        return [], []
    
    # Define colors for different stages (same as in add_stage_backgrounds)
    stage_colors = {
        'approach': 'lightblue',
        'pre_grasp': 'lightgreen', 
        'grasp': 'yellow',
        'after_grasp': 'lightcoral',
        'move_away': 'lightsalmon',
        'hover_after_grasp': 'lightpink',
        'take_off': 'lightsteelblue',
        'nav_wp1': 'lightcyan',
        'wp_1': 'lightgoldenrodyellow',
        'nav_wp2': 'lightseagreen',
        'hover': 'lavender',
        'land': 'lightgray'
    }
    
    import matplotlib.patches as mpatches
    
    handles = []
    labels = []
    
    # Track unique stages to avoid duplicates
    added_stages = set()
    
    for stage in stage_info['stages']:
        stage_name = stage['name'].lower()
        
        # Skip zero-duration stages and duplicates
        if stage['end_time'] <= stage['start_time'] or stage_name in added_stages:
            continue
            
        color = stage_colors.get(stage_name, 'lightgray')
        patch = mpatches.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, edgecolor='black', linewidth=0.5)
        
        handles.append(patch)
        labels.append(stage['name'])
        added_stages.add(stage_name)
    
    return handles, labels

def plot_trajectory(trajectory, trajectory_obj, traj_state_ref, control_force_torque, dt_traj_opt, state_array, save_dir=None, is_catch_task=False, catch_config=None, stage_info=None):
    """Plot optimized trajectory results.
    
    Args:
        trajectory: Trajectory object containing optimized data
        traj_state_ref: Reference state trajectory
        control_force_torque: Control inputs in force/torque format
        dt_traj_opt: Time step for trajectory optimization
        state_array: State array for forward kinematics
        save_dir: Directory to save plots (optional)
        is_catch_task: Whether this is a catch task (optional)
        catch_config: Catch task configuration dictionary (optional)
        stage_info: Dictionary containing stage information for background coloring (optional)
    """
    # Create time vector
    n_points_control = len(control_force_torque)
    n_points_state = len(traj_state_ref)
    time_control = np.arange(n_points_control) * dt_traj_opt / 1000  # Convert to seconds
    time_state = np.arange(n_points_state) * dt_traj_opt / 1000  # Convert to seconds
    
    # Parse stage information if not provided
    if stage_info is None:
        stage_info = parse_stage_info_from_trajectory(trajectory_obj, dt_traj_opt)
        if stage_info is not None and len(stage_info['stages']) > 0:
            print(f"✓ Found {len(stage_info['stages'])} trajectory stages for background coloring:")
            for stage in stage_info['stages']:
                print(f"  - {stage['name']}: {stage['start_time']:.2f}s - {stage['end_time']:.2f}s")
        else:
            print("ℹ No stage information found - plots will use default background")
    
    # Get robot model from trajectory
    model = trajectory_obj.robot_model
    data = model.createData()
    
    # Get number of joints
    joint_num = model.nq - 7  # Subtract 7 for base position (3) and orientation (4)
    # Get number of rotors from the model
    n_rotors = trajectory_obj.platform_params.n_rotors
    
    # Define key points from configuration
    if is_catch_task and catch_config is not None:
        # Use parameters from catch_config for catch tasks
        start_pose_drone = catch_config['initial_state'][:3]  # x, y, z from initial state
        start_joint_angle = catch_config['initial_state'][7:9]  # joint angles from initial state
        
        final_pose_drone = catch_config['final_state'][:3]  # x, y, z from final state
        final_joint_angle = catch_config['final_state'][7:9]  # joint angles from final state
        
        # Calculate grasp timing from catch_config
        grasp_time = catch_config['pre_grasp_time'] / 1000.0  # Convert to seconds
        grasp_duration = catch_config['grasp_time'] / 1000.0  # Convert to seconds
        
        # Use target gripper position as object pose approximation
        object_pose = catch_config['target_gripper_pos']
        
        # Set grasp pose based on target position (approximate drone position for grasping)
        # For now, we'll estimate drone position from gripper target
        # This could be improved by calculating actual drone position from inverse kinematics
        grasp_pose_drone = catch_config['target_gripper_pos'].copy()
        grasp_pose_drone[2] += 0.7  # Approximate offset between gripper and drone base
        
        # Estimate grasp joint angle (could be improved with actual IK solution)
        grasp_joint_angle = [-1.2, -0.6]  # Default grasp configuration
    else:
        # Default values for non-catch tasks
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
    
    # --------------------------------1. plot cost curve with stage legend--------------------------------
    fig_cost = plt.figure(figsize=(COL_WIDTH * 1.5, ROW_HEIGHT))
    
    # Create subplot for cost curve
    ax_cost = plt.subplot(1, 2, 1)
    ax_cost.plot(trajectory_obj.logger.costs, marker='o')
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('Total Cost')
    ax_cost.set_title('Cost Convergence during DDP Optimization')
    ax_cost.grid(True)
    
    # Create subplot for stage legend
    ax_legend = plt.subplot(1, 2, 2)
    ax_legend.axis('off')  # Hide axes
    
    # Create and display stage legend
    if stage_info is not None:
        handles, labels = create_stage_legend(stage_info)
        if handles:
            ax_legend.legend(handles, labels, loc='center', title='Trajectory Stages', 
                           title_fontsize=10, fontsize=9, frameon=True, 
                           fancybox=True, shadow=True)
            ax_legend.set_title('Stage Colors Legend', fontsize=11, pad=20)
    
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
            # Add background colors for different stages
            add_stage_backgrounds(ax, time_state, stage_info, show_labels=False, show_legend=False)
            ax.plot(time_state, gripper_positions[:, i], 'b-', label='Position')
            ax.plot(time_state, gripper_linear_vel[:, i], 'r--', label='Velocity')
            
            # Add markers for object pose only for catch tasks
            if is_catch_task:
                ax.plot(grasp_time, object_pose[i], 'ro', label='Object Pose', markersize=6)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Gripper {state_labels[i]}')
            ax.grid(True)
            ax.legend(fontsize='small', loc='best')
            ax.set_title(f'Gripper {state_labels[i]}')
            # Add stage labels after plotting data
            add_stage_labels(ax, stage_info)
        
        # Second row: Orientation and angular velocity
        for i in range(3):  # roll, pitch, yaw angles and angular velocities
            ax = plt.subplot(2, 3, i + 4)
            
            # Add background colors for different stages
            add_stage_backgrounds(ax, time_state, stage_info, show_labels=False, show_legend=False)
            
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
            
            # Add markers for gripper orientation at key points only for catch tasks
            if is_catch_task:
                ax.plot(grasp_time, 0, 'ro', label='Grasp Point', markersize=6)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Gripper {state_labels[i+3]}')
            ax.grid(True)
            ax.legend(fontsize='small', loc='best')
            ax.set_title(f'Gripper {state_labels[i+3]}')
            # Add stage labels after plotting data
            add_stage_labels(ax, stage_info)
        
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
        # Add background colors for different stages
        add_stage_backgrounds(ax, time_state, stage_info)
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
        if is_catch_task:
            if i < 3:  # Position states
                ax.plot(time_state[0], start_pose_drone[i], 'go', label='Initial Point', markersize=6)
                ax.plot(time_state[-1], final_pose_drone[i], 'mo', label='Final Point', markersize=6)
                
            elif i < 6:  # Orientation states
                ax.plot(time_state[0], 0, 'go', label='Initial Point', markersize=6)
                ax.plot(time_state[-1], 0, 'mo', label='Final Point', markersize=6)

        
        if i in [0, 1, 2]:  # Position states
            ax.plot(time_state, vel_world_data, 'g--', label='Velocity (world)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_labels[i])
        ax.grid(True)
        ax.legend(fontsize='small', loc='best')
        ax.set_title(state_labels[i])
        # Add stage labels after plotting data
        add_stage_labels(ax, stage_info)
    
    # Plot joint states
    for i in range(joint_num):
        ax = plt.subplot(total_rows, 3, 6 + i + 1)
        # Add background colors for different stages
        add_stage_backgrounds(ax, time_state, stage_info)
        joint_angle_data = [s[i+6] for s in traj_state_ref[:n_points_state]]  # Joint angles start at index 6
        joint_vel_data = [s[i+state_num+6] for s in traj_state_ref[:n_points_state]]  # Joint velocities start at state_num+7
        
        # Convert to degrees
        joint_angle_data = np.degrees(joint_angle_data)
        joint_vel_data = np.degrees(joint_vel_data)
        
        # Plot joint data
        ax.plot(time_state, joint_angle_data, 'b-', label='Joint Angle')
        ax.plot(time_state, joint_vel_data, 'r--', label='Joint Velocity')
        
        # Add markers for key points
        if is_catch_task:
            ax.plot(time_state[0], np.degrees(start_joint_angle[i]), 'go', label='Initial Point', markersize=6)
            ax.plot(time_state[-1], np.degrees(final_joint_angle[i]), 'mo', label='Final Point', markersize=6)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Joint {i+1} (deg)')
        ax.grid(True)
        ax.legend(fontsize='small', loc='best')
        ax.set_title(f'Joint {i+1} State')
        # Add stage labels after plotting data
        add_stage_labels(ax, stage_info)
    
    plt.tight_layout()
    
    # --------------------------------4. plot  control inputs--------------------------------
    # Control inputs have 2 rows and 2 columns
    fig_controls = plt.figure(figsize=(COL_WIDTH * 2, ROW_HEIGHT * 2))
    fig_controls.suptitle('Control Inputs', fontsize=12)
    
    # Plot rotor thrusts (first subplot)
    ax = plt.subplot(2, 2, 1)
    # Add background colors for different stages
    add_stage_backgrounds(ax, time_control, stage_info, show_labels=False, show_legend=False)
    for i in range(n_rotors):  # Plot all rotor thrusts
        rotor_data = [u[i] for u in trajectory.us_squash[:n_points_control]]
        ax.plot(time_control, rotor_data, label=f'Rotor {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Rotor Thrusts (N)')
    # Add stage labels after plotting data
    add_stage_labels(ax, stage_info)
    
    # Plot joint controls (second subplot)
    ax = plt.subplot(2, 2, 2)
    # Add background colors for different stages
    add_stage_backgrounds(ax, time_control, stage_info, show_labels=False, show_legend=False)
    for i in range(joint_num):  # Plot all joint controls
        joint_data = [u[i+n_rotors] for u in trajectory.us_squash[:n_points_control]]
        ax.plot(time_control, joint_data, label=f'Joint {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Control')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Joint Controls (Nm)')
    # Add stage labels after plotting data
    add_stage_labels(ax, stage_info)
    
    # Plot all force components in one subplot
    ax = plt.subplot(2, 2, 3)
    # Add background colors for different stages
    add_stage_backgrounds(ax, time_control, stage_info, show_labels=False, show_legend=False)
    force_labels = ['F_x', 'F_y', 'F_z']
    for i in range(3):
        force_data = [u[i] for u in control_force_torque[:n_points_control]]
        ax.plot(time_control, force_data, label=force_labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Collective Force (body frame)')
    # Add stage labels after plotting data
    add_stage_labels(ax, stage_info)
    
    # Plot all torque components in one subplot
    ax = plt.subplot(2, 2, 4)
    # Add background colors for different stages
    add_stage_backgrounds(ax, time_control, stage_info, show_labels=False, show_legend=False)
    torque_labels = ['τ_x', 'τ_y', 'τ_z']
    for i in range(3):
        torque_data = [u[i+3] for u in control_force_torque[:n_points_control]]
        ax.plot(time_control, torque_data, label=torque_labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.grid(True)
    ax.legend(fontsize='small', loc='best')
    ax.set_title('Torque (body frame)')
    # Add stage labels after plotting data
    add_stage_labels(ax, stage_info)
    
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
    parser.add_argument('--trajectory', type=str, default='hover',   # hover, catch_vicon
                      help='Trajectory name')
    parser.add_argument('--dt', type=int, default=20,
                      help='Time step for trajectory optimization (ms)')
    parser.add_argument('--use-squash', action='store_true', default=True,
                      help='Use squash function for control inputs')
    parser.add_argument('--gepetto-vis', action='store_true', default=True,
                      help='Enable Gepetto visualization')
    parser.add_argument('--save', action='store_true', default=False,
                      help='Save results to file')
    parser.add_argument('--config-path', type=str, default='config/yaml',
                      help='Path to MPC configuration files')
    
    # Catch task specific parameters
    parser.add_argument('--catch-initial-state', type=float, nargs=17,
                      default=[-1.5, 0, 1.2, 0, 0, 0, 1, -1.2, -0.6, 0, 0, 0, 0, 0, 0, 0, 0],
                      help='Initial state for catch task [x,y,z,qx,qy,qz,qw,j1,j2,vx,vy,vz,wx,wy,wz,vj1,vj2]')
    parser.add_argument('--catch-target-gripper-pos', type=float, nargs=3,
                      default=[0.0, 0, 0.8],
                      help='Target gripper position for catch task [x,y,z]')
    parser.add_argument('--catch-target-gripper-orient', type=float, nargs=4,
                      default=[0, 0, 0, 1],
                      help='Target gripper orientation for catch task [qx,qy,qz,qw]')
    parser.add_argument('--catch-final-state', type=float, nargs=17,
                      default=[1.5, 0, 1.2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      help='Final state for catch task [x,y,z,qx,qy,qz,qw,j1,j2,vx,vy,vz,wx,wy,wz,vj1,vj2]')
    parser.add_argument('--catch-pre-grasp-time', type=int, default=1500,
                      help='Pre-grasp time duration (ms)')
    parser.add_argument('--catch-grasp-time', type=int, default=200,
                      help='Grasp time duration (ms)')
    parser.add_argument('--catch-post-grasp-time', type=int, default=1500,
                      help='Post-grasp time duration (ms)')
    parser.add_argument('--catch-gripper-pitch-angle', type=float, default=0.0,
                      help='Gripper pitch angle for catch task (degrees)')
    
    args = parser.parse_args()
    
    # Settings
    mpc_yaml_path = args.config_path
    
    robot_name = args.robot
    trajectory_name = args.trajectory
    dt_traj_opt = args.dt
    useSquash = args.use_squash
    
    # Check if this is a catch task
    is_catch_task = False
    is_catch_task = 'catch' in trajectory_name.lower()
    is_catch_task = False
    
    # Process gripper pitch angle to update orientation if provided
    gripper_pitch_angle = args.catch_gripper_pitch_angle
    if gripper_pitch_angle != 0.0:
        # Convert degrees to radians and calculate quaternion for pitch rotation
        import math
        pitch_radians = math.radians(gripper_pitch_angle)
        qx = 0.0
        qy = math.sin(pitch_radians / 2.0)
        qz = 0.0
        qw = math.cos(pitch_radians / 2.0)
        updated_gripper_orient = [qx, qy, qz, qw]
    else:
        updated_gripper_orient = args.catch_target_gripper_orient
    
    # Catch task configuration
    catch_config = {
        'initial_state': args.catch_initial_state,
        'target_gripper_pos': args.catch_target_gripper_pos,
        'target_gripper_orient': updated_gripper_orient,  # Use updated orientation
        'gripper_pitch_angle': gripper_pitch_angle,  # Store original pitch angle
        'final_state': args.catch_final_state,
        'pre_grasp_time': args.catch_pre_grasp_time,
        'grasp_time': args.catch_grasp_time,
        'post_grasp_time': args.catch_post_grasp_time
    }
    
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
    
    # Display catch configuration if it's a catch task
    if is_catch_task:
        print(f"\nCatch Task Configuration:")
        print(f"  Initial state: {catch_config['initial_state']}")
        print(f"  Target gripper position: {catch_config['target_gripper_pos']}")
        print(f"  Target gripper orientation: {catch_config['target_gripper_orient']}")
        print(f"  Gripper pitch angle: {catch_config['gripper_pitch_angle']} degrees")
        print(f"  Final state: {catch_config['final_state']}")
        print(f"  Pre-grasp time: {catch_config['pre_grasp_time']} ms")
        print(f"  Grasp time: {catch_config['grasp_time']} ms")
        print(f"  Post-grasp time: {catch_config['post_grasp_time']} ms")
    
    # Run trajectory optimization
    trajectory, traj_state_ref, _, trajectory_obj = get_opt_traj(
        robot_name,
        trajectory_name, 
        dt_traj_opt, 
        useSquash,
        mpc_yaml_path,
        catch_config if is_catch_task else None
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
        save_dir,
        is_catch_task,
        catch_config,
        None  # stage_info will be parsed automatically
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