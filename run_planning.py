'''
Author: Lei He
Date: 2025-02-24 10:31:39
LastEditTime: 2025-06-10 11:47:05
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

import example_robot_data
import crocoddyl
import time
import gepetto
import pinocchio as pin

def plot_trajectory(trajectory, traj_state_ref, control_force_torque, dt_traj_opt, state_array,  save_dir=None):
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
    
    # Control labels
    control_labels = ['F_x (N)', 'F_y (N)', 'F_z (N)',
                     'τ_x (Nm)', 'τ_y (Nm)', 'τ_z (Nm)',
                     'τ_1 (Nm)', 'τ_2 (Nm)', 'τ_3 (Nm)']
    
    # State labels (convert radians to degrees for roll, pitch, yaw)
    state_labels = ['x (m)', 'y (m)', 'z (m)', 
                   'roll (deg)', 'pitch (deg)', 'yaw (deg)',
                   'joint1 (deg)', 'joint2 (deg)', 'joint3 (deg)']
    
    # Create figure for controls
    fig_controls = plt.figure(figsize=(16, 10), dpi=150)
    fig_controls.suptitle('Control Inputs', fontsize=16)
    
    control_num = control_force_torque.shape[1]
    state_num = trajectory.robot_model.nv
    
    # Plot controls
    for i in range(control_num):
        ax = plt.subplot(3, 3, i + 1)
        control_data = [u[i] for u in control_force_torque[:n_points_control]]
        ax.plot(time_control, control_data, 'g-', label='Control')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_labels[i])
        ax.grid(True)
        ax.set_title(control_labels[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create figure for states
    fig_states = plt.figure(figsize=(16, 10), dpi=150)
    fig_states.suptitle('State Trajectory', fontsize=16)
    
    # Plot states
    state_ref_world_frame = traj_state_ref.copy()
    # get vel_body and quat from traj_state_ref
    for i in range(len(state_array)):
        vel_body = state_ref_world_frame[i, state_num:state_num+3].copy()  # [vx, vy, vz]
        quat = state_array[i, 3:7]  # [qx, qy, qz, qw]

        # Convert body velocities to world frame
        rot = R.from_quat([quat[0], quat[1], quat[2], quat[3]])  # 四元数转旋转矩阵
        R_mat = rot.as_matrix()  # 3x3 旋转矩阵
        
        vel_world = R_mat @ vel_body
        
        state_ref_world_frame[i, state_num:state_num+3] = vel_world.copy()
    
    # Calculate gripper positions and orientations using pinocchio
    gripper_positions = []
    gripper_orientations = []
    
    # Get robot model from trajectory
    model = trajectory.robot_model
    data = model.createData()
    
    # Print model information
    print("\nRobot model information:")
    print(f"Number of joints (njoints): {model.njoints}")
    print(f"Number of frames (nframes): {model.nframes}")
    print(f"Number of configuration variables (nq): {model.nq}")
    print(f"Number of velocity variables (nv): {model.nv}")
    
    # Print all frame names and their parent joints for debugging
    print("\nDetailed frame information:")
    print("Index | Frame Name | Parent Joint | Parent Frame")
    print("-" * 50)
    for i in range(model.nframes):
        frame = model.frames[i]
        parent_joint = model.names[frame.parentJoint]
        parent_frame = frame.parentFrame if frame.parentFrame < model.nframes else "None"
        print(f"{i:5d} | {frame.name:11s} | {parent_joint:12s} | {parent_frame}")
    
    # Print all joint names
    print("\nAvailable joints in the robot model:")
    for i in range(model.njoints):
        print(f"Joint {i}: {model.names[i]}")
    
    # Get gripper frame ID
    try:
        gripper_frame_id = model.getFrameId("gripper_link")
        print(f"\nFound gripper_link at frame ID: {gripper_frame_id}")
        print(f"Total number of frames: {model.nframes}")
        print(f"Total number of joints: {model.njoints}")
        print(f"data.oMf size: {len(data.oMf)}")
        print(f"data.oMi size: {len(data.oMi)}")
        
        # Validate frame ID
        if gripper_frame_id >= model.nframes:
            print(f"\nError: Invalid frame ID {gripper_frame_id}. Frame IDs must be between 0 and {model.nframes-1}")
            print("Please check the URDF file for the correct gripper link name")
            return
            
        # Print the actual frame name at this ID to verify
        print(f"Frame name at ID {gripper_frame_id}: {model.frames[gripper_frame_id].name}")
        
    except Exception as e:
        print("\nError: Could not find gripper_link in the robot model")
        print("Please check the URDF file for the correct gripper link name")
        print(f"Error details: {str(e)}")
        return
    
    # Define grasping parameters
    grasp_time = 2.0  # Time to start grasping (in seconds)
    grasp_duration = 0.5  # Duration of grasping motion (in seconds)
    gripper_open_angle = 0.0  # Open position (radians)
    gripper_close_angle = 1.57  # Closed position (radians)
    
    for i in range(len(state_array)):
        # Get base position and orientation
        base_pos = state_array[i, :3]
        quat = state_array[i, 3:7]
        rot = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        R_mat = rot.as_matrix()
        
        # Get joint angles
        joint_angles = state_array[i, 7:9]
        
        # Calculate current time
        current_time = i * dt_traj_opt / 1000.0  # Convert to seconds
        
        # Calculate gripper angles based on time
        if current_time < grasp_time:
            # Gripper is open
            gripper_angle = gripper_open_angle
        elif current_time < grasp_time + grasp_duration:
            # Gripper is closing
            t = (current_time - grasp_time) / grasp_duration
            gripper_angle = gripper_open_angle + t * (gripper_close_angle - gripper_open_angle)
        else:
            # Gripper is closed
            gripper_angle = gripper_close_angle
        
        # Update robot configuration
        q = np.zeros(model.nq)
        q[0:3] = base_pos
        q[3:7] = quat
        q[7:9] = joint_angles
        
        # Print configuration for debugging
        if i == 0 or i == len(state_array)-1:  # Print first and last configuration
            print(f"\nConfiguration at step {i}:")
            print(f"Base position: {base_pos}")
            print(f"Base orientation (quaternion): {quat}")
            print(f"Joint angles: {joint_angles}")
            print(f"Gripper angle: {gripper_angle}")
            print(f"Full configuration vector q: {q}")
        
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get gripper pose
        gripper_pose = data.oMf[gripper_frame_id]
        gripper_pos = gripper_pose.translation
        gripper_rot = R.from_matrix(gripper_pose.rotation)
        
        # Get quaternion representation
        gripper_quat = gripper_rot.as_quat()
        
        # Get Euler angles with different sequences
        gripper_euler_xyz = gripper_rot.as_euler('xyz', degrees=True)
        gripper_euler_zyx = gripper_rot.as_euler('zyx', degrees=True)
        
        # Check for gimbal lock (pitch near ±90 degrees)
        pitch_xyz = gripper_euler_xyz[1]
        pitch_zyx = gripper_euler_zyx[1]
        
        # Choose the better representation based on pitch angle
        if abs(pitch_xyz) > 85:  # If pitch is near ±90 degrees
            # Use ZYX sequence which is more stable in this case
            gripper_euler = gripper_euler_zyx
            euler_seq = 'ZYX'
        else:
            # Use XYZ sequence for normal cases
            gripper_euler = gripper_euler_xyz
            euler_seq = 'XYZ'
        
        # Normalize angles to [-180, 180] range
        gripper_euler = np.mod(gripper_euler + 180, 360) - 180
        
        # Print gripper pose for debugging
        if i == 0 or i == len(state_array)-1:  # Print first and last pose
            print(f"\nGripper pose at step {i}:")
            print(f"Position: {gripper_pos}")
            print(f"Orientation (quaternion): {gripper_quat}")
            print(f"Orientation (Euler {euler_seq}): {gripper_euler}")
            print(f"Pitch angle: {gripper_euler[1]} degrees")
            if abs(gripper_euler[1]) > 85:
                print("Warning: Near gimbal lock condition!")
            print(f"Full transformation matrix:\n{gripper_pose}")
        
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
        
        # Angular velocity (in degrees/s)
        angle_diff = gripper_orientations[i+1] - gripper_orientations[i-1]
        # Handle angle wrapping
        angle_diff = np.mod(angle_diff + 180, 360) - 180
        gripper_angular_vel[i] = angle_diff / (2 * dt)
    
    # Handle endpoints using forward/backward difference
    gripper_linear_vel[0] = (gripper_positions[1] - gripper_positions[0]) / dt
    gripper_linear_vel[-1] = (gripper_positions[-1] - gripper_positions[-2]) / dt
    
    angle_diff_start = gripper_orientations[1] - gripper_orientations[0]
    angle_diff_end = gripper_orientations[-1] - gripper_orientations[-2]
    angle_diff_start = np.mod(angle_diff_start + 180, 360) - 180
    angle_diff_end = np.mod(angle_diff_end + 180, 360) - 180
    gripper_angular_vel[0] = angle_diff_start / dt
    gripper_angular_vel[-1] = angle_diff_end / dt
    
    # Print summary of gripper motion
    print("\nGripper motion summary:")
    print(f"Start position: {gripper_positions[0]}")
    print(f"End position: {gripper_positions[-1]}")
    print(f"Position range: {np.ptp(gripper_positions, axis=0)}")  # Peak to peak range
    print(f"Start orientation: {gripper_orientations[0]}")
    print(f"End orientation: {gripper_orientations[-1]}")
    print(f"Orientation range: {np.ptp(gripper_orientations, axis=0)}")  # Peak to peak range
    print(f"Max linear velocity: {np.max(np.abs(gripper_linear_vel), axis=0)} m/s")
    print(f"Max angular velocity: {np.max(np.abs(gripper_angular_vel), axis=0)} deg/s")
    
    # Check for gimbal lock in the trajectory
    pitch_angles = gripper_orientations[:, 1]
    gimbal_lock_steps = np.where(np.abs(pitch_angles) > 85)[0]
    if len(gimbal_lock_steps) > 0:
        print("\nWarning: Gimbal lock detected at steps:", gimbal_lock_steps)
        print("Pitch angles at these steps:", pitch_angles[gimbal_lock_steps])
    
    # Plot states and gripper information
    for i in range(state_num): 
        ax = plt.subplot(3, 3, i + 1)
        state_data = [s[i] for s in traj_state_ref[:n_points_state]]
        vel_data = [s[i+state_num] for s in traj_state_ref[:n_points_state]]
        vel_world_data = [s[i+state_num] for s in state_ref_world_frame[:n_points_state]]
        
        # Convert roll, pitch, yaw from radians to degrees
        if i in [3, 4, 5]:  # Indices for roll, pitch, yaw
            state_data = np.degrees(state_data)
            vel_data = np.degrees(vel_data)
        elif i in [6, 7, 8]:  # Indices for joint1, joint2, joint3
            state_data = np.degrees(state_data)  # Convert joint angles to degrees
            vel_data = np.degrees(vel_data)      # Convert joint velocities to degrees
        
        ax.plot(time_state, state_data, 'b-', label='Position')
        ax.plot(time_state, vel_data, 'r--', label='Velocity (body)')
        
        if i in [0, 1, 2]:
            ax.plot(time_state, vel_world_data, 'g--', label='Velocity (world)')
            # Add gripper position and velocity
            ax.plot(time_state, gripper_positions[:, i], 'm-', label='Gripper Position')
            ax.plot(time_state, gripper_linear_vel[:, i], 'm--', label='Gripper Velocity')
        elif i == 4:  # Only plot pitch (index 4) for gripper orientation
            # Add gripper orientation and angular velocity
            ax.plot(time_state, gripper_orientations[:, 1], 'm-', label='Gripper Pitch')
            ax.plot(time_state, gripper_angular_vel[:, 1], 'm--', label='Gripper Pitch Rate')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_labels[i])
        ax.grid(True)
        ax.legend(fontsize='small')
        ax.set_title(state_labels[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots if directory is specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_controls.savefig(save_dir / 'control_inputs.png', dpi=300, bbox_inches='tight')
        fig_states.savefig(save_dir / 'state_trajectory.png', dpi=300, bbox_inches='tight')
        
        # Save trajectory data
        np.save(save_dir / 'traj_state_ref.npy', traj_state_ref[:n_points_state])
        np.save(save_dir / 'control_force_torque.npy', control_force_torque[:n_points_control])
        np.save(save_dir / 'time_control.npy', time_control)
        np.save(save_dir / 'time_state.npy', time_state)
        np.save(save_dir / 'gripper_positions.npy', gripper_positions)
        np.save(save_dir / 'gripper_orientations.npy', gripper_orientations)
    
    plt.show()

def main():
    
    # Settings
    mpc_name = 'rail'
    mpc_yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_debugger/config/yaml'
    
    robot_name = 's500_uam'   # s500, s500_uam, hexacopter370_flying_arm_3
    trajectory_name = 'catch_vicon'
    dt_traj_opt = 20  # ms
    useSquash = True
    
    gepetto_vis = True   # 
    
    save_file = False
    save_dir = None
    
    # 获取当前屏幕分辨率
    import tkinter as tk
    root = tk.Tk()
    screen_dpi = root.winfo_fpixels('1i')  # 获取屏幕 DPI
    root.destroy()
    plt.rcParams['figure.dpi'] = screen_dpi
    
    task_name = robot_name + '_' + trajectory_name
    print(f"Running trajectory optimization for task: {task_name}")
    print(f"Parameters:")
    print(f"  dt_traj_opt: {dt_traj_opt} ms")
    print(f"  useSquash: {useSquash}")
    
    # Run trajectory optimization
    trajectory, traj_state_ref, _, trajectory_obj = get_opt_traj(
        robot_name,
        trajectory_name, 
        dt_traj_opt, 
        useSquash,
        mpc_yaml_path
    )
    
    # create mpc controller to get tau_f
    mpc_yaml = '{}/mpc/{}_mpc.yaml'.format(mpc_yaml_path, robot_name)    
    mpc_controller = create_mpc_controller(
        mpc_name,
        trajectory_obj,
        traj_state_ref,
        dt_traj_opt,
        mpc_yaml
    )
    
    # Get tau_f from MPC yaml file
    tau_f = mpc_controller.platform_params.tau_f
    
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
    # state_array[:, 3:7] = quaternion_to_euler(state_array[:, 3:7])
    quat = state_array[:, 3:7]
    rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
    euler_angles = rotation.as_euler('xyz', degrees=False)  # 将四元数转换为欧拉角
    
    state_array_new = np.hstack((state_array[:,:3], euler_angles, state_array[:,7:]))
    
    # Plot results
    plot_trajectory(
        trajectory_obj,
        state_array_new,
        control_force_torque,
        dt_traj_opt,
        state_array,
        save_dir
    )
    
    if gepetto_vis:
        if robot_name == 's500_uam':
            robot_name = 's500_uam_simple'
        robot = example_robot_data.load(robot_name)

        rate = -1
        freq = 1
        cameraTF = [-0.03, 4.4, 2.3, 0, 0.7071, 0, 0.7071]
        
        gepetto.corbaserver.Client()

        display = crocoddyl.GepettoDisplay(
            robot, rate, freq, cameraTF, floor=False)
        
        # display robot with initial state
        # initial_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
        # display.display(initial_state)

        while True:
            display.displayFromSolver(trajectory)
            time.sleep(1.0)

if __name__ == '__main__':
    main() 