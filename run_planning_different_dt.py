'''
Author: Lei He
Date: 2025-02-24 10:31:39
LastEditTime: 2025-04-18 15:46:07
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
    # n_points = min(len(traj_state_ref), len(control_force_torque))
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
    
    # state_array = np.array(traj_state_ref)
    
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
        # np.save(save_dir / 'time.npy', time)
    
    plt.show()

def plot_multiple_trajectories(trajectories, dt_values, save_dir=None):
    """Plot multiple trajectories for comparison.
    
    Args:
        trajectories: List of trajectory objects containing optimized data
        dt_values: List of time steps for trajectory optimization
        save_dir: Directory to save plots (optional)
    """
    # Create figure for states
    fig_states = plt.figure(figsize=(16, 10), dpi=150)
    fig_states.suptitle('State Trajectories Comparison', fontsize=16)
    
    # State labels (convert radians to degrees for roll, pitch, yaw)
    state_labels = ['x (m)', 'y (m)', 'z (m)', 
                   'roll (deg)', 'pitch (deg)', 'yaw (deg)',
                   'joint1 (deg)', 'joint2 (deg)', 'joint3 (deg)']
    
    # Define key positions and times
    key_positions = [(-1.5, 0, 1.5), (0, 0, 1.2), (1.5, 0, 1.5)]
    key_times = [0, 2, 4]  # seconds
    
    for idx, (trajectory, dt) in enumerate(zip(trajectories, dt_values)):
        traj_state_ref = trajectory[1]
        state_array = np.array(traj_state_ref)
        state_num = trajectory[0].robot_model.nv
        time_state = np.arange(len(traj_state_ref)) * dt / 1000  # Convert to seconds
        
        for i in range(state_num): 
            ax = plt.subplot(3, 3, i + 1)
            state_data = [s[i] for s in traj_state_ref]
            
            # Convert roll, pitch, yaw from radians to degrees
            if i in [3, 4, 5]:  # Indices for roll, pitch, yaw
                state_data = np.degrees(state_data)
            elif i in [6, 7, 8]:  # Indices for joint1, joint2, joint3
                state_data = np.degrees(state_data)  # Convert joint angles to degrees
            
            ax.plot(time_state, state_data, label=f'dt={dt} ms')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(state_labels[i])
            ax.grid(True)
            ax.legend(fontsize='small')
            ax.set_title(state_labels[i])
            
            # Plot key positions as points
            if i in [0, 1, 2]:  # x, y, z positions
                for (pos, t) in zip(key_positions, key_times):
                    ax.plot(t, pos[i], 'o')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots if directory is specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_states.savefig(save_dir / 'state_trajectories_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # Settings
    mpc_name = 'rail'
    mpc_yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml'
    
    robot_name = 's500_uam'   # s500, s500_uam, hexacopter370_flying_arm_3
    trajectory_name = 'catch'
    dt_values = [5, 10, 20, 30]  # Different dt values to compare
    useSquash = True
    
    gepetto_vis = False
    
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
    print(f"  useSquash: {useSquash}")
    
    trajectories = []
    for dt_traj_opt in dt_values:
        print(f"  dt_traj_opt: {dt_traj_opt} ms")
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
        
        # transfer quaternion to euler angle
        quat = state_array[:, 3:7]
        rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
        euler_angles = rotation.as_euler('xyz', degrees=False)  # 将四元数转换为欧拉角
        
        state_array_new = np.hstack((state_array[:,:3], euler_angles, state_array[:,7:]))
        
        trajectories.append((trajectory_obj, state_array_new, control_force_torque, dt_traj_opt, state_array))

    # Plot results
    plot_multiple_trajectories(trajectories, dt_values, save_dir)
    
    if gepetto_vis:
      robot = example_robot_data.load(trajectory_obj.robot_model.name)

      rate = -1
      freq = 1
      cameraTF = [-0.03, 4.4, 2.3, 0, 0.7071, 0, 0.7071]
      
      gepetto.corbaserver.Client()

      display = crocoddyl.GepettoDisplay(
          robot, rate, freq, cameraTF, floor=False)
      
      # display robot with initial state
    #   display.display(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]))

      while True:
          display.displayFromSolver(trajectory)
          time.sleep(1.0)

if __name__ == '__main__':
    main() 