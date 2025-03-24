'''
Author: Lei He
Date: 2025-02-24 10:31:39
LastEditTime: 2025-03-15 13:57:45
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
    n_points = min(len(traj_state_ref), len(control_force_torque))
    time = np.arange(n_points) * dt_traj_opt / 1000  # Convert to seconds
    
    # Control labels
    control_labels = ['F_x (N)', 'F_y (N)', 'F_z (N)',
                     'τ_x (Nm)', 'τ_y (Nm)', 'τ_z (Nm)',
                     'τ_1 (Nm)', 'τ_2 (Nm)', 'τ_3 (Nm)']
    
    # State labels (convert radians to degrees for roll, pitch, yaw)
    state_labels = ['x (m)', 'y (m)', 'z (m)', 
                   'roll (deg)', 'pitch (deg)', 'yaw (deg)',
                   'joint1 (deg)', 'joint2 (deg)', 'joint3 (deg)']
    
    # Create figure for controls
    fig_controls = plt.figure(figsize=(20, 12))
    fig_controls.suptitle('Control Inputs', fontsize=16)
    
    # state_array = np.array(traj_state_ref)
    
    control_num = control_force_torque.shape[1]
    
    # Plot controls
    for i in range(control_num):
        ax = plt.subplot(3, 3, i + 1)
        control_data = [u[i] for u in control_force_torque[:n_points]]
        ax.plot(time, control_data, 'g-', label='Control')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_labels[i])
        ax.grid(True)
        ax.set_title(control_labels[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create figure for states
    fig_states = plt.figure(figsize=(20, 12))
    fig_states.suptitle('State Trajectory', fontsize=16)
    
    # Plot states
    state_ref_world_frame = state_array.copy()
    # get vel_body and quat from traj_state_ref
    for i in range(len(state_array)):
        vel_body = state_array[i, 7:10]  # [vx, vy, vz]
        quat = state_array[i, 3:7]  # [qx, qy, qz, qw]

        # Convert body velocities to world frame
        R = quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])[:3, :3]
        vel_world = R @ vel_body
        state_ref_world_frame[i, 6:9] = vel_world
    
    for i in range(control_num):
        ax = plt.subplot(3, 3, i + 1)
        state_data = [s[i] for s in traj_state_ref[:n_points]]
        vel_data = [s[i+control_num] for s in traj_state_ref[:n_points]]
        vel_world_data = [s[i+control_num] for s in state_ref_world_frame[:n_points]]
        
        # Convert roll, pitch, yaw from radians to degrees
        if i in [3, 4, 5]:  # Indices for roll, pitch, yaw
            state_data = np.degrees(state_data)
            vel_data = np.degrees(vel_data)
        elif i in [6, 7, 8]:  # Indices for joint1, joint2, joint3
            state_data = np.degrees(state_data)  # Convert joint angles to degrees
            vel_data = np.degrees(vel_data)      # Convert joint velocities to degrees
        
        ax.plot(time, state_data, 'b-', label='Position')
        ax.plot(time, vel_data, 'r--', label='Velocity (body)')
        
        if i in [0, 1, 2]:
            ax.plot(time, vel_world_data, 'g--', label='Velocity (world)')
        
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
        np.save(save_dir / 'traj_state_ref.npy', traj_state_ref[:n_points])
        np.save(save_dir / 'control_force_torque.npy', control_force_torque[:n_points])
        np.save(save_dir / 'time.npy', time)
    
    plt.show()

def main():
    
    # Settings
    mpc_name = 'rail'
    mpc_yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_ros/eagle_mpc_yaml'
    
    robot_name = 's500'
    trajectory_name = 'displacement'
    dt_traj_opt = 5  # ms
    useSquash = True
    
    save_file = False
    save_dir = None
    
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
        trajectory,
        state_array_new,
        control_force_torque,
        dt_traj_opt,
        state_array,
        save_dir
    )

if __name__ == '__main__':
    main() 