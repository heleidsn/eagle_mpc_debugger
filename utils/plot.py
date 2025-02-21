import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.u_convert import thrustToForceTorqueAll_array
from scipy.spatial.transform import Rotation as R

def plot_trajectory(traj_solver, trajectory_obj, traj_state_ref_original, dt_traj_opt, save_dir=None):
    """Plot optimized trajectory results.
    
    Args:
        trajectory: Trajectory object containing optimized data
        traj_state_ref: Reference state trajectory
        control_force_torque: Control inputs in force/torque format
        dt_traj_opt: Time step for trajectory optimization
        save_dir: Directory to save plots (optional)
    """
    
    # Get tau_f from MPC yaml file
    tau_f = trajectory_obj.platform_params.tau_f
    
    # Convert control plan to force/torque
    control_plan_rotor = np.array(traj_solver.us_squash)
    control_force_torque = thrustToForceTorqueAll_array(
        control_plan_rotor, 
        tau_f
    )
    
    # Transfer traj_state_ref to state_array
    state_array = np.array(traj_state_ref_original)
    
    # transfer quaternion to euler angle
    # state_array[:, 3:7] = quaternion_to_euler(state_array[:, 3:7])
    quat = state_array[:, 3:7]
    rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
    euler_angles = rotation.as_euler('xyz', degrees=False)  # 将四元数转换为欧拉角
    
    state_array_new = np.hstack((state_array[:,:3], euler_angles, state_array[:,7:]))
    
    # Create time vector
    n_points = min(len(traj_state_ref_original), len(control_force_torque))
    time = np.arange(n_points) * dt_traj_opt / 1000  # Convert to seconds
    
    # Control labels
    rotor_labels = ['rotor1', 'rotor2', 'rotor3', 'rotor4']
    control_labels = ['F_x (N)', 'F_y (N)', 'F_z (N)',
                     'τ_x (Nm)', 'τ_y (Nm)', 'τ_z (Nm)',
                     'τ_1 (Nm)', 'τ_2 (Nm)', 'τ_3 (Nm)']
    
    # State labels (convert radians to degrees for roll, pitch, yaw)
    state_labels = ['x (m)', 'y (m)', 'z (m)', 
                   'roll (deg)', 'pitch (deg)', 'yaw (deg)',
                   'joint1 (deg)', 'joint2 (deg)', 'joint3 (deg)']
    
    
    # get control and state number
    rotor_num = len(control_plan_rotor[0])
    control_num = len(control_force_torque[0])
    state_num = int(len(state_array_new[0])/2)
    
    if control_num == 6:
        figure_size = (20, 8)
        plot_control_row = 2
        plot_control_col = 3
        plot_state_row = 2  
        plot_state_col = 3
    elif control_num == 9:
        figure_size = (20, 12)
        plot_control_row = 3
        plot_control_col = 3
        plot_state_row = 3
        plot_state_col = 3
    else:
        raise ValueError("Invalid control number")
    
    
    # plot control
    plt.figure()  # 创建一个新的图形
    for i in range(rotor_num):
        control_data = [u[i] for u in control_plan_rotor[:n_points]]
        plt.plot(time, control_data, label=rotor_labels[i])  # 在同一图上绘制所有曲线
    plt.xlabel('Time (s)')
    plt.ylabel('Control')
    plt.grid(True)
    plt.title('Control plan rotors')
    plt.legend()  # 添加图例
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    # plot control
    fig_controls = plt.figure(figsize=figure_size)
    fig_controls.suptitle('Control Inputs', fontsize=16)
    
    # Plot controls
    for i in range(control_num):
        ax = plt.subplot(plot_control_row, plot_control_col, i + 1)
        control_data = [u[i] for u in control_force_torque[:n_points]]
        ax.plot(time, control_data, 'g-', label='Control')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_labels[i])
        ax.grid(True)
        ax.set_title(control_labels[i])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create figure for states
    fig_states = plt.figure(figsize=figure_size)
    fig_states.suptitle('State Trajectory', fontsize=16)
    
    # Plot states
    for i in range(state_num):
        ax = plt.subplot(plot_state_row, plot_state_col, i + 1)
        state_data = [s[i] for s in state_array_new[:n_points]]
        vel_data = [s[i+state_num] for s in state_array_new[:n_points]]
        
        # Convert roll, pitch, yaw from radians to degrees
        if i in [3, 4, 5]:  # Indices for roll, pitch, yaw
            state_data = np.degrees(state_data)
            vel_data = np.degrees(vel_data)
        elif i in [6, 7, 8]:  # Indices for joint1, joint2, joint3
            state_data = np.degrees(state_data)  # Convert joint angles to degrees
            vel_data = np.degrees(vel_data)      # Convert joint velocities to degrees
        
        ax.plot(time, state_data, 'b-', label='Position')
        ax.plot(time, vel_data, 'r--', label='Velocity')
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
        np.save(save_dir / 'traj_state_ref.npy', state_array_new[:n_points])
        np.save(save_dir / 'control_force_torque.npy', control_force_torque[:n_points])
        np.save(save_dir / 'time.npy', time)
    
    plt.show()