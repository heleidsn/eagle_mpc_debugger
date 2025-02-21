import numpy as np
import matplotlib.pyplot as plt
from mpc_l1_control.config import SimulationConfig
from utils.traj_opt import get_opt_traj
from utils.u_convert import thrustToForceTorqueAll_array
import argparse
from pathlib import Path
import yaml
from mpc_l1_control.controllers import create_mpc_from_yaml
from scipy.spatial.transform import Rotation as R

def plot_trajectory(trajectory, traj_state_ref, control_force_torque, dt_traj_opt, save_dir=None):
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
    
    # Plot controls
    for i in range(9):
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
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        state_data = [s[i] for s in traj_state_ref[:n_points]]
        vel_data = [s[i+9] for s in traj_state_ref[:n_points]]
        
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
        np.save(save_dir / 'traj_state_ref.npy', traj_state_ref[:n_points])
        np.save(save_dir / 'control_force_torque.npy', control_force_torque[:n_points])
        np.save(save_dir / 'time.npy', time)
    
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trajectory optimization and visualization')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    args = parser.parse_args()
    
    # Load configuration
    config = SimulationConfig(args.config)
    
    print(f"Running trajectory optimization for task: {config.task_name}")
    print(f"Parameters:")
    print(f"  dt_traj_opt: {config.dt_traj_opt} ms")
    print(f"  using_simple_model: {config.using_simple_model}")
    print(f"  using_angle_setpoint: {config.using_angle_setpoint}")
    
    # Run trajectory optimization
    trajectory, traj_state_ref, _, trajectory_obj = get_opt_traj(
        config.task_name,
        config.dt_traj_opt,
        config.using_simple_model,
        config.using_angle_setpoint
    )
    
    # create mpc controller to get tau_f    
    mpc_controller = create_mpc_from_yaml(
        config.mpc_type,
        trajectory_obj,
        traj_state_ref,
        config.dt_traj_opt,
        config.mpc_yaml_path
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
        config.dt_traj_opt,
        args.save_dir
    )

if __name__ == '__main__':
    main() 