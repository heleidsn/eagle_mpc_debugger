# run mpc single step

import argparse
import numpy as np
import rospkg
import os
import yaml
import matplotlib.pyplot as plt
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import crocoddyl

# Import eagle_mpc for original controller
try:
    import eagle_mpc
    EAGLE_MPC_AVAILABLE = True
except ImportError:
    EAGLE_MPC_AVAILABLE = False
    print("Warning: eagle_mpc not available")

# Import Crocoddyl MPC controller
try:
    from crocoddyl_mpc_controller import CrocoddylMPCController, PlatformParams, load_mpc_config, load_platform_config
    CROCODDYL_AVAILABLE = True
except ImportError:
    CROCODDYL_AVAILABLE = False
    print("Warning: crocoddyl_mpc_controller not available")


def quaternion_to_euler(quaternions):
    """
    Convert quaternions to Euler angles (roll, pitch, yaw)
    
    Args:
        quaternions: Array of shape (N, 4) with quaternions [qx, qy, qz, qw]
        
    Returns:
        euler_angles: Array of shape (N, 3) with Euler angles [roll, pitch, yaw] in radians
    """
    if quaternions.ndim == 1:
        # Single quaternion
        quat = quaternions[3:7]  # Extract quaternion from state vector
        euler = R.from_quat(quat).as_euler('xyz', degrees=False)
        return euler
    else:
        # Multiple quaternions
        euler_angles = []
        for i in range(quaternions.shape[0]):
            quat = quaternions[i, 3:7]  # Extract quaternion from state vector
            euler = R.from_quat(quat).as_euler('xyz', degrees=False)
            euler_angles.append(euler)
        return np.array(euler_angles)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run MPC single step')
    parser.add_argument('--robot', type=str, default='s500',
                      choices=['s500', 's500_uam', 'hexacopter370_flying_arm_3'],
                      help='Robot model to use')
    parser.add_argument('--trajectory', type=str, default='catch_vicon',
                      help='Trajectory name')
    parser.add_argument('--dt', type=int, default=50,
                      help='Time step for trajectory optimization (ms)')
    parser.add_argument('--controller', type=str, default='crocoddyl',
                      choices=['eagle_mpc', 'crocoddyl'],
                      help='MPC controller type to use')
    
    args = parser.parse_args()
    
    # Check controller availability
    if args.controller == 'eagle_mpc' and not EAGLE_MPC_AVAILABLE:
        print("Error: eagle_mpc controller requested but not available")
        return
    elif args.controller == 'crocoddyl' and not CROCODDYL_AVAILABLE:
        print("Error: crocoddyl controller requested but not available")
        return
    
    print(f"Using {args.controller} controller")
    
    # Get package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('eagle_mpc_debugger')
    
    # Initialize variables
    mpc_controller = None
    dt_mpc = args.dt / 1000.0  # Convert ms to seconds
    
    if args.controller == 'eagle_mpc':
        # Eagle MPC controller setup
        mpc_yaml_path = os.path.join(package_path, 'config/yaml/mpc/s500_uam_mpc.yaml')
        if not mpc_yaml_path.startswith('/'):
            mpc_yaml_path = os.path.join(package_path, mpc_yaml_path)
        
        # First read the YAML file to process paths
        with open(mpc_yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        
        # Process paths in the YAML data
        if 'mpc_controller' in yaml_data:
            mpc_data = yaml_data['mpc_controller']
            if 'robot' in mpc_data:
                robot_data = mpc_data['robot']
                if 'urdf' in robot_data:
                    urdf_path = robot_data['urdf']
                    if not urdf_path.startswith('/'):
                        robot_data['urdf'] = os.path.join(package_path, urdf_path)
                if 'follow' in robot_data:
                    follow_path = robot_data['follow']
                    if not follow_path.startswith('/'):
                        robot_data['follow'] = os.path.join(package_path, follow_path)
        
        # Write the processed YAML to a temporary file
        temp_yaml_path = os.path.join(package_path, 'temp_mpc.yaml')
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        
        print(f"Eagle MPC config path: {temp_yaml_path}")
        
    elif args.controller == 'crocoddyl':
        # Crocoddyl MPC controller setup
        mpc_yaml_path = os.path.join(package_path, 'config/yaml/mpc/s500_uam_mpc_crocoddyl.yaml')
        platform_yaml_path = os.path.join(package_path, 'config/yaml/multicopter/s500.yaml')
        urdf_path = os.path.join(package_path, 'models/urdf/s500_uam_simple.urdf')
        
        print(f"Crocoddyl MPC config path: {mpc_yaml_path}")
        print(f"Platform config path: {platform_yaml_path}")
        print(f"URDF path: {urdf_path}")
        
        # Load configurations
        mpc_config = load_mpc_config(mpc_yaml_path)
        platform_config = load_platform_config(platform_yaml_path)
        
        # Load robot model with free-flyer joint
        robot_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        print(f"Robot model loaded: {robot_model.name}")
        print(f"  nq (position DOF): {robot_model.nq}")
        print(f"  nv (velocity DOF): {robot_model.nv}")
        
        # Create platform parameters
        platform_params = PlatformParams(platform_config)
    
    # Create reference trajectory
    # Each state is a 17-element vector [pos(3), quat(4), joint_pos(2), vel(3), ang_vel(3), joint_vel(2)]
    single_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 17 states
    single_control = np.array([4.94, 4.94, 4.94, 4.94, 0, 0])
    
    if args.controller == 'eagle_mpc':
        # Eagle MPC controller
        dt_traj_opt = args.dt  # ms
        # Create a trajectory with multiple states (e.g., 10 time steps)
        traj_state_ref = [single_state for _ in range(10)]  # Vector of states
        traj_control_ref = [single_control for _ in range(10)]
        
        # Create mpc controller without trajectory reference control command
        mpc_controller = eagle_mpc.RailMpc(traj_state_ref, dt_traj_opt, temp_yaml_path)
        
        logger = crocoddyl.CallbackLogger()
        CallbackVerbose = crocoddyl.CallbackVerbose()
        mpc_controller.solver.setCallbacks([logger, CallbackVerbose])
        mpc_controller.updateProblem(0)
        mpc_controller.solver.convergence_init = 1e-6
        mpc_controller.logger = logger
        
        # Alternative: mpc controller with trajectory reference control command
        # mpc_controller = eagle_mpc.RailMpc(traj_state_ref, traj_control_ref, dt_traj_opt, temp_yaml_path)
        
    elif args.controller == 'crocoddyl':
        # Crocoddyl MPC controller
        mpc_controller = CrocoddylMPCController(robot_model, platform_params, mpc_config, dt_mpc)
        
        # Create a trajectory with multiple states (horizon length)
        horizon = mpc_config.get('horizon', 100)
        traj_state_ref = [single_state for _ in range(horizon)]
        
        # Set reference trajectory
        mpc_controller.set_reference_trajectory(traj_state_ref)
    
    # Initial state (slightly above ground)
    current_state = np.array([0.1, 0, -0.1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Solve MPC problem based on controller type
    if args.controller == 'eagle_mpc':
        # Eagle MPC solving
        mpc_controller.problem.x0 = current_state
        mpc_controller.updateProblem(0)
        
        mpc_dt_ms = mpc_controller.dt
        
        # Run mpc single step
        mpc_controller.solver.solve(
            mpc_controller.solver.xs,
            mpc_controller.solver.us,
            mpc_controller.iters
        )
        
        state_history = np.array(mpc_controller.solver.xs)
        control_history = np.array(mpc_controller.solver.us_squash)
        # print(f"control_history: {control_history}")
        # control_history_us = np.array(mpc_controller.solver.us)
        
        print("iter num: ", mpc_controller.solver.iter)
        print(f"Control history shape: {control_history.shape}")
        print(f"State history shape: {state_history.shape}")
        print(f"MPC dt: {mpc_dt_ms} ms")
        print(f"final cost: {mpc_controller.solver.cost}")
        
        # Use control_history for plotting
        plot_control_history = control_history
        time_step = mpc_dt_ms / 1000.0  # Convert to seconds
        
    elif args.controller == 'crocoddyl':
        # Crocoddyl MPC solving
        print("Solving MPC problem...")
        control_input, solve_info = mpc_controller.solve(current_state, reference_index=0)
        
        # Get predicted trajectories
        state_trajectory, control_trajectory = mpc_controller.get_predicted_trajectory()
        
        # Convert to numpy arrays for plotting
        state_history = np.array([np.array(state) for state in state_trajectory])
        control_history = np.array([np.array(control) for control in control_trajectory])
        print(f"control_history: {control_history}")
        print(f"Solver iterations: {solve_info['iterations']}")
        print(f"Solver converged: {solve_info['solved']}")
        print(f"Final cost: {solve_info['cost']:.6f}")
        print(f"Control input shape: {control_input.shape}")
        print(f"Control history shape: {control_history.shape}")
        print(f"State history shape: {state_history.shape}")
        print(f"MPC dt: {dt_mpc*1000:.1f} ms")
        
        # Use control_history for plotting
        plot_control_history = control_history
        time_step = dt_mpc
    
    # Create separate time axes for control and state based on their dimensions
    control_time_steps = np.arange(plot_control_history.shape[0]) * time_step
    state_time_steps = np.arange(state_history.shape[0]) * time_step
    
    # Convert quaternions to Euler angles for plotting
    euler_angles = quaternion_to_euler(state_history)
    
    # plot control command, state history, and attitude in subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # plot control command with points and lines
    # Determine number of controls to plot (up to 6 for visualization)
    n_controls = min(plot_control_history.shape[1], 6)
    ax1.plot(control_time_steps, plot_control_history[:, :n_controls], 'o-', markersize=4)
    ax1.set_title(f'{args.controller.upper()} Control Command (dim: {plot_control_history.shape[1]}, time steps: {plot_control_history.shape[0]})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Control Value')
    ax1.legend([f'Control {i+1}' for i in range(n_controls)])
    ax1.grid(True)
    
    # plot state history with points and lines (position only)
    ax2.plot(state_time_steps, state_history[:, :3], 'o-', markersize=4)
    ax2.set_title(f'{args.controller.upper()} State History - Position (dim: {state_history.shape[1]}, time steps: {state_history.shape[0]})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend(['X', 'Y', 'Z'])
    ax2.grid(True)
    
    # plot attitude history (Euler angles in degrees)
    euler_degrees = np.degrees(euler_angles)
    ax3.plot(state_time_steps, euler_degrees[:, 0], 'o-', markersize=4, label='Roll', color='red')
    ax3.plot(state_time_steps, euler_degrees[:, 1], 'o-', markersize=4, label='Pitch', color='green')
    ax3.plot(state_time_steps, euler_degrees[:, 2], 'o-', markersize=4, label='Yaw', color='blue')
    ax3.set_title(f'{args.controller.upper()} Attitude History - Euler Angles')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.legend(['Roll', 'Pitch', 'Yaw'])
    ax3.grid(True)
    
    # Add horizontal lines at 0 degrees for reference
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
