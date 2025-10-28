# run mpc single step

import argparse
import numpy as np
import rospkg
import eagle_mpc
import os
import yaml
import matplotlib.pyplot as plt


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run MPC single step')
    parser.add_argument('--robot', type=str, default='s500',
                      choices=['s500', 's500_uam', 'hexacopter370_flying_arm_3'],
                      help='Robot model to use')
    parser.add_argument('--trajectory', type=str, default='catch_vicon',
                      help='Trajectory name')
    parser.add_argument('--dt', type=int, default=10,
                      help='Time step for trajectory optimization (ms)')
    
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('eagle_mpc_debugger')
    mpc_yaml_path = os.path.join(package_path, 'config/yaml/mpc/s500_uam_mpc.yaml')
    # temp_yaml_path = '/home/helei/catkin_eagle_mpc/src/eagle_mpc_debugger/config/yaml/mpc/s500_uam_mpc.yaml'
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
    
    print(f"Temp YAML path: {temp_yaml_path}")
    
    # create mpc controller
    dt_traj_opt = 10  # ms
    # Create a vector of states (multiple time steps)
    # Each state is a 17-element vector
    single_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 17 states
    
    single_control = np.array([4.94, 4.94, 4.94, 4.94, 0, 0])
    # Create a trajectory with multiple states (e.g., 10 time steps)
    traj_state_ref = [single_state for _ in range(10)]  # Vector of states

    traj_control_ref = [single_control for _ in range(10)]
    
    # mpc controller without trajectory reference control command
    mpcController = eagle_mpc.RailMpc(traj_state_ref, dt_traj_opt, temp_yaml_path)
    
    # mpc controller with trajectory reference control command
    # mpcController = eagle_mpc.RailMpc(traj_state_ref, traj_control_ref, dt_traj_opt, temp_yaml_path)
    
    # update problem
    state = np.array([0, 0, 0.1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    mpcController.problem.x0 = state
    mpcController.updateProblem(0)
    
    mpc_dt = mpcController.dt
    
    # run mpc single step
    mpcController.solver.solve(
        mpcController.solver.xs,
        mpcController.solver.us,
        mpcController.iters
    )
    
    state_history = np.array(mpcController.solver.xs)
    control_history = np.array(mpcController.solver.us_squash)
    control_history_us = np.array(mpcController.solver.us)
    
    print("iter num: ", mpcController.solver.iter)
    print(f"Control history shape: {control_history.shape}")
    print(f"State history shape: {state_history.shape}")
    print(f"MPC dt: {mpc_dt} ms")
    print(f"final cost: {mpcController.solver.cost}")
    
    # Create separate time axes for control and state based on their dimensions
    control_time_steps = np.arange(control_history.shape[0]) * mpc_dt / 1000.0  # Convert to seconds
    state_time_steps = np.arange(state_history.shape[0]) * mpc_dt / 1000.0  # Convert to seconds
    
    # plot control command and state history in subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # plot control command with points and lines
    # ax1.plot(control_time_steps, control_history_us[:, :4], '--')
    ax1.plot(control_time_steps, control_history[:, :4], 'o-', markersize=4)
    ax1.set_title(f'Control Command (dim: {control_history.shape[1]}, time steps: {control_history.shape[0]})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Control Value')
    ax1.legend([f'Control {i+1}' for i in range(control_history.shape[1])])
    ax1.grid(True)
    
    # plot state history with points and lines
    ax2.plot(state_time_steps, state_history[:, :3], 'o-', markersize=4)
    ax2.set_title(f'State History (dim: {state_history.shape[1]}, time steps: {state_history.shape[0]})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('State Value')
    ax2.legend([f'State {i+1}' for i in range(state_history.shape[1])])
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
