import time
import eagle_mpc
import crocoddyl
import matplotlib.pyplot as plt
import numpy as np
import rospkg
import os
import yaml

class SafeCallback(crocoddyl.CallbackAbstract):
    def __init__(self):
        super(SafeCallback, self).__init__()
        self.threshold = 20000
        self.cost = 0

    def __call__(self, solver):
        self.cost = solver.cost
        if not np.isfinite(self.cost) or self.cost > self.threshold:
            print(f"[SafeCallback] 🚨 Cost exploded: {self.cost}")
            print(f"[SafeCallback] Current state: {solver.xs[-1]}")
            print(f"[SafeCallback] Current control: {solver.us[-1]}")
            # Raise exception will NOT break the C++ solver, so you can use flags or logs instead

def get_opt_traj(robotName, trajectoryName, dt_traj_opt, useSquash, yaml_file_path):
    '''
    description: get optimized trajectory
    '''
    # Get the package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('eagle_mpc_debugger')
    
    # Construct absolute paths
    trajectory_config_path = os.path.join(package_path, yaml_file_path, 'trajectories', f'{robotName}_{trajectoryName}.yaml')
    
    # First read the YAML file to process paths
    with open(trajectory_config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Process paths in the YAML data
    if 'trajectory' in yaml_data and 'robot' in yaml_data['trajectory']:
        robot_data = yaml_data['trajectory']['robot']
        if 'urdf' in robot_data:
            urdf_path = robot_data['urdf']
            if not urdf_path.startswith('/'):
                robot_data['urdf'] = os.path.join(package_path, urdf_path)
        if 'follow' in robot_data:
            follow_path = robot_data['follow']
            if not follow_path.startswith('/'):
                robot_data['follow'] = os.path.join(package_path, follow_path)
    
    # Write the processed YAML to a temporary file
    temp_yaml_path = os.path.join(package_path, 'temp_trajectory.yaml')
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    try:
        # Load and process the YAML file
        trajectory = eagle_mpc.Trajectory()
        trajectory.autoSetup(temp_yaml_path)
        
        problem = trajectory.createProblem(dt_traj_opt, useSquash, "IntegratedActionModelEuler")

        if useSquash:
            solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)
        else:
            solver = crocoddyl.SolverBoxFDDP(problem)

        solver.convergence_init = 1e-6
        trajectory.logger = crocoddyl.CallbackLogger()
        
        solver.setCallbacks([trajectory.logger, crocoddyl.CallbackVerbose()])
        start_time = time.time()
        solver.solve([], [], maxiter=400)
        end_time = time.time()

        print("Time taken for trajectory optimization: {:.2f} ms".format((end_time - start_time)*1000))
        
        traj_state_ref = solver.xs
        
        return solver, traj_state_ref, problem, trajectory
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

def create_mpc_controller(mpc_name, trajectory, traj_state_ref, dt_traj_opt, mpc_yaml_path):
    '''
    description: create mpc controller
    '''
    # Get the package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('eagle_mpc_debugger')
    
    # Construct absolute path for MPC YAML
    if not mpc_yaml_path.startswith('/'):
        mpc_yaml_path = os.path.join(package_path, mpc_yaml_path)
    
    # First read the YAML file to process paths
    with open(mpc_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
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
    
    try:
        if mpc_name == 'rail':
            mpcController = eagle_mpc.RailMpc(traj_state_ref, dt_traj_opt, temp_yaml_path)
        elif mpc_name == 'weighted':
            mpcController = eagle_mpc.WeightedMpc(trajectory, dt_traj_opt, temp_yaml_path)
        elif mpc_name == 'carrot':
            mpcController = eagle_mpc.CarrotMpc(trajectory, traj_state_ref, dt_traj_opt, temp_yaml_path)
        
        logger = crocoddyl.CallbackLogger()
        CallbackVerbose = crocoddyl.CallbackVerbose()
        
        mpcController.safe_cb = SafeCallback()
        
        mpcController.solver.setCallbacks([logger, mpcController.safe_cb])
        mpcController.updateProblem(0)
        mpcController.solver.convergence_init = 1e-3
        
        mpcController.logger = logger
        
        return mpcController
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)