import time
import eagle_mpc
import crocoddyl
import matplotlib.pyplot as plt
import numpy as np
import rospkg
import os
import yaml
import pinocchio as pin
from collections import OrderedDict

# Custom YAML representer to keep lists on one line
def represent_list(dumper, data):
    # If the list is numeric and reasonably short, use flow style
    if len(data) <= 20 and all(isinstance(x, (int, float)) for x in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    else:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

# Custom representer for OrderedDict to maintain order
def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

# Add the custom representers
yaml.add_representer(list, represent_list)
yaml.add_representer(OrderedDict, represent_ordereddict)

# Custom constructor for OrderedDict to preserve order when loading
def construct_mapping(self, node):
    self.flatten_mapping(node)
    return OrderedDict(self.construct_pairs(node))

yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

def modify_catch_yaml_config(yaml_data, catch_config):
    """
    Modify YAML data for catch task configuration.
    
    Args:
        yaml_data: Original YAML data
        catch_config: Dictionary containing catch task parameters
    
    Returns:
        Modified YAML data
    """
    if 'trajectory' not in yaml_data:
        return yaml_data
    
    # Set initial state
    yaml_data['trajectory']['initial_state'] = catch_config['initial_state']
    
    # Modify stages based on catch configuration
    if 'stages' in yaml_data['trajectory']:
        stages = yaml_data['trajectory']['stages']
        
        for stage in stages:
            stage_name = stage.get('name', '').lower()
            
            # Modify approach stage
            if 'approach' in stage_name:
                stage['duration'] = catch_config['pre_grasp_time']
                # Update state reference for approach phase
                # for cost in stage.get('costs', []):
                #     if cost.get('name') == 'state_reg' and 'reference' in cost:
                #         # Use intermediate position between initial and grasp position
                #         initial_pos = catch_config['initial_state'][:3]
                #         target_pos = catch_config['target_gripper_pos']
                #         intermediate_pos = [(initial_pos[i] + target_pos[i]) / 2 for i in range(3)]
                #         cost['reference'][:3] = intermediate_pos
                
            # Modify pre_grasp stage
            elif stage_name == 'pre_grasp':   # define pitch angle before grasp
                # Update gripper target position and orientation
                for cost in stage.get('costs', []):
                    if cost.get('name') == 'translation_ee':
                        cost['position'] = catch_config['target_gripper_pos']   
                    if cost.get('name') == 'rotation_ee':
                        cost['orientation'] = catch_config['target_gripper_orient']
            
            # Modify grasp stage
            elif stage_name == 'grasp':  # constrain position and velocity of the gripper
                stage['duration'] = catch_config['grasp_time']
                # Update gripper target position and orientation
                for cost in stage.get('costs', []):
                    if cost.get('name') == 'translation_ee':
                        cost['position'] = catch_config['target_gripper_pos']
            
            # Modify move_away or post-grasp stage
            elif 'move_away' in stage_name:
                stage['duration'] = catch_config['post_grasp_time']
            
            # Modify final hover stage
            elif 'hover_after_grasp' in stage_name or 'final' in stage_name:
                # Set duration to 0 for terminal stage
                stage['duration'] = 0
                for cost in stage.get('costs', []):
                    if cost.get('name') == 'state_all' and 'reference' in cost:
                        cost['reference'] = catch_config['final_state']
    
    return yaml_data

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

def get_opt_traj(robotName, trajectoryName, dt_traj_opt, useSquash, yaml_file_path, catch_config=None):
    '''
    description: get optimized trajectory
    '''
    # Get the package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('eagle_mpc_debugger')
    
    # Construct absolute paths
    trajectory_config_path = os.path.join(package_path, yaml_file_path, 'trajectories', f'{robotName}_{trajectoryName}.yaml')
    # trajectory_config_path = os.path.join(package_path, yaml_file_path, 'trajectories', f'generated_trajectory.yaml')
    
    # First read the YAML file to process paths
    with open(trajectory_config_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    
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
    
    # Apply catch configuration if provided
    if catch_config is not None and 'catch' in trajectoryName.lower():
        # Import the modify function (assuming it's in the same module that calls this)
        # For now, we'll implement the modification logic here
        yaml_data = modify_catch_yaml_config(yaml_data, catch_config)
        # save the modified yaml to a file
        with open(os.path.join(package_path, yaml_file_path, 'trajectories', 'temp_trajectory.yaml'), 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    # Write the processed YAML to a temporary file
    temp_yaml_path = os.path.join(package_path, yaml_file_path, 'trajectories', 'temp_trajectory.yaml')
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    try:
        # Load and process the YAML file
        trajectory = eagle_mpc.Trajectory()
        trajectory.autoSetup(temp_yaml_path)
        
        problem = trajectory.createProblem(dt_traj_opt, useSquash, "IntegratedActionModelEuler")
        # problem = trajectory.createProblem()

        if useSquash:
            solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)
        else:
            solver = crocoddyl.SolverBoxFDDP(problem)

        solver.convergence_init = 1e-12
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
        pass
        # if os.path.exists(temp_yaml_path):
        #     os.remove(temp_yaml_path)
        
def load_trajectory_from_generated_yaml(dt_traj_opt, useSquash):
    '''
    description: load trajectory from config/yaml/trajectories/temp_trajectory.yaml
    '''
    # Get the package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('eagle_mpc_debugger')
    
    # Construct absolute path for the generated trajectory YAML
    trajectory_config_path = os.path.join(package_path, 'config/yaml/trajectories/temp_trajectory.yaml')
    
    try:
        trajectory = eagle_mpc.Trajectory()
        trajectory.autoSetup(trajectory_config_path)
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
        # if os.path.exists(temp_yaml_path):
        #     os.remove(temp_yaml_path)
        print(f"Loaded trajectory from generated YAML with {len(traj_state_ref)} points")

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
            
def create_state_update_model(urdf_model_path, dt):
    '''
    description: create state update model
    '''
    robot_model = pin.buildModelFromUrdf(urdf_model_path, pin.JointModelFreeFlyer())
    
    state = crocoddyl.StateMultibody(robot_model)
    actuation = crocoddyl.ActuationModelFull(state)
    action_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, crocoddyl.CostModelSum(state, actuation.nu))
    
    return crocoddyl.IntegratedActionModelEuler(action_model, dt/1000)

def create_state_update_model_quadrotor(robotModel, platformParams, dt):
    robotModel = robotModel
    robotState = crocoddyl.StateMultibody(robotModel)
    platformParams = platformParams
    dt = dt / 1000.

    actuationModel = crocoddyl.ActuationModelMultiCopterBase(
        robotState, platformParams.tau_f)
    difAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        robotState, actuationModel, crocoddyl.CostModelSum(robotState, actuationModel.nu))
    intAM = crocoddyl.IntegratedActionModelEuler(difAM, dt)
    intAD = intAM.createData()
    
    print(f"State update model created successfully")
    print(f"  MPC model uses dt = {dt} s")
    print(f"  Simulation model uses dt = {dt} s")
    
    return intAM, intAD