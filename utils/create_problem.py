import time
import eagle_mpc
import crocoddyl
import matplotlib.pyplot as plt
import numpy as np

class SafeCallback(crocoddyl.CallbackAbstract):
    def __init__(self):
        super(SafeCallback, self).__init__()
        self.threshold = 5000
        self.cost = 0

    def __call__(self, solver):
        self.cost = solver.cost
        if not np.isfinite(self.cost) or self.cost > self.threshold:
            print(f"[SafeCallback] 🚨 Cost exploded: {self.cost}")
            # Raise exception will NOT break the C++ solver, so you can use flags or logs instead

def get_opt_traj(robotName, trajectoryName, dt_traj_opt, useSquash, yaml_file_path):
    '''
    description: get optimized trajectory
    '''
    trajectory_config_path = '{}/trajectories/{}_{}.yaml'.format(yaml_file_path, robotName, trajectoryName)
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

def create_mpc_controller(mpc_name, trajectory, traj_state_ref, dt_traj_opt, mpc_yaml_path):
    '''
    description: create mpc controller
    '''
    if mpc_name == 'rail':
        mpcController = eagle_mpc.RailMpc(traj_state_ref, dt_traj_opt, mpc_yaml_path)
    elif mpc_name == 'weighted':
        mpcController = eagle_mpc.WeightedMpc(trajectory, dt_traj_opt, mpc_yaml_path)
    elif mpc_name == 'carrot':
        mpcController = eagle_mpc.CarrotMpc(trajectory, traj_state_ref, dt_traj_opt, mpc_yaml_path)
    
    logger = crocoddyl.CallbackLogger()
    CallbackVerbose = crocoddyl.CallbackVerbose()
    
    mpcController.safe_cb = SafeCallback()
    
    mpcController.solver.setCallbacks([logger, mpcController.safe_cb])
    mpcController.updateProblem(0)
    mpcController.solver.convergence_init = 1e-3
    
    mpcController.logger = logger
    
    return mpcController