import time
import eagle_mpc
import crocoddyl

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

    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    start_time = time.time()
    solver.solve([], [], maxiter=100)
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
        
    mpcController.solver.setCallbacks([crocoddyl.CallbackVerbose()])  # 设置回调函数 
    mpcController.updateProblem(0)
    mpcController.solver.convergence_init = 1e-3
    
    return mpcController