import os
import sys
import time

import example_robot_data
import numpy as np
import pinocchio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
DEBUG = "debug" in sys.argv or "CROCODDYL_DEBUG" in os.environ
WITHPLOT = True


def initialize_quadrotor_system(
    robot_name="hector",
    d_cog=0.1525,
    cf=6.6e-5,
    cm=1e-6,
    u_lim=8.0,
    l_lim=0.1,
    pos_lb=None,
    pos_ub=None,
    vel_lb=None,
    vel_ub=None,
    angvel_lb=None,
    angvel_ub=None,
):
    """
    Initialize quadrotor system: robot model, state, actuation, and bounds
    
    Args:
        robot_name: Name of the robot model to load
        d_cog: Distance from center of gravity to thrusters
        cf: Thrust coefficient
        cm: Moment coefficient
        u_lim: Upper control limit
        l_lim: Lower control limit
        pos_lb: Lower position bounds [x, y, z] (default: [-50, -50, 0])
        pos_ub: Upper position bounds [x, y, z] (default: [50, 50, 10])
        vel_lb: Lower velocity bounds [vx, vy, vz] (default: [-3, -4, -10])
        vel_ub: Upper velocity bounds [vx, vy, vz] (default: [3, 4, 10])
        angvel_lb: Lower angular velocity bounds [wx, wy, wz] (default: [-5, -5, -5])
        angvel_ub: Upper angular velocity bounds [wx, wy, wz] (default: [5, 5, 5])
    
    Returns:
        dict: Dictionary containing all initialized components:
            - hector: Robot data
            - robot_model: Pinocchio robot model
            - state: Crocoddyl state model
            - actuation: Crocoddyl actuation model
            - nu: Number of control inputs
            - u_hover: Hover thrust vector
            - hover_thrust: Hover thrust per rotor
            - u_lim: Upper control limit
            - l_lim: Lower control limit
            - state_tangent_lb: Lower state bounds (tangent space)
            - state_tangent_ub: Upper state bounds (tangent space)
            - state_barrier_weights: Weights for state barrier activation
    """
    # Load robot model
    hector = example_robot_data.load(robot_name)
    robot_model = hector.model
    state = crocoddyl.StateMultibody(robot_model)
    
    # Thruster configuration
    ps = [
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([d_cog, 0, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CCW,
        ),
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([0, d_cog, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CW,
        ),
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([-d_cog, 0, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CCW,
        ),
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([0, -d_cog, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CW,
        ),
    ]
    actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
    nu = actuation.nu
    
    # Calculate hover thrust
    mass = robot_model.inertias[1].mass
    g = 9.81
    hover_thrust = mass * g / nu
    u_hover = np.full(nu, hover_thrust)
    
    # State bounds (with defaults)
    if pos_lb is None:
        pos_lb = np.array([-50.0, -50.0, 0.0])
    if pos_ub is None:
        pos_ub = np.array([50.0, 50.0, 10.0])
    if vel_lb is None:
        vel_lb = np.array([-3.0, -4.0, -10.0])
    if vel_ub is None:
        vel_ub = np.array([3.0, 4.0, 10.0])
    if angvel_lb is None:
        angvel_lb = np.array([-5.0, -5.0, -5.0])
    if angvel_ub is None:
        angvel_ub = np.array([5.0, 5.0, 5.0])
    
    # Create state bounds in tangent space
    ndx = state.ndx
    state_tangent_lb = np.concatenate([
        pos_lb,
        np.array([-5, -5, -5]),
        vel_lb,
        angvel_lb,
        np.full(robot_model.nv - 6, -np.inf)
    ])[:ndx]
    
    state_tangent_ub = np.concatenate([
        pos_ub,
        np.array([5, 5, 5]),
        vel_ub,
        angvel_ub,
        np.full(robot_model.nv - 6, np.inf)
    ])[:ndx]
    
    state_barrier_weights = np.concatenate([
        np.array([100.0, 100.0, 100.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([10.0, 10.0, 10.0]),
        np.ones(robot_model.nv - 6)
    ])[:ndx]
    
    print(f"Quadrotor system initialized:")
    print(f"  Robot: {robot_name}, Mass: {mass:.4f} kg")
    print(f"  Hover thrust per rotor: {hover_thrust:.4f} N")
    print(f"  Control limits: [{l_lim}, {u_lim}]")
    print(f"  State bounds: pos[{pos_lb[0]:.1f}~{pos_ub[0]:.1f}], vel[{vel_lb[0]:.1f}~{vel_ub[0]:.1f}], angvel[{angvel_lb[0]:.1f}~{angvel_ub[0]:.1f}]")
    
    return {
        'hector': hector,
        'robot_model': robot_model,
        'state': state,
        'actuation': actuation,
        'nu': nu,
        'u_hover': u_hover,
        'hover_thrust': hover_thrust,
        'u_lim': u_lim,
        'l_lim': l_lim,
        'state_tangent_lb': state_tangent_lb,
        'state_tangent_ub': state_tangent_ub,
        'state_barrier_weights': state_barrier_weights,
    }


# Initialize system
system = initialize_quadrotor_system()
hector = system['hector']
robot_model = system['robot_model']
state = system['state']
actuation = system['actuation']
nu = system['nu']
u_hover = system['u_hover']
hover_thrust = system['hover_thrust']
u_lim = system['u_lim']
l_lim = system['l_lim']
state_tangent_lb = system['state_tangent_lb']
state_tangent_ub = system['state_tangent_ub']
state_barrier_weights = system['state_barrier_weights']


def generate_figure8_trajectory(center_pos, radius, height, duration, dt=0.01):
    """
    Generate a figure-8 (lemniscate) trajectory
    
    Args:
        center_pos: Center position of the figure-8 [x, y, z]
        radius: Radius of the figure-8 in x-y plane
        height: Constant height (z coordinate)
        duration: Total duration of the trajectory (seconds)
        dt: Time step for trajectory generation (seconds)
    
    Returns:
        trajectory: Dictionary with 'times', 'positions', 'quaternions', 'velocities'
    """
    times = np.arange(0, duration, dt)
    n_points = len(times)
    
    # Generate figure-8 trajectory
    # Parametric equations: x = radius * sin(t), y = radius * sin(t) * cos(t)
    # This creates a figure-8 shape in the x-y plane
    omega = 2 * np.pi / duration  # Angular frequency for one complete loop
    
    positions = np.zeros((n_points, 3))
    quaternions = []
    velocities = np.zeros((n_points, 3))
    
    for i, t in enumerate(times):
        # Figure-8 parametric equations
        positions[i, 0] = center_pos[0] + radius * np.sin(omega * t)
        positions[i, 1] = center_pos[1] + radius * np.sin(omega * t) * np.cos(omega * t)
        positions[i, 2] = height
        
        # Compute velocities (derivatives)
        velocities[i, 0] = radius * omega * np.cos(omega * t)
        velocities[i, 1] = radius * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
        velocities[i, 2] = 0.0
        
        # Keep orientation horizontal (identity quaternion)
        quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)
        quaternions.append(quat)
    
    return {
        'times': times,
        'positions': positions,
        'quaternions': quaternions,
        'velocities': velocities
    }


def create_cost_models(target_pos, target_quat, u_hover, state, nu, robot_model,
                       state_tangent_lb, state_tangent_ub, state_barrier_weights,
                       pos_weight=10.0, vel_weight=1.0):
    """
    Create cost models for trajectory optimization or MPC tracking
    
    Args:
        target_pos: Target position [x, y, z]
        target_quat: Target quaternion
        u_hover: Hover thrust vector
        state: Crocoddyl state model
        nu: Number of control inputs
        robot_model: Pinocchio robot model
        state_tangent_lb: Lower state bounds (tangent space)
        state_tangent_ub: Upper state bounds (tangent space)
        state_barrier_weights: Weights for state barrier activation
        pos_weight: Weight for position tracking error (default: 1000.0)
        vel_weight: Weight for velocity tracking error (default: 100.0)
    """
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    
    # State regularization
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1] * 3 + [1] * 3 + [1.0] * robot_model.nv)
    )
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    
    # Control regularization (relative to hover thrust)
    uResidual = crocoddyl.ResidualModelControl(state, u_hover)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    
    # State bounds
    xBoundsResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xBoundsActivation = crocoddyl.ActivationModelWeightedQuadraticBarrier(
        crocoddyl.ActivationBounds(state_tangent_lb, state_tangent_ub, 1.0),
        state_barrier_weights
    )
    xBoundsCost = crocoddyl.CostModelResidual(state, xBoundsActivation, xBoundsResidual)
    
    # Goal tracking: Position and orientation (using FramePlacement)
    goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        robot_model.getFrameId("base_link"),
        pinocchio.SE3(target_quat.matrix(), target_pos),
        nu,
    )
    goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
    
    # Velocity tracking: Create reference state with target pose and zero velocity
    # For StateMultibody, state is [q, v] where q=[x,y,z,qx,qy,qz,qw,...] and v=[vx,vy,vz,wx,wy,wz,...]
    # Create reference configuration q_ref
    ref_q = np.concatenate([target_pos, target_quat.coeffs()])  # [x, y, z, qx, qy, qz, qw]
    if len(ref_q) < robot_model.nq:
        # Pad with zeros if needed (for joint positions if any)
        ref_q = np.concatenate([ref_q, np.zeros(robot_model.nq - len(ref_q))])
    
    # Create reference velocities (all zeros)
    ref_v = np.zeros(robot_model.nv)
    
    # Combine into full reference state [q, v]
    ref_state_full = np.concatenate([ref_q, ref_v])
    
    # Create weighted activation for velocity tracking
    # ResidualModelState works in tangent space (ndx dimensions)
    # Tangent space structure for floating base: [pos(3), quat_tangent(3), lin_vel(3), ang_vel(3), ...]
    ndx = state.ndx
    vel_tracking_weights = np.zeros(ndx)
    # Position weights: 0 (handled by FramePlacement)
    # Quaternion tangent space weights: 0 (handled by FramePlacement)
    # Linear velocity weights (indices 6-8 in tangent space for floating base)
    if ndx >= 9:
        vel_tracking_weights[6:9] = 1.0  # Linear velocity
    # Angular velocity weights (indices 9-11 in tangent space for floating base)
    if ndx >= 12:
        vel_tracking_weights[9:12] = 1.0  # Angular velocity
    # Other velocities (if any) set to 0
    
    velTrackingResidual = crocoddyl.ResidualModelState(state, ref_state_full, nu)
    velTrackingActivation = crocoddyl.ActivationModelWeightedQuad(vel_tracking_weights)
    velTrackingCost = crocoddyl.CostModelResidual(state, velTrackingActivation, velTrackingResidual)
    
    # Add costs to running model
    runningCostModel.addCost("xReg", xRegCost, 0)
    runningCostModel.addCost("uReg", uRegCost, 2)  # Control regularization weight
    runningCostModel.addCost("trackPose", goalTrackingCost, pos_weight)  # Position and orientation tracking
    runningCostModel.addCost("trackVel", velTrackingCost, vel_weight)  # Velocity tracking
    runningCostModel.addCost("stateBounds", xBoundsCost, 1.0)
    
    # Add costs to terminal model
    terminalCostModel.addCost("goalPose", goalTrackingCost, pos_weight)  # Terminal position and orientation
    terminalCostModel.addCost("stateBounds", xBoundsCost, 10.0)
    
    return runningCostModel, terminalCostModel


def trajectory_optimization(target_pos, target_quat, dt=0.1, T=100, max_iter=400, 
                            with_plot=False, debug=False):
    """
    Trajectory optimization: optimize control sequence for reaching a fixed target
    
    Args:
        target_pos: Target position [x, y, z]
        target_quat: Target quaternion
        dt: Time step for optimization
        T: Number of time steps
        max_iter: Maximum optimization iterations
        with_plot: Whether to plot during optimization
        debug: Whether to pause at each iteration
    """
    print("=" * 60)
    print("Trajectory Optimization")
    print("=" * 60)
    print(f"Target position: {target_pos}")
    print(f"Hover thrust per rotor: {hover_thrust:.4f} N")
    
    # Create cost models
    runningCostModel, terminalCostModel = create_cost_models(
        target_pos, target_quat, u_hover, state, nu, robot_model,
        state_tangent_lb, state_tangent_ub, state_barrier_weights
    )
    
    # Create action models
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt,
    )
    
    # Set control bounds
    runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
    runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
    terminalModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
    terminalModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
    
    # Create problem and solver
    x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    solver = crocoddyl.SolverBoxDDP(problem)

    # Setup callbacks
    logger = crocoddyl.CallbackLogger()
    callbacks = [crocoddyl.CallbackVerbose(), logger]
    
    if with_plot:
        # Custom plot callback (simplified version)
        class PlotCallback(crocoddyl.CallbackAbstract):
            def __init__(self, solver, dt, logger, u_hover):
                crocoddyl.CallbackAbstract.__init__(self)
                self.solver = solver
                self.dt = dt
                self.logger = logger
                self.u_hover = u_hover
                self.iteration = 0
            
            def __call__(self, solver):
                if self.iteration == 0:
                    # Create figure on first iteration
                    self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
                    self.axes = self.axes.flatten()
                
                xs = solver.xs
                us = solver.us
                xs_array = np.array([np.array(x) for x in xs])
                us_array = np.array([np.array(u) for u in us])
                time_states = np.arange(len(xs_array)) * self.dt
                time_controls = np.arange(len(us_array)) * self.dt
                
                for i in range(3):
                    self.axes[i].clear()
                
                # Plot positions
                self.axes[0].plot(time_states, xs_array[:, 0], 'r-', label='x', linewidth=2)
                self.axes[0].plot(time_states, xs_array[:, 1], 'g-', label='y', linewidth=2)
                self.axes[0].plot(time_states, xs_array[:, 2], 'b-', label='z', linewidth=2)
                self.axes[0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5)
                self.axes[0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5)
                self.axes[0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5)
                self.axes[0].set_xlabel('Time (s)')
                self.axes[0].set_ylabel('Position (m)')
                self.axes[0].set_title(f'Position - Iteration {self.iteration}')
                self.axes[0].legend()
                self.axes[0].grid(True)
                
                # Plot controls
                for i in range(us_array.shape[1]):
                    self.axes[1].plot(time_controls, us_array[:, i], 'o-', markersize=3, label=f'u{i+1}')
                self.axes[1].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5)
                self.axes[1].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5)
                self.axes[1].axhline(y=self.u_hover[0], color='g', linestyle=':', linewidth=2, label='hover')
                self.axes[1].set_xlabel('Time (s)')
                self.axes[1].set_ylabel('Control')
                self.axes[1].set_title(f'Control - Iteration {self.iteration}')
                self.axes[1].legend()
                self.axes[1].grid(True)
                
                # Plot cost
                if len(self.logger.costs) > 0:
                    self.axes[2].clear()
                    iterations = np.arange(len(self.logger.costs))
                    self.axes[2].semilogy(iterations, self.logger.costs, 'b-o')
                    self.axes[2].set_xlabel('Iteration')
                    self.axes[2].set_ylabel('Cost')
                    self.axes[2].set_title('Cost Convergence')
                    self.axes[2].grid(True)
                
                self.fig.tight_layout()
                plt.pause(0.01)
                
                if debug:
                    print(f"\n[DEBUG] Iteration {self.iteration}. Press Enter...")
                    input()
                
                self.iteration += 1
        
        plot_callback = PlotCallback(solver, dt, logger, u_hover)
        callbacks.append(plot_callback)
        plt.ion()
    
    solver.setCallbacks(callbacks)
    
    # Solve
    xs_init = [x0.copy() for _ in range(T + 1)]
    us_init = [u_hover.copy() for _ in range(T)]
    
    start_time = time.time()
    solver.solve(xs_init, us_init, max_iter)
    solve_time = time.time() - start_time
    
    print(f"Optimization completed in {solve_time:.2f} seconds")
    
    return solver, logger, dt


def plot_trajectory_optimization_results(solver, logger, dt, target_pos, u_hover):
    """Plot final results of trajectory optimization"""
    xs = solver.xs
    us = solver.us
    
    # Convert to numpy arrays
    xs_array = np.array([np.array(x) for x in xs])
    us_array = np.array([np.array(u) for u in us])
    
    # Time axis
    time_states = np.arange(len(xs_array)) * dt
    time_controls = np.arange(len(us_array)) * dt
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Positions
    axes[0, 0].plot(time_states, xs_array[:, 0], 'r-', label='x', linewidth=2)
    axes[0, 0].plot(time_states, xs_array[:, 1], 'g-', label='y', linewidth=2)
    axes[0, 0].plot(time_states, xs_array[:, 2], 'b-', label='z', linewidth=2)
    axes[0, 0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5, label='target x')
    axes[0, 0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5, label='target y')
    axes[0, 0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5, label='target z')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Optimized Trajectory: Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Linear velocities
    if xs_array.shape[1] > 6:
        axes[0, 1].plot(time_states, xs_array[:, 7], 'r-', label='vx', linewidth=2)
        axes[0, 1].plot(time_states, xs_array[:, 8], 'g-', label='vy', linewidth=2)
        axes[0, 1].plot(time_states, xs_array[:, 9], 'b-', label='vz', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Linear Velocity (m/s)')
        axes[0, 1].set_title('Optimized Trajectory: Linear Velocity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot 3: Angular velocities
    if xs_array.shape[1] > 9:
        axes[0, 2].plot(time_states, xs_array[:, 10], 'r-', label='wx', linewidth=2)
        axes[0, 2].plot(time_states, xs_array[:, 11], 'g-', label='wy', linewidth=2)
        axes[0, 2].plot(time_states, xs_array[:, 12], 'b-', label='wz', linewidth=2)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 2].set_title('Optimized Trajectory: Angular Velocity')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # Plot 4: Controls
    for i in range(us_array.shape[1]):
        axes[1, 0].plot(time_controls, us_array[:, i], 'o-', markersize=3, label=f'u{i+1}', linewidth=1.5)
    axes[1, 0].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
    axes[1, 0].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
    axes[1, 0].axhline(y=u_hover[0], color='g', linestyle=':', linewidth=2, label='hover thrust')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Input')
    axes[1, 0].set_title('Optimized Trajectory: Control')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Cost convergence
    if len(logger.costs) > 0:
        iterations = np.arange(len(logger.costs))
        axes[1, 1].semilogy(iterations, logger.costs, 'b-o', markersize=4, linewidth=1.5, label='Cost')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Cost')
        axes[1, 1].set_title('Cost Convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Plot 6: Position error over time
    pos_errors = xs_array[:, :3] - target_pos
    pos_error_norm = np.linalg.norm(pos_errors, axis=1) * 100  # Convert to cm
    axes[1, 2].plot(time_states, pos_error_norm, 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Position Error (cm)')
    axes[1, 2].set_title('Position Error vs Target')
    axes[1, 2].grid(True)
    
    # Print final statistics
    final_pos = xs_array[-1, :3]
    final_error = np.linalg.norm(final_pos - target_pos)
    print(f"\nFinal position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
    print(f"Target position: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
    print(f"Final position error: {final_error:.4f} m")
    print(f"Final cost: {solver.cost:.6f}")
    
    fig.tight_layout()
    plt.show()
    
    return fig


class QuadrotorSimulator:
    """Simulator for quadrotor dynamics using crocoddyl"""
    
    def __init__(self, state, actuation, sim_dt=0.01):
        """
        Initialize quadrotor simulator
        
        Args:
            state: Crocoddyl state model
            actuation: Crocoddyl actuation model
            sim_dt: Simulation time step (seconds)
        """
        self.state = state
        self.actuation = actuation
        self.sim_dt = sim_dt
        
        # Create a simple cost model for simulation (no cost, just dynamics)
        sim_cost = crocoddyl.CostModelSum(state, actuation.nu)
        
        # Create differential action model for simulation
        diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, sim_cost
        )
        
        # Create integrated action model with simulation time step
        self.sim_model = crocoddyl.IntegratedActionModelEuler(diff_model, sim_dt)
        self.sim_data = self.sim_model.createData()
        
        print(f"QuadrotorSimulator initialized with dt={sim_dt} s")
    
    def step(self, current_state, control):
        """
        Simulate one step forward
        
        Args:
            current_state: Current state vector
            control: Control input vector
            
        Returns:
            next_state: Next state after applying control
        """
        # Ensure inputs are numpy arrays
        if not isinstance(current_state, np.ndarray):
            current_state = np.array(current_state)
        if not isinstance(control, np.ndarray):
            control = np.array(control)
        
        # Compute next state using crocoddyl dynamics
        self.sim_model.calc(self.sim_data, current_state, control)
        next_state = self.sim_data.xnext.copy()
        
        # Verify next_state is valid
        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            print(f"WARNING: Invalid state after simulation step!")
            print(f"  current_state: {current_state}")
            print(f"  control: {control}")
            print(f"  next_state: {next_state}")
            # Return current state if next state is invalid
            return current_state.copy()
        
        return next_state
    
    def simulate(self, initial_state, controls, dt=None):
        """
        Simulate multiple steps
        
        Args:
            initial_state: Initial state
            controls: List of control inputs
            dt: Optional time step (overrides sim_dt if provided)
            
        Returns:
            states: List of states
            times: List of time points
        """
        if dt is None:
            dt = self.sim_dt
        
        states = [initial_state.copy()]
        times = [0.0]
        current_state = initial_state.copy()
        
        for i, control in enumerate(controls):
            next_state = self.step(current_state, control)
            states.append(next_state.copy())
            times.append((i + 1) * dt)
            current_state = next_state.copy()
        
        return states, times


def mpc_tracking(target_pos, target_quat, dt_mpc=0.5, N=20, 
                sim_duration=10.0, sim_dt=0.01, max_iter=100, debug_level=0):
    """
    MPC tracking: track a fixed target using MPC control
    
    Args:
        target_pos: Target position [x, y, z]
        target_quat: Target quaternion
        dt_mpc: MPC control time step (seconds)
        N: MPC prediction horizon
        sim_duration: Total simulation duration (seconds)
        sim_dt: Simulation time step (seconds)
        max_iter: Maximum MPC solver iterations per step
        debug_level: Debug level (0=no debug, 1=show step results, 2=show step results + MPC iterations)
    """
    print("=" * 60)
    print("MPC Tracking")
    print("=" * 60)
    print(f"Target position: {target_pos}")
    print(f"MPC dt: {dt_mpc} s, Horizon: {N}, Max iter: {max_iter}")
    print(f"Simulation duration: {sim_duration} s, dt: {sim_dt} s")
    
    # Create cost models (same as trajectory optimization)
    runningCostModel, terminalCostModel = create_cost_models(
        target_pos, target_quat, u_hover, state, nu, robot_model,
        state_tangent_lb, state_tangent_ub, state_barrier_weights
    )
    
    # Create action models
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt_mpc,  # Use MPC time step
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt_mpc,
    )
    
    # Set control bounds
    runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
    runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
    terminalModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
    terminalModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
    
    # Initial state
    x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
    current_state = x0.copy()
    
    # Create simulator
    simulator = QuadrotorSimulator(state, actuation, sim_dt=sim_dt)
    
    # Create initial problem (will be recreated each MPC step to ensure cost models are fresh)
    problem = crocoddyl.ShootingProblem(current_state, [runningModel] * N, terminalModel)
    solver = crocoddyl.SolverBoxDDP(problem)
    
    # Setup callbacks based on debug level
    callbacks = []
    if debug_level >= 2:
        # Level 2: Show MPC iteration process
        callbacks.append(crocoddyl.CallbackVerbose())
    else:
        # Level 0 or 1: Silent MPC solving
        pass
    
    solver.setCallbacks(callbacks)
    
    # Simulation data storage
    sim_states = [current_state.copy()]
    sim_controls = []
    sim_times = [0.0]
    mpc_solve_times = []
    
    # Current MPC control (will be updated when MPC is solved)
    current_mpc_control = u_hover.copy()
    
    # Store previous MPC solution for warm start
    prev_mpc_xs = None
    prev_mpc_us = None
    
    # Setup plotting for debug mode
    fig = None
    axes = None
    if debug_level > 0:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        fig.suptitle('MPC Tracking - Debug Mode', fontsize=16)
    
    # Simulation loop
    num_sim_steps = int(sim_duration / sim_dt)
    mpc_steps_per_solve = 1  # How many sim steps per MPC solve (calculate from dt_mpc)
    
    print(f"\nStarting simulation...")
    print(f"Total sim steps: {num_sim_steps}, MPC solve every {mpc_steps_per_solve} steps")
    print(f"Simulator dt: {sim_dt} s, MPC dt: {dt_mpc} s")
    if debug_level > 0:
        print(f"Debug mode: Level {debug_level}")
        if debug_level == 1:
            print("  - Will show step results (state, control) after each step")
        elif debug_level == 2:
            print("  - Will show step results + MPC iteration process")
        print("  - Will plot real-time state and control")
        print("  - Press Enter to continue after each step")
    
    mpc_step_count = 0  # Counter for MPC solves
    
    for sim_step in range(num_sim_steps):
        current_time = sim_step * sim_dt
        
        # Solve MPC every dt_mpc seconds
        if sim_step % mpc_steps_per_solve == 0:
            mpc_step_count += 1
            
            if debug_level > 0:
                print("\n" + "=" * 60)
                print(f"MPC Solve #{mpc_step_count} at t={current_time:.3f}s")
                print("=" * 60)
                print(f"Current state (before MPC): pos=[{current_state[0]:.4f}, {current_state[1]:.4f}, {current_state[2]:.4f}]")
            
            # Recreate cost models to ensure they are fresh (though target is fixed, this ensures consistency)
            # For fixed target tracking, cost models don't need to change, but recreating ensures everything is fresh
            runningCostModel, terminalCostModel = create_cost_models(
                target_pos, target_quat, u_hover, state, nu, robot_model, 
                state_tangent_lb, state_tangent_ub, state_barrier_weights
            )
            
            # Recreate action models with fresh cost models
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuation, runningCostModel
                ),
                dt_mpc,
            )
            terminalModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuation, terminalCostModel
                ),
                dt_mpc,
            )
            
            # Set control bounds
            runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
            runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
            terminalModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
            terminalModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
            
            # Recreate problem with current state and fresh models
            problem = crocoddyl.ShootingProblem(current_state.copy(), [runningModel] * N, terminalModel)
            
            # Recreate solver with new problem
            solver = crocoddyl.SolverBoxDDP(problem)
            
            # Restore callbacks
            callbacks = []
            if debug_level >= 2:
                callbacks.append(crocoddyl.CallbackVerbose())
            solver.setCallbacks(callbacks)
            
            # Warm start: shift previous solution
            if prev_mpc_xs is not None and prev_mpc_us is not None and len(prev_mpc_xs) > 1:
                # Use previous solution as warm start (shift forward)
                xs_init = [current_state.copy()] + [prev_mpc_xs[i+1].copy() for i in range(min(N, len(prev_mpc_xs)-1))]
                # Pad if needed
                while len(xs_init) < N + 1:
                    xs_init.append(xs_init[-1].copy())
                
                us_init = [prev_mpc_us[i+1].copy() if i+1 < len(prev_mpc_us) else u_hover.copy() 
                          for i in range(N)]
                if debug_level > 0:
                    print(f"Using warm start from previous MPC solution")
            else:
                # First solve: use hover thrust
                xs_init = [current_state.copy() for _ in range(N + 1)]
                us_init = [u_hover.copy() for _ in range(N)]
                if debug_level > 0:
                    print(f"First MPC solve: using hover thrust as initial guess")
            
            # Solve MPC
            if debug_level >= 2:
                print(f"Solving MPC (max_iter={max_iter})...")
            
            solve_start = time.time()
            solver.solve(xs_init, us_init, max_iter)
            solve_time = time.time() - solve_start
            mpc_solve_times.append(solve_time)
            
            # Store solution for next warm start
            prev_mpc_xs = [x.copy() for x in solver.xs] if len(solver.xs) > 0 else None
            prev_mpc_us = [u.copy() for u in solver.us] if len(solver.us) > 0 else None
            
            # Update current MPC control (use first control from solution)
            prev_mpc_control = current_mpc_control.copy() if hasattr(current_mpc_control, 'copy') else current_mpc_control
            if len(solver.us) > 0:
                current_mpc_control = solver.us[0].copy()
                if debug_level > 0:
                    control_diff = np.linalg.norm(current_mpc_control - prev_mpc_control) if hasattr(prev_mpc_control, '__len__') else float('inf')
                    print(f"MPC control updated: {current_mpc_control} (change: {control_diff:.6f})")
            else:
                current_mpc_control = u_hover.copy()
                if debug_level > 0:
                    print(f"WARNING: No control from solver, using hover thrust: {current_mpc_control}")
            
            # Display MPC solve results
            if debug_level > 0:
                print(f"MPC solved: cost={solver.cost:.6f}, iter={solver.iter}, "
                      f"solve_time={solve_time*1000:.1f}ms")
                print(f"Predicted horizon: {N} steps ({N*dt_mpc:.2f}s)")
                
                # Show predicted trajectory (first few steps)
                if len(solver.xs) > 0:
                    print("\nPredicted trajectory (first 5 steps):")
                    for i in range(min(5, len(solver.xs))):
                        pred_pos = solver.xs[i][:3]
                        print(f"  Step {i}: pos=[{pred_pos[0]:.3f}, {pred_pos[1]:.3f}, {pred_pos[2]:.3f}]")
                
                # Show control sequence (first few steps)
                if len(solver.us) > 0:
                    print("\nControl sequence (first 5 steps):")
                    for i in range(min(5, len(solver.us))):
                        print(f"  Step {i}: u={solver.us[i]}")
            
            # Normal print (non-debug mode)
            if debug_level == 0 and (sim_step == 0 or (sim_step // mpc_steps_per_solve) % 10 == 0):
                print(f"t={current_time:.2f}s: MPC solved, cost={solver.cost:.4f}, "
                      f"iter={solver.iter}, solve_time={solve_time*1000:.1f}ms")
        
        # Apply current MPC control and simulate one step forward
        prev_state = current_state.copy()
        next_state = simulator.step(current_state, current_mpc_control)
        
        # Verify state actually changed
        state_diff = np.linalg.norm(next_state - current_state)
        if debug_level > 0:
            if state_diff < 1e-10:
                print(f"WARNING: State did not change! diff={state_diff:.2e}")
            else:
                pos_diff = np.linalg.norm(next_state[:3] - current_state[:3])
                print(f"State updated: pos_diff={pos_diff:.6f} m, total_diff={state_diff:.6f}")
        
        # Store data
        sim_controls.append(current_mpc_control.copy())
        current_state = next_state.copy()  # Update current state for next iteration
        sim_states.append(current_state.copy())
        sim_times.append((sim_step + 1) * sim_dt)
        
        # Debug mode: display step information and plot
        if debug_level > 0:
            print(f"\n--- Simulation Step {sim_step+1}/{num_sim_steps} (t={sim_times[-1]:.3f}s) ---")
            
            # Current state information
            pos = current_state[:3]
            quat = current_state[3:7]
            if len(current_state) > 7:
                vel = current_state[7:10] if len(current_state) > 10 else current_state[7:7+state.nv]
                angvel = current_state[10:13] if len(current_state) > 13 else np.zeros(3)
            else:
                vel = np.zeros(3)
                angvel = np.zeros(3)
            
            print(f"Current State:")
            print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            if len(vel) >= 3:
                print(f"  Linear velocity: [{vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f}] m/s")
            if len(angvel) >= 3:
                print(f"  Angular velocity: [{angvel[0]:.4f}, {angvel[1]:.4f}, {angvel[2]:.4f}] rad/s")
            
            # Control information
            print(f"Applied Control: {current_mpc_control}")
            
            # Position error
            pos_error = np.linalg.norm(pos - target_pos)
            print(f"Position error to target: {pos_error:.4f} m")
            
            # Update plots
            if fig is not None and axes is not None:
                # Convert to arrays for plotting
                sim_states_array = np.array([np.array(x) for x in sim_states])
                sim_controls_array = np.array([np.array(u) for u in sim_controls])
                
                # Clear all axes
                for ax in axes:
                    ax.clear()
                
                # Plot 1: Position trajectory
                axes[0].plot(sim_times, sim_states_array[:, 0], 'r-', label='x', linewidth=2)
                axes[0].plot(sim_times, sim_states_array[:, 1], 'g-', label='y', linewidth=2)
                axes[0].plot(sim_times, sim_states_array[:, 2], 'b-', label='z', linewidth=2)
                axes[0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5, label='target x')
                axes[0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5, label='target y')
                axes[0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5, label='target z')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Position (m)')
                axes[0].set_title(f'Position Trajectory (t={sim_times[-1]:.2f}s)')
                axes[0].legend()
                axes[0].grid(True)
                
                # Plot 2: Linear velocity
                if sim_states_array.shape[1] > 6:
                    axes[1].plot(sim_times, sim_states_array[:, 7], 'r-', label='vx', linewidth=2)
                    axes[1].plot(sim_times, sim_states_array[:, 8], 'g-', label='vy', linewidth=2)
                    axes[1].plot(sim_times, sim_states_array[:, 9], 'b-', label='vz', linewidth=2)
                    axes[1].set_xlabel('Time (s)')
                    axes[1].set_ylabel('Linear Velocity (m/s)')
                    axes[1].set_title('Linear Velocity')
                    axes[1].legend()
                    axes[1].grid(True)
    
                # Plot 3: Angular velocity
                if sim_states_array.shape[1] > 9:
                    axes[2].plot(sim_times, sim_states_array[:, 10], 'r-', label='wx', linewidth=2)
                    axes[2].plot(sim_times, sim_states_array[:, 11], 'g-', label='wy', linewidth=2)
                    axes[2].plot(sim_times, sim_states_array[:, 12], 'b-', label='wz', linewidth=2)
                    axes[2].set_xlabel('Time (s)')
                    axes[2].set_ylabel('Angular Velocity (rad/s)')
                    axes[2].set_title('Angular Velocity')
                    axes[2].legend()
                    axes[2].grid(True)
                
                # Plot 4: Control inputs
                for i in range(sim_controls_array.shape[1]):
                    axes[3].plot(sim_times[:-1], sim_controls_array[:, i], 'o-', markersize=3, 
                               label=f'u{i+1}', linewidth=1.5)
                axes[3].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
                axes[3].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
                axes[3].axhline(y=u_hover[0], color='g', linestyle=':', linewidth=2, label='hover thrust')
                axes[3].set_xlabel('Time (s)')
                axes[3].set_ylabel('Control Input')
                axes[3].set_title('Control Trajectory')
                axes[3].legend()
                axes[3].grid(True)
                
                # Plot 5: Position error
                pos_errors = sim_states_array[:, :3] - target_pos
                pos_error_norm = np.linalg.norm(pos_errors, axis=1) * 100  # Convert to cm
                axes[4].plot(sim_times, pos_error_norm, 'b-', linewidth=2)
                axes[4].set_xlabel('Time (s)')
                axes[4].set_ylabel('Position Error (cm)')
                axes[4].set_title('Position Error to Target')
                axes[4].grid(True)
                
                # Plot 6: MPC predicted trajectory (if available)
                if len(solver.xs) > 0 and sim_step % mpc_steps_per_solve == 0:
                    # Show MPC predicted trajectory
                    mpc_pred_times = np.array([i * dt_mpc for i in range(len(solver.xs))])
                    mpc_pred_states = np.array([np.array(x) for x in solver.xs])
                    axes[5].plot(mpc_pred_times, mpc_pred_states[:, 0], 'r--', label='pred x', linewidth=1.5, alpha=0.7)
                    axes[5].plot(mpc_pred_times, mpc_pred_states[:, 1], 'g--', label='pred y', linewidth=1.5, alpha=0.7)
                    axes[5].plot(mpc_pred_times, mpc_pred_states[:, 2], 'b--', label='pred z', linewidth=1.5, alpha=0.7)
                    # Current position as starting point
                    axes[5].plot(0, pos[0], 'ro', markersize=8, label='current x')
                    axes[5].plot(0, pos[1], 'go', markersize=8, label='current y')
                    axes[5].plot(0, pos[2], 'bo', markersize=8, label='current z')
                    axes[5].axhline(y=target_pos[0], color='r', linestyle=':', alpha=0.5)
                    axes[5].axhline(y=target_pos[1], color='g', linestyle=':', alpha=0.5)
                    axes[5].axhline(y=target_pos[2], color='b', linestyle=':', alpha=0.5)
                    axes[5].set_xlabel('Time (s)')
                    axes[5].set_ylabel('Position (m)')
                    axes[5].set_title(f'MPC Predicted Trajectory (horizon={N*dt_mpc:.1f}s)')
                    axes[5].legend()
                    axes[5].grid(True)
                else:
                    axes[5].text(0.5, 0.5, 'MPC prediction\n(updated at MPC solve)', 
                               ha='center', va='center', transform=axes[5].transAxes)
                    axes[5].set_title('MPC Predicted Trajectory')
                
                fig.tight_layout()
                plt.pause(0.01)  # Update plot
            
            # Wait for user input
            input("\nPress Enter to continue to next step...")
    
    print(f"\nSimulation completed!")
    print(f"Average MPC solve time: {np.mean(mpc_solve_times)*1000:.1f} ms")
    print(f"Total MPC solves: {len(mpc_solve_times)}")
    
    # Convert to arrays for plotting
    sim_states_array = np.array([np.array(x) for x in sim_states])
    sim_controls_array = np.array([np.array(u) for u in sim_controls])
    
    return sim_states_array, sim_controls_array, sim_times, mpc_solve_times


def mpc_trajectory_tracking(trajectory, dt_mpc=0.05, N=20, 
                            sim_duration=10.0, sim_dt=0.01, max_iter=10, debug_level=0):
    """
    MPC trajectory tracking: track a time-varying trajectory using MPC control
    
    Args:
        trajectory: Dictionary with 'times', 'positions', 'quaternions', 'velocities'
        dt_mpc: MPC control time step (seconds)
        N: MPC prediction horizon
        sim_duration: Total simulation duration (seconds)
        sim_dt: Simulation time step (seconds)
        max_iter: Maximum MPC solver iterations per step
        debug_level: Debug level (0=no debug, 1=show step results, 2=show step results + MPC iterations)
    """
    print("=" * 60)
    print("MPC Trajectory Tracking")
    print("=" * 60)
    print(f"Trajectory duration: {trajectory['times'][-1]:.2f} s, {len(trajectory['times'])} points")
    print(f"MPC dt: {dt_mpc} s, Horizon: {N}, Max iter: {max_iter}")
    print(f"Simulation duration: {sim_duration} s, dt: {sim_dt} s")
    
    # Initial state
    x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
    current_state = x0.copy()
    
    # Create simulator
    simulator = QuadrotorSimulator(state, actuation, sim_dt=sim_dt)
    
    # Simulation data storage
    sim_states = [current_state.copy()]
    sim_controls = []
    sim_times = [0.0]
    mpc_solve_times = []
    
    # Current MPC control (will be updated when MPC is solved)
    current_mpc_control = u_hover.copy()
    
    # Store previous MPC solution for warm start
    prev_mpc_xs = None
    prev_mpc_us = None
    
    # Setup plotting for debug mode
    fig = None
    axes = None
    if debug_level > 0:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        fig.suptitle('MPC Trajectory Tracking - Debug Mode', fontsize=16)
    
    # Simulation loop
    num_sim_steps = int(sim_duration / sim_dt)
    mpc_steps_per_solve = int(dt_mpc / sim_dt)  # How many sim steps per MPC solve
    
    print(f"\nStarting simulation...")
    print(f"Total sim steps: {num_sim_steps}, MPC solve every {mpc_steps_per_solve} steps")
    print(f"Simulator dt: {sim_dt} s, MPC dt: {dt_mpc} s")
    if debug_level > 0:
        print(f"Debug mode: Level {debug_level}")
        if debug_level == 1:
            print("  - Will show step results (state, control) after each step")
        elif debug_level == 2:
            print("  - Will show step results + MPC iteration process")
        print("  - Will plot real-time state and control")
        print("  - Press Enter to continue after each step")
    
    mpc_step_count = 0  # Counter for MPC solves
    
    for sim_step in range(num_sim_steps):
        current_time = sim_step * sim_dt
        
        # Solve MPC every dt_mpc seconds
        if sim_step % mpc_steps_per_solve == 0:
            mpc_step_count += 1
            
            if debug_level > 0:
                print("\n" + "=" * 60)
                print(f"MPC Solve #{mpc_step_count} at t={current_time:.3f}s")
                print("=" * 60)
                print(f"Current state (before MPC): pos=[{current_state[0]:.4f}, {current_state[1]:.4f}, {current_state[2]:.4f}]")
            
            # Get reference state from trajectory at current time
            traj_idx = min(int(current_time / sim_dt), len(trajectory['times']) - 1)
            ref_pos = trajectory['positions'][traj_idx]
            ref_quat = trajectory['quaternions'][traj_idx]
            
            if debug_level > 0:
                print(f"Reference position: [{ref_pos[0]:.4f}, {ref_pos[1]:.4f}, {ref_pos[2]:.4f}]")
            
            # Create cost models with current reference
            runningCostModel, terminalCostModel = create_cost_models(
                ref_pos, ref_quat, u_hover, state, nu, robot_model,
                state_tangent_lb, state_tangent_ub, state_barrier_weights
            )
            
            # Recreate action models with fresh cost models
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuation, runningCostModel
                ),
                dt_mpc,
            )
            terminalModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuation, terminalCostModel
                ),
                dt_mpc,
            )
            
            # Set control bounds
            runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
            runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
            terminalModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
            terminalModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
            
            # Recreate problem with current state and fresh models
            problem = crocoddyl.ShootingProblem(current_state.copy(), [runningModel] * N, terminalModel)
            
            # Recreate solver with new problem
            solver = crocoddyl.SolverBoxDDP(problem)
            
            # Restore callbacks
            callbacks = []
            if debug_level >= 2:
                callbacks.append(crocoddyl.CallbackVerbose())
            solver.setCallbacks(callbacks)
            
            # Warm start: shift previous solution
            if prev_mpc_xs is not None and prev_mpc_us is not None and len(prev_mpc_xs) > 1:
                # Use previous solution as warm start (shift forward)
                xs_init = [current_state.copy()] + [prev_mpc_xs[i+1].copy() for i in range(min(N, len(prev_mpc_xs)-1))]
                # Pad if needed
                while len(xs_init) < N + 1:
                    xs_init.append(xs_init[-1].copy())
                
                us_init = [prev_mpc_us[i+1].copy() if i+1 < len(prev_mpc_us) else u_hover.copy() 
                          for i in range(N)]
                if debug_level > 0:
                    print(f"Using warm start from previous MPC solution")
            else:
                # First solve: use hover thrust
                xs_init = [current_state.copy() for _ in range(N + 1)]
                us_init = [u_hover.copy() for _ in range(N)]
                if debug_level > 0:
                    print(f"First MPC solve: using hover thrust as initial guess")
            
            # Solve MPC
            if debug_level >= 2:
                print(f"Solving MPC (max_iter={max_iter})...")
            
            solve_start = time.time()
            solver.solve(xs_init, us_init, max_iter)
            solve_time = time.time() - solve_start
            mpc_solve_times.append(solve_time)
            
            # Store solution for next warm start
            prev_mpc_xs = [x.copy() for x in solver.xs] if len(solver.xs) > 0 else None
            prev_mpc_us = [u.copy() for u in solver.us] if len(solver.us) > 0 else None
            
            # Update current MPC control (use first control from solution)
            prev_mpc_control = current_mpc_control.copy() if hasattr(current_mpc_control, 'copy') else current_mpc_control
            if len(solver.us) > 0:
                current_mpc_control = solver.us[0].copy()
                if debug_level > 0:
                    control_diff = np.linalg.norm(current_mpc_control - prev_mpc_control) if hasattr(prev_mpc_control, '__len__') else float('inf')
                    print(f"MPC control updated: {current_mpc_control} (change: {control_diff:.6f})")
            else:
                current_mpc_control = u_hover.copy()
                if debug_level > 0:
                    print(f"WARNING: No control from solver, using hover thrust: {current_mpc_control}")
            
            # Display MPC solve results
            if debug_level > 0:
                print(f"MPC solved: cost={solver.cost:.6f}, iter={solver.iter}, "
                      f"solve_time={solve_time*1000:.1f}ms")
                print(f"Predicted horizon: {N} steps ({N*dt_mpc:.2f}s)")
            
            # Normal print (non-debug mode)
            if debug_level == 0 and (sim_step == 0 or (sim_step // mpc_steps_per_solve) % 10 == 0):
                print(f"t={current_time:.2f}s: MPC solved, cost={solver.cost:.4f}, "
                      f"iter={solver.iter}, solve_time={solve_time*1000:.1f}ms")
        
        # Apply current MPC control and simulate one step forward
        prev_state = current_state.copy()
        next_state = simulator.step(current_state, current_mpc_control)
        
        # Verify state actually changed
        state_diff = np.linalg.norm(next_state - current_state)
        if debug_level > 0:
            if state_diff < 1e-10:
                print(f"WARNING: State did not change! diff={state_diff:.2e}")
            else:
                pos_diff = np.linalg.norm(next_state[:3] - current_state[:3])
                print(f"State updated: pos_diff={pos_diff:.6f} m, total_diff={state_diff:.6f}")
        
        # Store data
        sim_controls.append(current_mpc_control.copy())
        current_state = next_state.copy()  # Update current state for next iteration
        sim_states.append(current_state.copy())
        sim_times.append((sim_step + 1) * sim_dt)
        
        # Debug mode: display step information and plot
        if debug_level > 0:
            print(f"\n--- Simulation Step {sim_step+1}/{num_sim_steps} (t={sim_times[-1]:.3f}s) ---")
            
            # Current state information
            pos = current_state[:3]
            quat = current_state[3:7]
            if len(current_state) > 7:
                vel = current_state[7:10] if len(current_state) > 10 else current_state[7:7+state.nv]
                angvel = current_state[10:13] if len(current_state) > 13 else np.zeros(3)
            else:
                vel = np.zeros(3)
                angvel = np.zeros(3)
            
            print(f"Current State:")
            print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            if len(vel) >= 3:
                print(f"  Linear velocity: [{vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f}] m/s")
            if len(angvel) >= 3:
                print(f"  Angular velocity: [{angvel[0]:.4f}, {angvel[1]:.4f}, {angvel[2]:.4f}] rad/s")
            
            # Control information
            print(f"Applied Control: {current_mpc_control}")
            
            # Get current reference for error calculation
            traj_idx = min(int(sim_times[-1] / sim_dt), len(trajectory['times']) - 1)
            ref_pos = trajectory['positions'][traj_idx]
            pos_error = np.linalg.norm(pos - ref_pos)
            print(f"Position error to reference: {pos_error:.4f} m")
            
            # Update plots
            if fig is not None and axes is not None:
                # Convert to arrays for plotting
                sim_states_array = np.array([np.array(x) for x in sim_states])
                sim_controls_array = np.array([np.array(u) for u in sim_controls])
                
                # Clear all axes
                for ax in axes:
                    ax.clear()
                
                # Plot 1: Position trajectory vs reference
                axes[0].plot(sim_times, sim_states_array[:, 0], 'r-', label='x', linewidth=2)
                axes[0].plot(sim_times, sim_states_array[:, 1], 'g-', label='y', linewidth=2)
                axes[0].plot(sim_times, sim_states_array[:, 2], 'b-', label='z', linewidth=2)
                # Plot reference trajectory
                ref_times = trajectory['times'][:len(sim_times)]
                ref_positions = trajectory['positions'][:len(sim_times)]
                axes[0].plot(ref_times, ref_positions[:, 0], 'r--', alpha=0.5, label='ref x')
                axes[0].plot(ref_times, ref_positions[:, 1], 'g--', alpha=0.5, label='ref y')
                axes[0].plot(ref_times, ref_positions[:, 2], 'b--', alpha=0.5, label='ref z')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Position (m)')
                axes[0].set_title(f'Position Trajectory (t={sim_times[-1]:.2f}s)')
                axes[0].legend()
                axes[0].grid(True)
                
                # Plot 2: Linear velocity
                if sim_states_array.shape[1] > 6:
                    axes[1].plot(sim_times, sim_states_array[:, 7], 'r-', label='vx', linewidth=2)
                    axes[1].plot(sim_times, sim_states_array[:, 8], 'g-', label='vy', linewidth=2)
                    axes[1].plot(sim_times, sim_states_array[:, 9], 'b-', label='vz', linewidth=2)
                    axes[1].set_xlabel('Time (s)')
                    axes[1].set_ylabel('Linear Velocity (m/s)')
                    axes[1].set_title('Linear Velocity')
                    axes[1].legend()
                    axes[1].grid(True)
                
                # Plot 3: Angular velocity
                if sim_states_array.shape[1] > 9:
                    axes[2].plot(sim_times, sim_states_array[:, 10], 'r-', label='wx', linewidth=2)
                    axes[2].plot(sim_times, sim_states_array[:, 11], 'g-', label='wy', linewidth=2)
                    axes[2].plot(sim_times, sim_states_array[:, 12], 'b-', label='wz', linewidth=2)
                    axes[2].set_xlabel('Time (s)')
                    axes[2].set_ylabel('Angular Velocity (rad/s)')
                    axes[2].set_title('Angular Velocity')
                    axes[2].legend()
                    axes[2].grid(True)
                
                # Plot 4: Control inputs
                for i in range(sim_controls_array.shape[1]):
                    axes[3].plot(sim_times[:-1], sim_controls_array[:, i], 'o-', markersize=3, 
                              label=f'u{i+1}', linewidth=1.5)
                axes[3].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
                axes[3].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
                axes[3].axhline(y=u_hover[0], color='g', linestyle=':', linewidth=2, label='hover thrust')
                axes[3].set_xlabel('Time (s)')
                axes[3].set_ylabel('Control Input')
                axes[3].set_title('Control Trajectory')
                axes[3].legend()
                axes[3].grid(True)
                
                # Plot 5: Position error
                ref_positions_aligned = trajectory['positions'][:len(sim_times)]
                pos_errors = sim_states_array[:, :3] - ref_positions_aligned
                pos_error_norm = np.linalg.norm(pos_errors, axis=1) * 100  # Convert to cm
                axes[4].plot(sim_times, pos_error_norm, 'b-', linewidth=2)
                axes[4].set_xlabel('Time (s)')
                axes[4].set_ylabel('Position Error (cm)')
                axes[4].set_title('Position Error to Reference')
                axes[4].grid(True)
                
                # Plot 6: 3D trajectory visualization
                axes[5].plot(sim_states_array[:, 0], sim_states_array[:, 1], 'b-', label='actual', linewidth=2)
                ref_positions_aligned = trajectory['positions'][:len(sim_times)]
                axes[5].plot(ref_positions_aligned[:, 0], ref_positions_aligned[:, 1], 'r--', label='reference', linewidth=2, alpha=0.7)
                axes[5].plot(sim_states_array[0, 0], sim_states_array[0, 1], 'go', markersize=10, label='start')
                axes[5].plot(sim_states_array[-1, 0], sim_states_array[-1, 1], 'ro', markersize=10, label='current')
                axes[5].set_xlabel('X (m)')
                axes[5].set_ylabel('Y (m)')
                axes[5].set_title('2D Trajectory (X-Y plane)')
                axes[5].legend()
                axes[5].grid(True)
                axes[5].axis('equal')
                
                fig.tight_layout()
                plt.pause(0.01)  # Update plot
            
            # Wait for user input
            input("\nPress Enter to continue to next step...")
    
    print(f"\nSimulation completed!")
    print(f"Average MPC solve time: {np.mean(mpc_solve_times)*1000:.1f} ms")
    print(f"Total MPC solves: {len(mpc_solve_times)}")
    
    # Convert to arrays for plotting
    sim_states_array = np.array([np.array(x) for x in sim_states])
    sim_controls_array = np.array([np.array(u) for u in sim_controls])
    
    return sim_states_array, sim_controls_array, sim_times, mpc_solve_times


def plot_mpc_results(sim_states, sim_controls, sim_times, target_pos):
    """Plot MPC tracking results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot positions
    axes[0].plot(sim_times[:-1], sim_states[:-1, 0], 'r-', label='x', linewidth=2)
    axes[0].plot(sim_times[:-1], sim_states[:-1, 1], 'g-', label='y', linewidth=2)
    axes[0].plot(sim_times[:-1], sim_states[:-1, 2], 'b-', label='z', linewidth=2)
    axes[0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5, label='target x')
    axes[0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5, label='target y')
    axes[0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5, label='target z')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('MPC Tracking: Position')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot velocities
    if sim_states.shape[1] > 6:
        axes[1].plot(sim_times[:-1], sim_states[:-1, 7], 'r-', label='vx', linewidth=2)
        axes[1].plot(sim_times[:-1], sim_states[:-1, 8], 'g-', label='vy', linewidth=2)
        axes[1].plot(sim_times[:-1], sim_states[:-1, 9], 'b-', label='vz', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Linear Velocity (m/s)')
        axes[1].set_title('MPC Tracking: Linear Velocity')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot controls
    for i in range(sim_controls.shape[1]):
        axes[2].plot(sim_times[:-1], sim_controls[:, i], 'o-', markersize=2, label=f'u{i+1}', linewidth=1)
    axes[2].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
    axes[2].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
    axes[2].axhline(y=u_hover[0], color='g', linestyle=':', linewidth=2, label='hover thrust')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Control Input')
    axes[2].set_title('MPC Tracking: Control')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot position error
    pos_errors = sim_states[:-1, :3] - target_pos
    pos_error_norm = np.linalg.norm(pos_errors, axis=1) * 100  # Convert to cm
    axes[3].plot(sim_times[:-1], pos_error_norm, 'b-', linewidth=2)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Position Error (cm)')
    axes[3].set_title('MPC Tracking: Position Error')
    axes[3].grid(True)
    
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    traj_opt = False
    mpc = True
    
    # MPC tracking mode: "fixed_point" or "figure8"
    mpc_mode = "figure8"  # Options: "fixed_point" or "figure8"
    
    # Fixed point target (for fixed_point mode)
    target_pos = np.array([1.0, 1.0, 1.0])
    target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)
    
    # Figure-8 trajectory parameters (for figure8 mode)
    figure8_center = np.array([0.0, 0.0, 0.0])  # Center position
    figure8_radius = 2.0  # Radius of the figure-8
    figure8_height = 0.0  # Constant height
    figure8_duration = 20.0  # Duration for one complete figure-8 loop

    # Run trajectory optimization
    if traj_opt:
        solver, logger, dt = trajectory_optimization(
            target_pos, target_quat, 
            dt=0.1, T=100, max_iter=400,
            with_plot=WITHPLOT, debug=DEBUG
        )
        
        # Plot final results
        if WITHPLOT:
            plot_trajectory_optimization_results(solver, logger, dt, target_pos, u_hover)
    
    # Run MPC tracking
    if mpc:
        # Determine debug level from command line or environment
        debug_level = 0
        if "debug" in sys.argv:
            debug_level = 1  # Default to level 1
        if "debug2" in sys.argv or "debug=2" in " ".join(sys.argv):
            debug_level = 2  # Level 2: show MPC iterations
        elif "debug=1" in " ".join(sys.argv):
            debug_level = 1  # Level 1: show step results only
        
        debug_level = 0
        
        if mpc_mode == "fixed_point":
            # Fixed point tracking
            print("\n" + "=" * 60)
            print("MPC Mode: Fixed Point Tracking")
            print("=" * 60)
            sim_states, sim_controls, sim_times, mpc_solve_times = mpc_tracking(
                target_pos, target_quat,
                dt_mpc=0.05, N=20,
                sim_duration=10.0, sim_dt=0.01,
                max_iter=100,
                debug_level=debug_level
            )
            
            if WITHPLOT:
                plot_mpc_results(sim_states, sim_controls, sim_times, target_pos)
        
        elif mpc_mode == "figure8":
            # Figure-8 trajectory tracking
            print("\n" + "=" * 60)
            print("MPC Mode: Figure-8 Trajectory Tracking")
            print("=" * 60)
            
            # Generate figure-8 trajectory
            trajectory = generate_figure8_trajectory(
                center_pos=figure8_center,
                radius=figure8_radius,
                height=figure8_height,
                duration=figure8_duration,
                dt=0.01
            )
            
            print(f"Generated figure-8 trajectory:")
            print(f"  Center: {figure8_center}")
            print(f"  Radius: {figure8_radius} m")
            print(f"  Height: {figure8_height} m")
            print(f"  Duration: {figure8_duration} s")
            
            # Run MPC trajectory tracking
            sim_states, sim_controls, sim_times, mpc_solve_times = mpc_trajectory_tracking(
                trajectory,
                dt_mpc=0.05, N=20,
                sim_duration=figure8_duration, sim_dt=0.01,
                max_iter=100,
                debug_level=debug_level
            )
            
            if WITHPLOT:
                # Plot trajectory tracking results
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                # Plot 1: 3D trajectory
                axes[0] = fig.add_subplot(2, 2, 1, projection='3d')
                # Align arrays for 3D plot
                min_len = min(len(sim_states), len(trajectory['positions']))
                axes[0].plot(sim_states[:min_len, 0], sim_states[:min_len, 1], sim_states[:min_len, 2], 
                           'b-', label='actual', linewidth=2)
                axes[0].plot(trajectory['positions'][:min_len, 0], trajectory['positions'][:min_len, 1], 
                           trajectory['positions'][:min_len, 2], 'r--', label='reference', linewidth=2, alpha=0.7)
                axes[0].set_xlabel('X (m)')
                axes[0].set_ylabel('Y (m)')
                axes[0].set_zlabel('Z (m)')
                axes[0].set_title('3D Trajectory')
                axes[0].legend()
                axes[0].grid(True)
                
                # Set z-axis range to at least 0.5m
                z_min = min(np.min(sim_states[:min_len, 2]), np.min(trajectory['positions'][:min_len, 2]))
                z_max = max(np.max(sim_states[:min_len, 2]), np.max(trajectory['positions'][:min_len, 2]))
                z_range = z_max - z_min
                if z_range < 0.5:
                    z_center = (z_min + z_max) / 2.0
                    z_min = z_center - 0.25
                    z_max = z_center + 0.25
                axes[0].set_zlim(z_min, z_max)
                
                # Plot 2: Position tracking
                axes[1].plot(sim_times, sim_states[:, 0], 'r-', label='x', linewidth=2)
                axes[1].plot(sim_times, sim_states[:, 1], 'g-', label='y', linewidth=2)
                axes[1].plot(sim_times, sim_states[:, 2], 'b-', label='z', linewidth=2)
                # Align reference trajectory with simulation states
                min_len = min(len(sim_times), len(trajectory['times']))
                ref_times = trajectory['times'][:min_len]
                ref_positions = trajectory['positions'][:min_len]
                axes[1].plot(ref_times, ref_positions[:, 0], 'r--', alpha=0.5, label='ref x')
                axes[1].plot(ref_times, ref_positions[:, 1], 'g--', alpha=0.5, label='ref y')
                axes[1].plot(ref_times, ref_positions[:, 2], 'b--', alpha=0.5, label='ref z')
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('Position (m)')
                axes[1].set_title('Position Tracking')
                axes[1].legend()
                axes[1].grid(True)
                
                # Plot 3: Position error
                # Align arrays: take minimum length and pad reference if needed
                min_len = min(len(sim_states), len(trajectory['positions']))
                ref_positions_aligned = trajectory['positions'][:min_len]
                sim_states_aligned = sim_states[:min_len, :3]
                pos_errors = sim_states_aligned - ref_positions_aligned
                pos_error_norm = np.linalg.norm(pos_errors, axis=1) * 100  # Convert to cm
                sim_times_aligned = sim_times[:min_len]
                axes[2].plot(sim_times_aligned, pos_error_norm, 'b-', linewidth=2)
                axes[2].set_xlabel('Time (s)')
                axes[2].set_ylabel('Position Error (cm)')
                axes[2].set_title('Position Error to Reference')
                axes[2].grid(True)
                
                # Plot 4: Control inputs
                for i in range(sim_controls.shape[1]):
                    axes[3].plot(sim_times[:-1], sim_controls[:, i], 'o-', markersize=3, 
                              label=f'u{i+1}', linewidth=1.5)
                axes[3].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
                axes[3].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
                axes[3].axhline(y=u_hover[0], color='g', linestyle=':', linewidth=2, label='hover thrust')
                axes[3].set_xlabel('Time (s)')
                axes[3].set_ylabel('Control Input')
                axes[3].set_title('Control Trajectory')
                axes[3].legend()
                axes[3].grid(True)
                
                fig.tight_layout()
                plt.show()
        
        else:
            print(f"Unknown MPC mode: {mpc_mode}")
            print("Available modes: 'fixed_point', 'figure8'")
