import os
import sys
import time

import example_robot_data
import numpy as np
import pinocchio
import matplotlib.pyplot as plt

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
# WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
WITHPLOT = True
DEBUG = "debug" in sys.argv or "CROCODDYL_DEBUG" in os.environ
# DEBUG = True

hector = example_robot_data.load("hector")
robot_model = hector.model

target_pos = np.array([1.0, 1.0, 1.0])
target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)

state = crocoddyl.StateMultibody(robot_model)

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
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
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Calculate hover thrust (gravity compensation)
# Get mass from base link (usually index 1, index 0 is universe)
mass = robot_model.inertias[1].mass
g = 9.81  # gravity acceleration
hover_thrust = mass * g / nu  # Evenly distributed among all rotors
u_hover = np.full(nu, hover_thrust)  # Reference control for hover

print(f"Hover thrust per rotor: {hover_thrust:.4f} N")
print(f"Total hover thrust: {hover_thrust * nu:.4f} N (mass: {mass:.4f} kg)")

# State bounds (constraints)
# Position bounds (x, y, z) - in meters
pos_lb = np.array([-50.0, -50.0, 0.0])  # Lower bounds: x, y, z (z >= 0 to stay above ground)
pos_ub = np.array([50.0, 50.0, 10.0])   # Upper bounds: x, y, z

# Velocity bounds (vx, vy, vz) - in m/s
vel_lb = np.array([-3.0, -4.0, -10.0])  # Lower bounds
vel_ub = np.array([3.0, 4.0, 10.0])     # Upper bounds

# Angular velocity bounds (wx, wy, wz) - in rad/s
angvel_lb = np.array([-5.0, -5.0, -5.0])  # Lower bounds
angvel_ub = np.array([5.0, 5.0, 5.0])     # Upper bounds

print(f"State bounds:")
print(f"  Position: x[{pos_lb[0]:.1f}, {pos_ub[0]:.1f}], y[{pos_lb[1]:.1f}, {pos_ub[1]:.1f}], z[{pos_lb[2]:.1f}, {pos_ub[2]:.1f}]")
print(f"  Linear velocity: [{vel_lb[0]:.1f}, {vel_ub[0]:.1f}] m/s")
print(f"  Angular velocity: [{angvel_lb[0]:.1f}, {angvel_ub[0]:.1f}] rad/s")

# Costs
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1] * 3 + [1] * 3 + [1.0] * robot_model.nv)
)
# Control residual relative to hover thrust
uResidual = crocoddyl.ResidualModelControl(state, u_hover)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# State bounds cost using barrier activation
# Create state bounds: [pos_lb, pos_ub, vel_lb, vel_ub, angvel_lb, angvel_ub]
# For StateMultibody, the state is in configuration space (q) and velocity space (v)
# q: [position(3), quaternion(4), ...], v: [linear_vel(3), angular_vel(3), ...]
# We need to create bounds for the tangent space (ndx dimension)
ndx = state.ndx  # Tangent space dimension
# Tangent space: [dp(3), dq(3), dv(3), dw(3), ...]
# For position: dp bounds
# For orientation: dq bounds (small, for quaternion)
# For velocities: dv and dw bounds

# Create bounds for tangent space
# Position tangent: [dx, dy, dz] - same as position bounds
# Orientation tangent: [drx, dry, drz] - small bounds for quaternion variations
# Velocity tangent: [dvx, dvy, dvz, dwx, dwy, dwz] - same as velocity bounds
state_tangent_lb = np.concatenate([
    pos_lb,                    # Position tangent (3)
    np.array([-5, -5, -5]),  # Orientation tangent (3) - small bounds
    vel_lb,                    # Linear velocity tangent (3)
    angvel_lb,                 # Angular velocity tangent (3)
    np.full(robot_model.nv - 6, -np.inf)  # Other velocities (if any)
])[:ndx]

state_tangent_ub = np.concatenate([
    pos_ub,                    # Position tangent (3)
    np.array([5, 5, 5]),   # Orientation tangent (3)
    vel_ub,                    # Linear velocity tangent (3)
    angvel_ub,                 # Angular velocity tangent (3)
    np.full(robot_model.nv - 6, np.inf)   # Other velocities (if any)
])[:ndx]

# Weights for barrier activation (higher weight = stronger constraint)
state_barrier_weights = np.concatenate([
    np.array([100.0, 100.0, 100.0]),  # Position weights
    np.array([10.0, 10.0, 10.0]),     # Orientation weights
    np.array([10.0, 10.0, 10.0]),     # Linear velocity weights
    np.array([10.0, 10.0, 10.0]),     # Angular velocity weights
    np.ones(robot_model.nv - 6)       # Other velocities
])[:ndx]

# State bounds residual and cost
xBoundsResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xBoundsActivation = crocoddyl.ActivationModelWeightedQuadraticBarrier(
    crocoddyl.ActivationBounds(state_tangent_lb, state_tangent_ub, 1.0),
    state_barrier_weights
)
xBoundsCost = crocoddyl.CostModelResidual(state, xBoundsActivation, xBoundsResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("base_link"),
    pinocchio.SE3(target_quat.matrix(), target_pos),
    nu,
)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
runningCostModel.addCost("xReg", xRegCost, 0)
runningCostModel.addCost("uReg", uRegCost, 2)
runningCostModel.addCost("trackPose", goalTrackingCost, 1)
runningCostModel.addCost("stateBounds", xBoundsCost, 1.0)  # High weight for state bounds
terminalCostModel.addCost("goalPose", goalTrackingCost, 0.0)
terminalCostModel.addCost("stateBounds", xBoundsCost, 10.0)  # Also apply bounds to terminal state

dt = 0.1
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
runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])
terminalModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
terminalModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])

# Creating the shooting problem and the BoxDDP solver
T = 100
x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
solver = crocoddyl.SolverBoxDDP(problem)

# Custom callback for plotting during optimization
class PlotCallback(crocoddyl.CallbackAbstract):
    def __init__(self, solver, dt, logger, fig=None, axes=None, debug=False, u_hover=None):
        crocoddyl.CallbackAbstract.__init__(self)
        self.solver = solver
        self.dt = dt
        self.logger = logger
        self.fig = fig
        self.axes = axes
        self.iteration = 0
        self.debug = debug
        self.u_hover = u_hover
        
    def __call__(self, solver):
        if self.fig is None or self.axes is None:
            return
        
        # Get current states and controls
        xs = solver.xs
        us = solver.us
        
        # Convert to numpy arrays
        xs_array = np.array([np.array(x) for x in xs])
        us_array = np.array([np.array(u) for u in us])
        
        # Time axis
        time_states = np.arange(len(xs_array)) * self.dt
        time_controls = np.arange(len(us_array)) * self.dt
        
        # Clear and update plots (except cost plot which accumulates)
        for i in range(3):  # Clear first 3 plots
            self.axes[i].clear()
        
        # Plot positions (x, y, z)
        self.axes[0].plot(time_states, xs_array[:, 0], 'r-', label='x', linewidth=2)
        self.axes[0].plot(time_states, xs_array[:, 1], 'g-', label='y', linewidth=2)
        self.axes[0].plot(time_states, xs_array[:, 2], 'b-', label='z', linewidth=2)
        self.axes[0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5, label='target x')
        self.axes[0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5, label='target y')
        self.axes[0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5, label='target z')
        self.axes[0].set_xlabel('Time (s)')
        self.axes[0].set_ylabel('Position (m)')
        self.axes[0].set_title(f'Position Trajectory - Iteration {self.iteration}')
        self.axes[0].legend()
        self.axes[0].grid(True)
        
        # Plot velocities (vx, vy, vz)
        if xs_array.shape[1] > 6:
            self.axes[1].plot(time_states, xs_array[:, 7], 'r-', label='vx', linewidth=2)
            self.axes[1].plot(time_states, xs_array[:, 8], 'g-', label='vy', linewidth=2)
            self.axes[1].plot(time_states, xs_array[:, 9], 'b-', label='vz', linewidth=2)
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].set_ylabel('Linear Velocity (m/s)')
            self.axes[1].set_title(f'Linear Velocity - Iteration {self.iteration}')
            self.axes[1].legend()
            self.axes[1].grid(True)
        
        # Plot controls
        for i in range(us_array.shape[1]):
            self.axes[2].plot(time_controls, us_array[:, i], 'o-', markersize=3, label=f'u{i+1}', linewidth=1.5)
        self.axes[2].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
        self.axes[2].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
        # Add hover thrust reference line
        if self.u_hover is not None:
            hover_thrust_val = self.u_hover[0]  # All rotors have same hover thrust
            self.axes[2].axhline(y=hover_thrust_val, color='g', linestyle=':', alpha=0.7, linewidth=2, label='hover thrust')
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('Control Input')
        self.axes[2].set_title(f'Control Trajectory - Iteration {self.iteration}')
        self.axes[2].legend()
        self.axes[2].grid(True)
        
        # Plot cost convergence from logger (always update to show all iterations)
        if self.logger is not None and len(self.logger.costs) > 0:
            self.axes[3].clear()  # Clear and redraw with all data
            iterations = np.arange(len(self.logger.costs))
            self.axes[3].semilogy(iterations, self.logger.costs, 'b-o', markersize=4, linewidth=1.5)
            self.axes[3].set_xlabel('Iteration')
            self.axes[3].set_ylabel('Cost')
            self.axes[3].set_title('Cost Convergence')
            self.axes[3].grid(True)
        
        self.fig.tight_layout()
        plt.pause(0.01)  # Small pause to update the plot
        
        # Debug mode: wait for user to press Enter before continuing
        if self.debug:
            print(f"\n[DEBUG] Iteration {self.iteration} completed. Press Enter to continue to next iteration...")
            input()
        
        self.iteration += 1

# Setup plotting if WITHPLOT is True
if WITHPLOT:
    # Create logger first
    logger = crocoddyl.CallbackLogger()
    
    # Create figure and axes for real-time plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Get initial guess for plotting (use hover thrust for controls)
    xs_init_plot = [x0.copy() for _ in range(T + 1)]
    us_init_plot = [u_hover.copy() for _ in range(T)]  # Use hover thrust instead of zeros
    
    # Plot initial state before optimization
    xs_init_array = np.array([np.array(x) for x in xs_init_plot])
    us_init_array = np.array([np.array(u) for u in us_init_plot])
    time_states = np.arange(len(xs_init_array)) * dt
    time_controls = np.arange(len(us_init_array)) * dt
    
    # Plot initial positions
    axes[0].plot(time_states, xs_init_array[:, 0], 'r-', label='x', linewidth=2, alpha=0.7)
    axes[0].plot(time_states, xs_init_array[:, 1], 'g-', label='y', linewidth=2, alpha=0.7)
    axes[0].plot(time_states, xs_init_array[:, 2], 'b-', label='z', linewidth=2, alpha=0.7)
    axes[0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5, label='target x')
    axes[0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5, label='target y')
    axes[0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5, label='target z')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Position Trajectory - Initial (Before Optimization)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot initial velocities
    if xs_init_array.shape[1] > 6:
        axes[1].plot(time_states, xs_init_array[:, 7], 'r-', label='vx', linewidth=2, alpha=0.7)
        axes[1].plot(time_states, xs_init_array[:, 8], 'g-', label='vy', linewidth=2, alpha=0.7)
        axes[1].plot(time_states, xs_init_array[:, 9], 'b-', label='vz', linewidth=2, alpha=0.7)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Linear Velocity (m/s)')
        axes[1].set_title('Linear Velocity - Initial (Before Optimization)')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot initial controls
    for i in range(us_init_array.shape[1]):
        axes[2].plot(time_controls, us_init_array[:, i], 'o-', markersize=3, label=f'u{i+1}', linewidth=1.5, alpha=0.7)
    axes[2].axhline(y=u_lim, color='r', linestyle='--', alpha=0.5, label='upper limit')
    axes[2].axhline(y=l_lim, color='b', linestyle='--', alpha=0.5, label='lower limit')
    # Add hover thrust reference line
    hover_thrust_val = u_hover[0]  # All rotors have same hover thrust
    axes[2].axhline(y=hover_thrust_val, color='g', linestyle=':', alpha=0.7, linewidth=2, label='hover thrust')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Control Input')
    axes[2].set_title('Control Trajectory - Initial (Before Optimization)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Initialize cost plot
    axes[3].set_xlabel('Iteration')
    axes[3].set_ylabel('Cost')
    axes[3].set_title('Cost Convergence')
    axes[3].grid(True)
    
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Show initial plot
    
    # Debug mode: wait for user to press Enter before starting optimization
    if DEBUG:
        print("\n[DEBUG] Initial state plotted. Press Enter to start optimization...")
        input()
    
    # Create plot callback
    plot_callback = PlotCallback(solver, dt, logger, fig, axes, debug=DEBUG, u_hover=u_hover)
    
    # Setup callbacks (logger must be before plot_callback)
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            logger,
            plot_callback,
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    # Debug mode: wait for user to press Enter before starting optimization (even without plotting)
    if DEBUG:
        print("\n[DEBUG] Ready to start optimization. Press Enter to continue...")
        input()

# Solving the problem with the BoxDDP solver
# Create initial guess: states and controls
xs_init = [x0.copy() for _ in range(T + 1)]  # Initial state sequence (T+1 states)
us_init = [u_hover.copy() for _ in range(T)]  # Initial control sequence (T controls) - all set to hover thrust

print(f"Initial control guess: all set to hover thrust = {u_hover[0]:.4f} N per rotor")

start_time = time.time()
solver.solve(xs_init, us_init, 400)
# print(np.array(solver.Qu))
print(time.time() - start_time)

# Plotting the entire motion (final results)
if WITHPLOT:
    # Get the logger callback (should be the second callback)
    callbacks = solver.getCallbacks()
    for cb in callbacks:
        if isinstance(cb, crocoddyl.CallbackLogger):
            log = cb
            break
    else:
        # Fallback: create a new logger if not found
        log = crocoddyl.CallbackLogger()
    
    # Plot final solution in separate figure
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=3, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=4
    )
    plt.show()  # Show all plots

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [-0.03, 4.4, 2.3, -0.02, 0.56, 0.83, -0.03]
        display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF, floor=False)
        hector.viewer.gui.addXYZaxis("world/wp", [1.0, 0.0, 0.0, 1.0], 0.03, 0.5)
        hector.viewer.gui.applyConfiguration(
            "world/wp",
            [
                *target_pos.tolist(),
                target_quat[0],
                target_quat[1],
                target_quat[2],
                target_quat[3],
            ],
        )
    except Exception:
        display = crocoddyl.MeshcatDisplay(hector)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
