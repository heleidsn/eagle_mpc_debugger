#!/usr/bin/env python3
"""
Pure Crocoddyl MPC Controller for Trajectory Tracking
Author: Assistant
Date: 2025-10-28
Description: A pure Crocoddyl implementation of MPC controller without eagle_mpc dependency
"""

import argparse
import numpy as np
import crocoddyl
import pinocchio as pin
import yaml
import os
from scipy.spatial.transform import Rotation as R


class CrocoddylMPCController:
    """
    Pure Crocoddyl MPC Controller for trajectory tracking
    """
    
    def __init__(self, robot_model, platform_params, mpc_config, dt_mpc=0.02):
        """
        Initialize Crocoddyl MPC Controller
        
        Args:
            robot_model: Pinocchio robot model
            platform_params: Platform parameters (thrust coefficients, etc.)
            mpc_config: MPC configuration dictionary
            dt_mpc: MPC time step (seconds)
        """
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.platform_params = platform_params
        self.dt_mpc = dt_mpc
        
        # MPC parameters
        self.horizon = int(mpc_config.get('horizon', 20))
        self.max_iterations = int(mpc_config.get('max_iterations', 20))
        self.convergence_tolerance = float(mpc_config.get('convergence_tolerance', 1e-6))
        
        # State and control dimensions
        self.state_dim = robot_model.nq + robot_model.nv
        
        # Initialize state first to get the correct tangent space dimension
        self.state = crocoddyl.StateMultibody(self.robot_model)
        
        # Create actuation model first to get correct control dimension
        self.actuation = self._create_actuation_model()
        
        # Use actuation model's control dimension (nu) instead of robot.nv
        self.control_dim = self.actuation.nu
        
        # Get the correct tangent space dimension (ndx) for weights
        # For quaternion states: nq=9, nv=8, but ndx=16 (quaternion tangent space is 3D, not 4D)
        tangent_dim = self.state.ndx
        
        # Read weights and bounds from config with safe defaults.
        state_weights = mpc_config.get('state_weights')
        control_weights = mpc_config.get('control_weights')
        terminal_weights = mpc_config.get('terminal_weights')
        control_lower_bounds = mpc_config.get('control_lower_bounds')
        control_upper_bounds = mpc_config.get('control_upper_bounds')
        
        # Adjust weights to match actual dimensions and ensure float type
        self.state_weights = np.array(state_weights, dtype=np.float64)
        self.control_weights = np.array(control_weights, dtype=np.float64)
        self.terminal_weights = np.array(terminal_weights, dtype=np.float64)
        
        # Store dt scaling option for use in cost function creation
        self.enable_dt_scaling = mpc_config.get('enable_dt_scaling', True)
        
        if self.enable_dt_scaling:
            print(f"Eagle MPC-style dt scaling enabled (dt = {self.dt_mpc}s)")
            print("  Weights will be applied in cost function with dt scaling")
            print(f"  State weights: {self.state_weights}")
            print(f"  Control weights: {self.control_weights}")
            print(f"  Terminal weights: {self.terminal_weights}")
        else:
            print("Eagle MPC-style dt scaling disabled - using standard Crocoddyl approach")
            print(f"  State weights: {self.state_weights}")
            print(f"  Control weights: {self.control_weights}")
            print(f"  Terminal weights: {self.terminal_weights}")
        
        # Adjust control bounds to match actual control dimensions
        self.control_lower_bounds = np.array(control_lower_bounds, dtype=np.float64)
        self.control_upper_bounds = np.array(control_upper_bounds, dtype=np.float64)
        
        # Initialize trajectory reference
        self.reference_trajectory = []
        self.current_reference_index = 0
        
        # Initialize MPC problem
        self.problem = None
        self.solver = None
        
        print(f"Crocoddyl MPC Controller initialized:")
        print(f"  State dimension (nq+nv): {self.state_dim}")
        print(f"  Tangent space dimension (ndx): {tangent_dim}")
        print(f"  Control dimension (robot.nv): {self.control_dim}")
        print(f"  Actuation control dimension (nu): {self.actuation.nu}")
        print(f"  Horizon: {self.horizon}")
        print(f"  Max solver iterations: {self.max_iterations}")
        print(f"  MPC time step: {self.dt_mpc}")
        print(f"  State weights shape: {self.state_weights.shape}")
        print(f"  Control weights shape: {self.control_weights.shape}")
        print(f"  Terminal weights shape: {self.terminal_weights.shape}")
        print(f"  Control bounds: [{self.control_lower_bounds.min():.1f}, {self.control_upper_bounds.max():.1f}]")
        print(f"  Control lower bounds: {self.control_lower_bounds}")
        print(f"  Control upper bounds: {self.control_upper_bounds}")
    
    def _create_actuation_model(self):
        """Create actuation model based on platform parameters"""
        # Create multirotor actuation model
        n_rotors = self.platform_params.n_rotors
        tau_f = self.platform_params.tau_f
        
        # Create actuation model
        try:
            # Try to use the new FloatingBaseThrusters model
            # This requires creating individual thruster objects
            thrusters = []
            for i in range(n_rotors):
                # Create thruster with torque coefficient only
                # According to C++ signature: Thruster(double ctorque)
                torque_coeff = float(tau_f[5, i]) if tau_f.shape[0] > 5 else 0.1
                thruster = crocoddyl.Thruster(torque_coeff)
                thrusters.append(thruster)
            
            actuation = crocoddyl.ActuationModelFloatingBaseThrusters(
                self.state, thrusters
            )
            print("Using ActuationModelFloatingBaseThrusters")
        except (AttributeError, TypeError) as e:
            print(f"FloatingBaseThrusters failed: {e}")
            try:
                # Fallback to deprecated MultiCopterBase without n_rotors
                actuation = crocoddyl.ActuationModelMultiCopterBase(
                    self.state, tau_f
                )
                print("Using deprecated ActuationModelMultiCopterBase")
            except (AttributeError, TypeError) as e:
                print(f"MultiCopterBase failed: {e}")
                # Final fallback to full actuation model
                print("Warning: Multicopter actuation models not available, using full actuation")
                actuation = crocoddyl.ActuationModelFull(self.state)
        
        return actuation
    
    def set_reference_trajectory(self, reference_states, reference_controls=None):
        """
        Set reference trajectory for MPC

        若只需跟踪固定目标状态，可用 make_constant_tracking_reference(...) 或
        set_constant_tracking_reference(...)。

        Args:
            reference_states: List of reference states
            reference_controls: List of reference controls (optional)
        """
        self.reference_trajectory = reference_states
        self.reference_controls = reference_controls if reference_controls is not None else []
        print(f"Reference trajectory set with {len(reference_states)} states")
    
    def _create_running_model(self, ref_state, ref_control=None):
        """Create running model for a single time step"""
        # Create cost model first with correct control dimension
        cost_model = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        
        # Apply dt scaling to cost weights (like Eagle MPC) if enabled
        if self.enable_dt_scaling:
            # Eagle MPC scales weights by dt_s = dt_mpc (in seconds)
            dt_scale = self.dt_mpc
        else:
            # Standard Crocoddyl approach without dt scaling
            dt_scale = 1.0
        
        # State regulation cost
        state_residual = crocoddyl.ResidualModelState(self.state, ref_state, self.actuation.nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(self.state_weights)
        state_cost = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        cost_model.addCost("state_reg", state_cost, dt_scale*100)
        
        # Control regulation cost
        if ref_control is not None:
            # Ensure ref_control has correct dimension
            if len(ref_control) != self.actuation.nu:
                ref_control_resized = np.zeros(self.actuation.nu)
                ref_control_resized[:min(len(ref_control), self.actuation.nu)] = ref_control[:min(len(ref_control), self.actuation.nu)]
                control_residual = crocoddyl.ResidualModelControl(self.state, ref_control_resized)
            else:
                control_residual = crocoddyl.ResidualModelControl(self.state, ref_control)
        else:
            control_residual = crocoddyl.ResidualModelControl(self.state, self.actuation.nu)
        control_activation = crocoddyl.ActivationModelWeightedQuad(self.control_weights)
        control_cost = crocoddyl.CostModelResidual(self.state, control_activation, control_residual)
        cost_model.addCost("control_reg", control_cost, dt_scale)
        
        # Create differential action model with all required arguments
        diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_model
        )
        
        # Create integrated action model
        int_model = crocoddyl.IntegratedActionModelEuler(diff_model, self.dt_mpc)
        
        # Set control bounds directly on the running model
        int_model.u_lb = self.control_lower_bounds
        int_model.u_ub = self.control_upper_bounds
        
        return int_model
    
    def _create_terminal_model(self, ref_state):
        """Create terminal model"""
        # Create cost model first with correct control dimension
        cost_model = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        
        # Terminal state cost with optional dt scaling (like Eagle MPC)
        if self.enable_dt_scaling:
            dt_scale = self.dt_mpc
        else:
            dt_scale = 1.0
            
        state_residual = crocoddyl.ResidualModelState(self.state, ref_state, self.actuation.nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(self.terminal_weights)
        state_cost = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        cost_model.addCost("terminal_state", state_cost, dt_scale)
        
        # Create differential action model with all required arguments
        diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_model
        )
        
        # Create integrated action model
        int_model = crocoddyl.IntegratedActionModelEuler(diff_model, self.dt_mpc)
        
        # Set control bounds directly on the terminal model (optional)
        int_model.u_lb = self.control_lower_bounds
        int_model.u_ub = self.control_upper_bounds
        
        return int_model
    
    def update_problem(self, current_state, reference_index=0):
        """
        Update MPC problem with current state and reference
        
        Args:
            current_state: Current system state
            reference_index: Index in reference trajectory
        """
        self.current_reference_index = reference_index
        
        # Create running models
        running_models = []
        for i in range(self.horizon):
            ref_idx = min(reference_index + i, len(self.reference_trajectory) - 1)
            ref_state = self.reference_trajectory[ref_idx]
            
            # Get reference control if available
            ref_control = None
            if self.reference_controls and ref_idx < len(self.reference_controls):
                ref_control = self.reference_controls[ref_idx]
            
            running_model = self._create_running_model(ref_state, ref_control)
            running_models.append(running_model)
        
        # Create terminal model
        terminal_ref_idx = min(reference_index + self.horizon, len(self.reference_trajectory) - 1)
        terminal_ref_state = self.reference_trajectory[terminal_ref_idx]
        terminal_model = self._create_terminal_model(terminal_ref_state)
        
        # Create shooting problem
        self.problem = crocoddyl.ShootingProblem(current_state, running_models, terminal_model)
        
        # Create solver
        self.solver = crocoddyl.SolverBoxFDDP(self.problem)
        # self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
        
        # Set solver parameters
        self.solver.th_stop = self.convergence_tolerance
        
        # Initialize with warm start if available
        if hasattr(self, 'xs_warm') and hasattr(self, 'us_warm'):
            if len(self.xs_warm) == len(running_models) + 1 and len(self.us_warm) == len(running_models):
                self.solver.xs = self.xs_warm
                self.solver.us = self.us_warm
    
    def solve(self, current_state, reference_index=0, warm_start=True):
        """
        Solve MPC optimization problem
        
        Args:
            current_state: Current system state
            reference_index: Index in reference trajectory
            warm_start: Whether to use warm start
            
        Returns:
            control_input: Optimal control input
            solve_info: Solver information dictionary
        """
        # Update problem
        self.update_problem(current_state, reference_index)
        
        # Set solver parameters
        self.solver.th_stop = self.convergence_tolerance
        
        # solve(init_xs, init_us, maxiter, ...)；空列表表示沿用当前候选（含 warm start）
        solved = self.solver.solve([], [], self.max_iterations)
        
        # Extract solution
        if solved or self.solver.iter > 0:
            control_input = self.solver.us[0].copy()
            
            # Store warm start for next iteration
            if warm_start and len(self.solver.xs) > 1:
                # Shift solution for warm start
                # Convert to list first, then concatenate
                xs_list = list(self.solver.xs)
                us_list = list(self.solver.us)
                self.xs_warm = xs_list[1:] + [xs_list[-1]]
                self.us_warm = us_list[1:] + [us_list[-1]]
        else:
            # Fallback: use zero control or previous control
            control_input = np.zeros(self.control_dim)
            if hasattr(self, 'last_control'):
                control_input = self.last_control.copy()
        
        # Store last control
        self.last_control = control_input.copy()
        
        # Prepare solve info
        solve_info = {
            'solved': solved,
            'iterations': self.solver.iter,
            'cost': self.solver.cost,
            'convergence': self.solver.th_stop,
            'solve_time': 0.0  # Would need timing implementation
        }
        
        return control_input, solve_info
    
    def get_predicted_trajectory(self):
        """
        Get predicted state and control trajectories
        
        Returns:
            xs: Predicted state trajectory
            us: Predicted control trajectory
        """
        if self.solver is not None:
            return self.solver.xs.copy(), self.solver.us.copy()
        else:
            return [], []

    def set_constant_tracking_reference(self, reference_state, extra_steps=2, reference_controls=None):
        """
        构造并设置「常值参考」轨迹，用于跟踪单个目标状态（setpoint regulation）。

        Args:
            reference_state: Pinocchio 状态向量 (nq+nv)，在视界内每步相同
            extra_steps: 在 horizon 基础上多出的点数，保证终端代价能取到合法下标（默认 2）
            reference_controls: 可选，与 reference_states 等长的控制参考列表
        """
        traj = make_constant_tracking_reference(reference_state, self.horizon, extra_steps=extra_steps)
        if reference_controls is not None:
            if len(reference_controls) == len(traj):
                controls = reference_controls
            else:
                u = np.asarray(reference_controls, dtype=np.float64).reshape(-1)
                controls = [u.copy() for _ in range(len(traj))]
            self.set_reference_trajectory(traj, controls)
        else:
            self.set_reference_trajectory(traj)


class PlatformParams:
    """Platform parameters class"""
    
    def __init__(self, config_dict):
        """Initialize from configuration dictionary"""
        # Extract platform parameters from config
        platform_config = config_dict.get('platform', config_dict)
        
        self.n_rotors = platform_config.get('n_rotors', 4)
        self.cf = platform_config.get('cf', 1.0)
        self.cm = platform_config.get('cm', 0.1)
        self.max_thrust = platform_config.get('max_thrust', 10.0)
        self.min_thrust = platform_config.get('min_thrust', 0.0)
        self.base_link_name = platform_config.get('base_link_name', 'base_link')
        
        # Parse rotor configurations
        self.rotors = self._parse_rotors(platform_config)
        
        # Create tau_f matrix (thrust to force/torque mapping)
        self.tau_f = self._create_tau_f_matrix(platform_config)
        
        print(f"Platform parameters loaded:")
        print(f"  n_rotors: {self.n_rotors}")
        print(f"  cf: {self.cf}")
        print(f"  cm: {self.cm}")
        print(f"  max_thrust: {self.max_thrust}")
        print(f"  min_thrust: {self.min_thrust}")
        print(f"  base_link_name: {self.base_link_name}")
        print(f"  rotors: {len(self.rotors)} configured")
    
    def _parse_rotors(self, platform_config):
        """Parse rotor configurations from platform config"""
        rotors = []
        
        # Check for $rotors key (with $ prefix as in the YAML)
        rotor_configs = platform_config.get('$rotors', platform_config.get('rotors', []))
        
        for i, rotor_config in enumerate(rotor_configs):
            if i >= self.n_rotors:
                break
                
            # Support both legacy keys (translation/spin_direction)
            # and alternative keys (position/direction) from custom YAML files.
            position = rotor_config.get('translation', rotor_config.get('position', [0, 0, 0]))
            spin_direction = rotor_config.get('spin_direction', rotor_config.get('direction', 1))
            if isinstance(spin_direction, list):
                spin_direction = spin_direction[0] if len(spin_direction) > 0 else 1

            rotor = {
                'index': i,
                'translation': position,
                'orientation': rotor_config.get('orientation', [0, 0, 0, 1]),
                'spin_direction': spin_direction
            }
            rotors.append(rotor)
            
        # If no rotors configured, use default quadrotor layout
        if not rotors:
            arm_length = 0.171  # From s500.yaml
            default_rotors = [
                {'index': 0, 'translation': [arm_length, -arm_length, 0.045], 'spin_direction': -1},
                {'index': 1, 'translation': [-arm_length, arm_length, 0.045], 'spin_direction': -1},
                {'index': 2, 'translation': [arm_length, arm_length, 0.045], 'spin_direction': 1},
                {'index': 3, 'translation': [-arm_length, -arm_length, 0.045], 'spin_direction': 1},
            ]
            rotors = default_rotors[:self.n_rotors]
            
        return rotors
    
    def _create_tau_f_matrix(self, config_dict):
        """
        机体螺旋桨推力到空间螺旋力 (6 × n_rotors)，与 Crocoddyl ActuationModelMultiCopterBase 约定一致。

        控制量 u_i 必须与 YAML 中 min/max_thrust、control_reference 同单位——**每旋翼推力 (N)**，
        不是 Gazebo 里的 omega^2 系数；因此 **不能** 把 cf (thrust = cf * omega^2 中的 cf) 乘在 Fz 行上。

        几何与 scripts/trajectory_optimization/s500_trajectory_planner.py、常见四旋翼建模一致：
        - Fz: 各电机 z 向推力，系数为 1
        - Mx, My: 力臂 × 推力，即 y、-x（米）
        - Mz: 反扭矩与推力比，常用 spin * (cm/cf)（cf、cm 来自电机/桨模型）
        """
        n_rotors = self.n_rotors
        cf = float(self.cf)
        cm = float(self.cm)
        if cf <= 0.0:
            raise ValueError('platform cf must be > 0 for tau_f Mz scaling (cm/cf)')

        # Initialize tau_f matrix (6 x n_rotors)
        tau_f = np.zeros((6, n_rotors))

        # Use parsed rotor configurations
        for i, rotor in enumerate(self.rotors):
            if i >= n_rotors:
                break

            pos = np.asarray(rotor['translation'], dtype=np.float64).reshape(3)
            direction = float(rotor['spin_direction'])

            tau_f[0, i] = 0.0
            tau_f[1, i] = 0.0
            tau_f[2, i] = 1.0
            tau_f[3, i] = pos[1]
            tau_f[4, i] = -pos[0]
            tau_f[5, i] = direction * (cm / cf)

        print(f"tau_f matrix created:")
        print(f"  Shape: {tau_f.shape}")
        print(f"  Force coefficients (Fz): {tau_f[2, :]}")
        print(f"  Moment coefficients (Mx): {tau_f[3, :]}")
        print(f"  Moment coefficients (My): {tau_f[4, :]}")
        print(f"  Moment coefficients (Mz): {tau_f[5, :]}")

        return tau_f


def make_constant_tracking_reference(reference_state, horizon, extra_steps=2):
    """
    生成用于 MPC 跟踪的常值参考轨迹（每步同一目标状态）。

    update_problem 会用到下标 reference_index + horizon，故列表长度至少为 horizon + 1；
    默认生成长度 horizon + extra_steps，避免末端索引越界。

    Args:
        reference_state: 目标状态 (nq+nv) 一维数组
        horizon: MPC 预测步数（与 CrocoddylMPCController.horizon 一致）
        extra_steps: 额外点数（默认 2）

    Returns:
        list[np.ndarray]: 参考状态序列，可直接传入 set_reference_trajectory
    """
    length = int(horizon) + int(extra_steps)
    x = np.asarray(reference_state, dtype=np.float64).reshape(-1)
    return [x.copy() for _ in range(length)]


def load_mpc_config(config_path):
    """Load MPC configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('mpc_controller', {})


def load_platform_config(config_path):
    """Load platform configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('platform', {})


def _ee_frame_id(robot_model):
    for name in ('gripper_link', 'link_4'):
        if robot_model.existFrame(name):
            return robot_model.getFrameId(name)
    return robot_model.getFrameId(robot_model.frames[-1].name)


def _compute_mpc_plot_series(robot_model, X):
    """
    从 Pinocchio 状态矩阵 X (T, nq+nv) 提取基座 / 末端 / 关节序列。
    角度、角速度输出为度与 deg/s；关节角为度数。
    """
    X = np.asarray(X, dtype=np.float64)
    T, nx = X.shape
    nq = robot_model.nq
    nv = robot_model.nv
    n_arm = nq - 7
    q_all = X[:, :nq]
    v_all = X[:, nq:nq + nv]

    # --- base: world position, world linear vel, euler (deg), omega (deg/s)
    p_b = q_all[:, :3]
    quat = q_all[:, 3:7]
    eul_b = np.zeros((T, 3))
    for t in range(T):
        eul_b[t] = R.from_quat(quat[t]).as_euler('xyz', degrees=True)
    vlin_b = v_all[:, :3]
    w_b = np.rad2deg(v_all[:, 3:6])

    # --- EE (gripper): world pose & twist at frame
    data = robot_model.createData()
    fid = _ee_frame_id(robot_model)
    p_ee = np.zeros((T, 3))
    eul_ee = np.zeros((T, 3))
    vlin_ee = np.zeros((T, 3))
    w_ee = np.zeros((T, 3))
    for t in range(T):
        q = q_all[t]
        v = v_all[t]
        pin.forwardKinematics(robot_model, data, q, v)
        pin.updateFramePlacements(robot_model, data)
        oMf = data.oMf[fid]
        p_ee[t] = oMf.translation
        eul_ee[t] = R.from_matrix(oMf.rotation).as_euler('xyz', degrees=True)
        fv = pin.getFrameVelocity(robot_model, data, fid, pin.ReferenceFrame.WORLD)
        vlin_ee[t] = fv.linear
        w_ee[t] = np.rad2deg(fv.angular)

    qj_deg = np.rad2deg(q_all[:, 7:7 + n_arm])
    vj_deg = np.rad2deg(v_all[:, 6:6 + n_arm])

    return {
        'p_b': p_b,
        'vlin_b': vlin_b,
        'eul_b': eul_b,
        'w_b': w_b,
        'p_ee': p_ee,
        'vlin_ee': vlin_ee,
        'eul_ee': eul_ee,
        'w_ee': w_ee,
        'qj_deg': qj_deg,
        'vj_deg': vj_deg,
        'n_arm': n_arm,
    }


def _plot_mpc_test(
    robot_model,
    dt,
    x_goal,
    *,
    open_loop,
    xs_open=None,
    us_open=None,
    x_hist=None,
    u_hist=None,
    save_path=None,
    show=True,
):
    """
    3×4 子图：
      行1 — 基座：位置、速度、欧拉角(°)、角速度(°/s)
      行2 — 末端：位置、速度、欧拉角(°)、角速度(°/s)
      行3 — 关节角(°)、关节角速度(°/s)、关节力矩、四旋翼推力
    """
    import matplotlib.pyplot as plt

    g = np.asarray(x_goal[:3], dtype=float)

    if open_loop and xs_open is not None:
        X = np.asarray(xs_open, dtype=np.float64)
        title_prefix = 'Open-loop prediction'
        t_x = np.arange(X.shape[0]) * dt
        U = np.asarray(us_open, dtype=np.float64) if us_open is not None else None
    elif x_hist is not None:
        X = np.asarray(x_hist, dtype=np.float64)
        title_prefix = 'Closed-loop'
        t_x = np.arange(X.shape[0]) * dt
        U = np.asarray(u_hist, dtype=np.float64) if u_hist is not None else None
    else:
        raise ValueError('No trajectory data to plot')

    sig = _compute_mpc_plot_series(robot_model, X)
    n_arm = sig['n_arm']
    joint_labels = [f'q{i + 1}' for i in range(n_arm)]
    vel_labels = [f'qd{i + 1}' for i in range(n_arm)]

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=False)
    fig.suptitle(f'{title_prefix} (angles in deg, rates in deg/s)', fontsize=11)

    xyz = ('x', 'y', 'z')
    # Row 1 — base
    row1_meta = [
        ('p_b', 'm', 'base pos'),
        ('vlin_b', 'm/s', 'base vel'),
        ('eul_b', 'deg', 'base euler'),
        ('w_b', 'deg/s', 'base omega'),
    ]
    for i, (k, unit, ylab) in enumerate(row1_meta):
        ax = axes[0, i]
        Y = sig[k]
        for j in range(3):
            ax.plot(t_x, Y[:, j], label=xyz[j])
        if i == 0:
            for j in range(3):
                ax.axhline(g[j], color=f'C{j}', ls='--', alpha=0.35)
        ax.set_ylabel(f'{ylab} ({unit})')
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # Row 2 — EE
    for i, (key, ylab) in enumerate(
        zip(
            ['p_ee', 'vlin_ee', 'eul_ee', 'w_ee'],
            ['EE pos', 'EE vel', 'EE euler', 'EE omega'],
        )
    ):
        ax = axes[1, i]
        units = ['m', 'm/s', 'deg', 'deg/s'][i]
        Y = sig[key]
        for j in range(3):
            ax.plot(t_x, Y[:, j], label=xyz[j])
        ax.set_ylabel(f'{ylab} ({units})')
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # Row 3 — joints & controls
    ax = axes[2, 0]
    for j in range(n_arm):
        ax.plot(t_x, sig['qj_deg'][:, j], label=joint_labels[j])
    ax.set_ylabel('joint q (deg)')
    ax.set_title('joint angle')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 1]
    for j in range(n_arm):
        ax.plot(t_x, sig['vj_deg'][:, j], label=vel_labels[j])
    ax.set_ylabel('joint qd (deg/s)')
    ax.set_title('joint vel')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 2]
    if U is not None and U.shape[1] >= 6:
        t_u = (np.arange(U.shape[0]) + 0.5) * dt
        ax.plot(t_u, U[:, 4], label='tau j1')
        ax.plot(t_u, U[:, 5], label='tau j2')
    ax.set_ylabel('joint torque (Nm)')
    ax.set_title('arm torque')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 3]
    if U is not None and U.shape[1] >= 4:
        t_u = (np.arange(U.shape[0]) + 0.5) * dt
        for j in range(4):
            ax.plot(t_u, U[:, j], label=f'rot{j + 1}')
    ax.set_ylabel('thrust (N)')
    ax.set_title('rotor thrust')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    for j in range(4):
        axes[2, j].set_xlabel('time (s)')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Figure saved: {save_path}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    """
    直接运行本文件时的冒烟测试：加载 s500_uam + 默认 crocoddyl YAML，
    构造常值参考轨迹并求解 MPC（可选闭环仿真若干步）。

    用法:
        python3 crocoddyl_mpc_controller.py
        python3 crocoddyl_mpc_controller.py --dt 0.02 --steps 80 --save /tmp/mpc_test.png
    """
    root = os.path.abspath(os.path.dirname(__file__))
    urdf = os.path.join(root, 'models/urdf/s500_uam_simple.urdf')
    mpc_yaml = os.path.join(root, 'config/yaml/mpc/s500_uam_mpc_crocoddyl.yaml')
    platform_yaml = os.path.join(root, 'config/yaml/multicopter/s500.yaml')

    parser = argparse.ArgumentParser(description='Test CrocoddylMPCController with default s500_uam YAML.')
    parser.add_argument('--dt', type=float, default=0.01, help='MPC 步长 (s)')
    parser.add_argument(
        '--steps',
        type=int,
        default=300,
        help='闭环仿真步数；1 表示只解一次并打印，>1 时用第一步动作模型积分',
    )
    parser.add_argument('--no-plot', action='store_true', help='不绘图')
    parser.add_argument('--save', type=str, default=None, help='将图保存为 PNG 路径')
    parser.add_argument('--no-show', action='store_true', help='保存图形但不弹窗（适合无显示器环境）')
    args = parser.parse_args()

    for path, label in ((urdf, 'URDF'), (mpc_yaml, 'MPC YAML'), (platform_yaml, 'platform YAML')):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{label} 不存在: {path}')

    robot_model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    mpc_config = load_mpc_config(mpc_yaml)
    platform_params = PlatformParams(load_platform_config(platform_yaml))
    mpc = CrocoddylMPCController(robot_model, platform_params, mpc_config, dt_mpc=args.dt)

    nqnv = robot_model.nq + robot_model.nv
    x0 = np.zeros(nqnv)
    x0[6] = 1

    x_goal = x0.copy()
    x_goal[0] = 0.5
    x_goal[1] = 0.0
    x_goal[2] = -1.0

    u_ref = mpc_config.get('control_reference')
    if u_ref is not None:
        u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
        mpc.set_constant_tracking_reference(x_goal, extra_steps=2, reference_controls=u_ref)
    else:
        mpc.set_constant_tracking_reference(x_goal, extra_steps=2)

    n_steps = max(1, args.steps)
    x = x0.copy()
    xs_open = None
    x_hist = [x0.copy()]
    u_hist = []

    for k in range(n_steps):
        u, info = mpc.solve(x, reference_index=0, warm_start=(k > 0))
        print(
            f'step {k}: solved={info["solved"]} iters={info["iterations"]} '
            f'cost={info["cost"]:.6g} u0[:4]={u[:4]}'
        )
        if n_steps == 1:
            xs_open = np.array([np.array(xi) for xi in mpc.solver.xs])
            err = np.linalg.norm(xs_open[:, :3] - x_goal[:3], axis=1)
            print(f'  open-loop position error: init={err[0]:.4f} min={err.min():.4f} (node {err.argmin()})')
            break
        u_hist.append(u.copy())
        m = mpc.problem.runningModels[0]
        d = m.createData()
        m.calc(d, x, u)
        x = d.xnext.copy()
        x_hist.append(x.copy())
        print(f'  integrated p={x[:3]}  |p-p_goal|={np.linalg.norm(x[:3] - x_goal[:3]):.4f}')

    if not args.no_plot:
        us_open = None
        if n_steps == 1 and xs_open is not None:
            us_open = np.array([np.array(ui) for ui in mpc.solver.us])
        _plot_mpc_test(
            robot_model,
            args.dt,
            x_goal,
            open_loop=(n_steps == 1),
            xs_open=xs_open,
            us_open=us_open,
            x_hist=x_hist if n_steps > 1 else None,
            u_hist=u_hist if n_steps > 1 else None,
            save_path=args.save,
            show=not args.no_show,
        )


if __name__ == '__main__':
    main()
