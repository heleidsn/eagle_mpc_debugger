"""
Analytic trajectories (figure-8, minimum snap) for run_controller reference generation.
State layout matches eagle_mpc / run_controller: quadrotor 13-dim (nq=7+nv=6), UAM 17-dim (nq=9+nv=8).
"""
import numpy as np
from tf.transformations import quaternion_from_euler, quaternion_matrix

try:
    import minsnap_trajectories as ms

    MINSNAP_AVAILABLE = True
except ImportError:
    MINSNAP_AVAILABLE = False
    ms = None


def quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))
    return np.array([qx, qy, qz, qw], dtype=float)


def pack_multibody_state(
    template_state: np.ndarray,
    robot_name: str,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    vel_world: np.ndarray,
    omega_world: np.ndarray = None,
) -> np.ndarray:
    s = np.array(template_state, dtype=float, copy=True)
    if omega_world is None:
        omega_world = np.zeros(3, dtype=float)
    pos = np.asarray(pos, dtype=float).reshape(3)
    quat_xyzw = np.asarray(quat_xyzw, dtype=float).reshape(4)
    vel_world = np.asarray(vel_world, dtype=float).reshape(3)

    if robot_name == "s500_uam":
        nq = 9
    else:
        nq = 7

    s[0:3] = pos
    s[3:7] = quat_xyzw

    if robot_name == "s500_uam":
        # Keep arm joints from template at indices 7:9
        s[9:12] = vel_world
        s[12:15] = omega_world
        s[15:17] = 0.0
    else:
        R = quaternion_matrix([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]])[:3, :3]
        v_body = R.T @ vel_world
        s[7:10] = v_body
        s[10:13] = omega_world
    return s


def states_from_flat_trajectory(template_state: np.ndarray, robot_name: str, traj_dict: dict) -> list:
    """Build traj_state_ref list from positions/velocities in world frame."""
    positions = traj_dict["positions"]
    velocities = traj_dict["velocities"]
    n = len(positions)
    states = []
    prev_yaw = 0.0
    for i in range(n):
        p = positions[i]
        v = velocities[i]
        vh = float(np.linalg.norm(v[:2]))
        if vh > 1e-3:
            yaw = float(np.arctan2(v[1], v[0]))
            prev_yaw = yaw
        else:
            yaw = prev_yaw
        q = quat_xyzw_from_yaw(yaw)
        states.append(pack_multibody_state(template_state, robot_name, p, q, v))
    return states


def generate_figure8_trajectory(center_pos, radius, height, duration, dt=0.01, ramp_time=1.0):
    """
    Figure-8 in XY at fixed z = height. Ported from hextor_mpc_test (smooth ramp in/out).
    """
    center_pos = np.asarray(center_pos, dtype=float).reshape(3)
    times = np.arange(0, duration + dt / 2, dt)
    times = times[times <= duration]
    if len(times) == 0:
        times = np.array([0.0, duration])
    if times[-1] < duration - dt / 2:
        times = np.append(times, duration)

    ramp_time = float(min(ramp_time, duration / 2.0))
    effective_duration = duration - ramp_time
    omega = 2 * np.pi / max(effective_duration, 1e-6)

    n_points = len(times)
    positions = np.zeros((n_points, 3))
    velocities = np.zeros((n_points, 3))
    scaled_t_prev = 0.0

    for i, t in enumerate(times):
        if t < ramp_time:
            s = 0.5 * (1.0 - np.cos(np.pi * t / ramp_time))
        elif t >= duration - ramp_time:
            t_remaining = duration - t
            s = 0.5 * (1.0 - np.cos(np.pi * t_remaining / ramp_time))
        else:
            s = 1.0

        d_scaled_t_dt = s * omega

        if i == 0:
            scaled_t = 0.0
            dt_actual = dt
        else:
            dt_actual = times[i] - times[i - 1]
            scaled_t = scaled_t_prev + d_scaled_t_dt * dt_actual
        scaled_t_prev = scaled_t

        positions[i, 0] = center_pos[0] + radius * np.sin(scaled_t)
        positions[i, 1] = center_pos[1] + radius * np.sin(scaled_t) * np.cos(scaled_t)
        positions[i, 2] = height

        velocities[i, 0] = radius * np.cos(scaled_t) * d_scaled_t_dt
        velocities[i, 1] = radius * (np.cos(scaled_t) ** 2 - np.sin(scaled_t) ** 2) * d_scaled_t_dt
        velocities[i, 2] = 0.0

    if n_points > 0:
        scaled_t_final = 2 * np.pi
        positions[-1, 0] = center_pos[0] + radius * np.sin(scaled_t_final)
        positions[-1, 1] = center_pos[1] + radius * np.sin(scaled_t_final) * np.cos(scaled_t_final)
        positions[-1, 2] = height
        velocities[-1, :] = 0.0

    return {"times": times, "positions": positions, "velocities": velocities}


def default_min_snap_waypoints():
    """Simple closed path in XY with mild Z change (for testing)."""
    return [
        {"time": 0.0, "position": np.array([0.0, 0.0, 1.2]), "velocity": np.zeros(3)},
        {"time": 2.0, "position": np.array([2.0, 0.0, 1.2])},
        {"time": 4.0, "position": np.array([2.0, 2.0, 3.0])},
        {"time": 6.0, "position": np.array([0.0, 2.0, 1.2])},
        {"time": 8.0, "position": np.array([0.0, 0.0, 1.2]), "velocity": np.zeros(3)},
    ]


def generate_minimum_snap_trajectory(
    waypoints,
    dt=0.01,
    degree=8,
    idx_minimized_orders=(3, 4),
    num_continuous_orders=4,
    algorithm="closed-form",
):
    """
    Minimum-snap trajectory through waypoints. Requires minsnap_trajectories + scipy.
    """
    if not MINSNAP_AVAILABLE:
        raise ImportError(
            "minsnap_trajectories (and scipy) are required for minimum_snap / min_snap trajectory. "
            "e.g. pip install minsnap_trajectories scipy"
        )

    ms_waypoints = []
    for wp in waypoints:
        kwargs = {"time": wp["time"], "position": np.array(wp["position"], dtype=float)}
        if "velocity" in wp:
            kwargs["velocity"] = np.array(wp["velocity"], dtype=float)
        if "acceleration" in wp:
            kwargs["acceleration"] = np.array(wp["acceleration"], dtype=float)
        if "jerk" in wp:
            kwargs["jerk"] = np.array(wp["jerk"], dtype=float)
        ms_waypoints.append(ms.Waypoint(**kwargs))

    polys = ms.generate_trajectory(
        ms_waypoints,
        degree=degree,
        idx_minimized_orders=idx_minimized_orders,
        num_continuous_orders=num_continuous_orders,
        algorithm=algorithm,
    )

    total_duration = float(waypoints[-1]["time"])
    times = np.arange(0, total_duration + dt / 2, dt)
    times = times[times <= total_duration]
    if len(times) == 0:
        times = np.array([0.0, total_duration])
    if times[-1] < total_duration - dt / 2:
        times = np.append(times, total_duration)

    derivatives = ms.compute_trajectory_derivatives(polys, times, 2)
    positions = derivatives[0]
    velocities = derivatives[1]

    return {
        "times": times,
        "positions": positions,
        "velocities": velocities,
    }
