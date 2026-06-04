# Eagle MPC Debugger

> ⚠️ **Project status: deprecated / no longer maintained.**
>
> This package has been **superseded by [`eagle-mpc-python`](https://github.com/heleidsn/eagle-mpc-python)**, which provides a cleaner, ROS-independent Python implementation of the same trajectory-optimization and MPC stack. New development, bug fixes, and features will happen there. This repository is kept **for reference only** and will not receive further updates. Please migrate to `eagle-mpc-python` for any new work.

---

Eagle MPC Debugger is a ROS (catkin) toolbox for **developing, simulating, debugging, and visualizing MPC controllers** for aerial vehicles and aerial manipulators. It was built around the [`eagle_mpc`](https://github.com/heleidsn) / [`crocoddyl`](https://github.com/loco-3d/crocoddyl) optimal-control libraries and integrates with PX4 SITL + Gazebo and MAVROS.

It supports two main robot platforms:

- **`s500`** — an S500 quadrotor.
- **`s500_uam`** — an S500 quadrotor equipped with a robotic arm (Unmanned Aerial Manipulator).
- (`hexacopter370_flying_arm_3` is partially supported for offline planning.)

## Demo

[![Eagle MPC Debugger Demo](https://img.youtube.com/vi/ga8kUAdP3Mg/maxresdefault.jpg)](https://www.youtube.com/watch?v=ga8kUAdP3Mg)

## Features

- **Trajectory optimization** with `eagle_mpc` (SbFDDP) or pure `crocoddyl` (BoxFDDP) backends.
- **MPC tracking controllers** (Rail / Weighted / Carrot MPC) running against PX4 SITL or a built-in numeric simulator.
- **L1 adaptive control** add-on (multiple versions in `l1_control/`).
- **PX4 + Gazebo SITL** launch files for both ideal and realistic motor models.
- **Interactive GUIs** for launching, controlling, monitoring, and plotting.
- **Rich plotting / data recording** for states, controls, gripper pose, and solver performance.
- **Robotic-arm control** via `ros_control` (position / velocity / effort modes).

## Repository Layout

| Path | Description |
|------|-------------|
| `run_main_gui.py` | Main control-center GUI: launch simulation, planning, MPC controller, services. |
| `run_debugger.py` | MPC debug interface (inspect/adjust states and references before deploying). |
| `run_planning.py` | Offline trajectory optimization for a single `dt`. |
| `run_planning_different_dt.py` | Trajectory optimization swept over multiple `dt` values. |
| `run_controller.py` | Main online MPC trajectory publisher / controller node. |
| `run_numeric_sim.py` | Standalone numeric closed-loop MPC simulation (no Gazebo). |
| `run_mpc_single_step.py` | Single-step MPC solve for inspection/debugging. |
| `run_plotter.py` | Live PyQtGraph plotting GUI for ROS MPC topics. |
| `run_takeoff_move_land.py` | Simple takeoff → move → land demo script. |
| `trajectory_optimization_crocoddyl.py` | Standalone Crocoddyl-based trajectory optimization example. |
| `crocoddyl_mpc_controller.py` | Pure Crocoddyl MPC controller (no `eagle_mpc` dependency). |
| `launch/` | PX4/Gazebo SITL, RViz, arm-controller, and VRPN/Vicon launch files. |
| `config/yaml/` | `mpc/`, `trajectories/`, and `multicopter/` configuration files. |
| `config/rviz/`, `config/plotjuggler/` | RViz and PlotJuggler layouts. |
| `models/` | URDF/xacro and Gazebo SDF models (see model variants below). |
| `worlds/` | Gazebo worlds (`empty`, `table_beer`, `table_beer_with_stand`, ...). |
| `utils/` | Problem creation, conversions, analytic trajectories, plotting helpers. |
| `scripts/` | Helper nodes (groundtruth, TF, arm trajectory, PID) and offline optimization examples. |
| `l1_control/` | L1 adaptive controller implementations. |

### Gazebo model variants (`models/sdf/s500_uam/`)

- `s500.sdf` — **real** S500 model, includes motor time constants.
- `s500_ideal.sdf` — **ideal** model, no motor time constant and higher motor thrust; better suited for agile flight.
- `s500_uam_real.sdf` / `s500_uam_ideal.sdf` — UAM (with arm) counterparts.
- `s500_uam_no_friction.sdf`, `s500_uam_no_motor_constant.sdf`, `s500_uam_camera.sdf` — additional UAM variants.

## Dependencies

System / Python:

```bash
pip install numpy scipy matplotlib PyQt5 pyqtgraph
```

ROS / optimal-control stack (expected in the same catkin workspace or environment):

- ROS 1 (Noetic, Python 3) + catkin
- `eagle_mpc`, `eagle_mpc_msgs`, `eagle_mpc_viz`
- `crocoddyl`, `pinocchio`, `example_robot_data`, `gepetto` (for visualization)
- PX4-Autopilot (SITL) + `mavros`
- `ros_control` (`controller_manager`, `position/velocity/effort_controllers`, `joint_state_controller`)

Build the package inside your catkin workspace:

```bash
cd ~/catkin_eagle_mpc
catkin build eagle_mpc_debugger   # or catkin_make
source devel/setup.bash
```

## Common Usage

### 1. Launch the main GUI (recommended entry point)

```bash
rosrun eagle_mpc_debugger run_main_gui.py
# or
python3 run_main_gui.py
```

From the GUI you can: select the robot/trajectory, run planning (with `eagle_mpc` or `crocoddyl`),
start Gazebo SITL (choosing world + model type), start/stop the MPC controller, and trigger services.

### 2. Start PX4 + Gazebo SITL

S500 quadrotor:

```bash
roslaunch eagle_mpc_debugger s500_sitl.launch world_name:=empty model_type:=real
# agile flight with the ideal (high-thrust, no motor time constant) model:
roslaunch eagle_mpc_debugger s500_sitl.launch world_name:=empty model_type:=ideal
```

S500 aerial manipulator:

```bash
roslaunch eagle_mpc_debugger s500_uam_sitl.launch \
    world_name:=table_beer_with_stand model_type:=real arm_control_mode:=position
```

Key launch arguments:

- `world_name` — Gazebo world file in `worlds/` (without extension).
- `model_type` — `real` (with motor time constant) or `ideal` (higher thrust, no motor lag); the UAM launch also supports `no_friction`, `no_motor_constant`, `camera`.
- `arm_control_mode` (UAM) — `position`, `velocity`, or `effort`.

### 3. Offline trajectory optimization (planning)

```bash
# single dt
python3 run_planning.py --robot s500 --trajectory hover --dt 50 --planner eagle_mpc
python3 run_planning.py --robot s500 --trajectory displacement --dt 20 --planner crocoddyl

# sweep over multiple dt values
python3 run_planning_different_dt.py --robot s500_uam --trajectory catch_vicon --dt 10
```

Useful flags: `--use-squash`, `--gepetto-vis` (requires `gepetto-gui` running), `--save`.
Trajectory names map to `config/yaml/trajectories/<robot>_<trajectory>.yaml`.

### 4. Run the online MPC controller

```bash
# parameters are typically set via the GUI or rosparam, then:
python3 run_controller.py
```

### 5. Numeric (Gazebo-free) closed-loop simulation

```bash
python3 run_numeric_sim.py --config numeric_sim_config_hover.yaml
```

See `numeric_sim_config_*.yaml` for ready-made hover / regulation / catch / crocoddyl setups.

### 6. Debugging and visualization

```bash
python3 run_debugger.py        # MPC debug interface
python3 run_plotter.py         # live ROS MPC data plots
```

PlotJuggler layouts are available in `config/plotjuggler/`, and RViz layouts in `config/rviz/`.

## Configuration

- **MPC configs:** `config/yaml/mpc/` (e.g. `s500_mpc.yaml`, `s500_uam_mpc.yaml`, `s500_uam_mpc_crocoddyl.yaml`).
- **Trajectory configs:** `config/yaml/trajectories/` (e.g. `s500_hover.yaml`, `s500_displacement.yaml`, `s500_uam_catch_vicon.yaml`).
- **Platform config:** `config/yaml/multicopter/s500.yaml`.

## Robotic Arm Note

The default arm state publish rate is ~62 Hz, which is insufficient for high-frequency (especially torque) control. To raise it, change the USB `latency_timer` from `16ms` (default) to `1ms`:

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   # 16
echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   # 1
```

You can then push the rate up to ~1000 Hz with baud rate `3000000`.
Reference: <https://emanual.robotis.com/docs/en/parts/interface/u2d2/#linux>

## Migration

Because this repository is no longer maintained, please use **[`eagle-mpc-python`](https://github.com/heleidsn/eagle-mpc-python)** going forward. Related repositories:

- `eagle-mpc-python` — successor project (actively maintained).
- `eagle_mpc_ros` — ROS integration for the eagle_mpc stack.

## License

MIT License.
