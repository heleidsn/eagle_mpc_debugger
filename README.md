# Eagle MPC Debugger

Eagle MPC Debugger is a tool for debugging and visualizing MPC controllers. It provides an interactive graphical interface that allows users to monitor and adjust MPC controller states and parameters in real-time.

This tool is writting for eagle_mpc project. It can also be used for other MPC frameworks, such as acados.

## Demo

[![Eagle MPC Debugger Demo](https://img.youtube.com/vi/ga8kUAdP3Mg/maxresdefault.jpg)](https://www.youtube.com/watch?v=ga8kUAdP3Mg)

## Features

1. Real-time State Monitoring

   - Position (x, y, z)
   - Orientation (roll, pitch, yaw)
   - Linear Velocity (vx, vy, vz)
   - Angular Velocity (wx, wy, wz)
2. Interactive Control

   - Direct state adjustment via sliders
   - Precise control through numerical input
   - Real-time state feedback
3. Visualization

   - State trajectory plots
   - Attitude angle plots
   - Linear velocity plots
   - Angular velocity plots
   - Control input plots
   - MPC solving time statistics

## Dependencies

```bash
pip install numpy scipy matplotlib PyQt5
```

## Usage

1. Standalone Mode (without ROS):

```bash
python eagle_mpc_debugger.py
```

2. ROS Mode:
   Set `using_ros` to `True`

## Configuration Files

Main configuration files are located in:

- MPC Configuration: `eagle_mpc_yaml/mpc/`
- Trajectory Configuration: `eagle_mpc_yaml/trajectories/`

## TODO

1. Adding New State Variables for quadrotor with robotic arm.
2. Adding button to change current state to reference state directly.

## Contributing

Issues and Pull Requests are welcome.

## License

MIT License
