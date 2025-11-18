"""
Copyright © 2024 Hs293Go

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import minsnap_trajectories as ms


def main():
    refs = [
        ms.Waypoint(
            time=0.0,
            position=np.array([-1.5, 0.0, 1.5]),
        ),
        ms.Waypoint(  # Any higher-order derivatives
            time=2.0,
            position=np.array([0.0, 0.0, 1.2]),
            # velocity=np.array([1.0, 0.0, 0.0]),
            # acceleration=np.array([0.0, 0.0, 0.0]),
        ),
        ms.Waypoint(  # Potentially leave intermediate-order derivatives unspecified
            time=4.0,
            position=np.array([1.5, 0.0, 1.5]),
            # velocity=np.array([0.0, 0.0, 0.0]),
            # jerk=np.array([0.1, 0.0, 0.2]),
        ),
    ]

    polys = ms.generate_trajectory(
        refs,
        degree=8,  # Polynomial degree
        idx_minimized_orders=(3, 4),  # Minimize derivatives in these orders (>= 2)
        num_continuous_orders=4,  # Constrain continuity of derivatives up to order (>= 3)
        # 设置为4可以保证jerk（3阶导数）也连续，避免jerk突变
        algorithm="closed-form",  # Or "constrained"
    )

    t = np.linspace(0, 6, 100)
    #  Sample up to the 5th order (snap's derivative) -----v
    derivatives = ms.compute_trajectory_derivatives(polys, t, 6)
    position = derivatives[0]
    velocity = derivatives[1]
    acceleration = derivatives[2]
    jerk = derivatives[3]
    snap = derivatives[4]
    snap_derivative = derivatives[5]

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 3D trajectory plot (spans 2 rows, 1 column)
    ax_3d = fig.add_subplot(gs[0:2, 0], projection='3d')
    ax_3d.plot(position[:, 0], position[:, 1], position[:, 2], 
              label="Position Trajectory", linewidth=2)
    
    position_waypoints = np.array([it.position for it in refs])
    ax_3d.plot(
        position_waypoints[:, 0],
        position_waypoints[:, 1],
        position_waypoints[:, 2],
        "ro",
        markersize=8,
        label="Position Waypoints",
    )
    # ax_3d.quiver(
    #     *refs[1].position,
    #     *refs[1].velocity,
    #     color="g",
    #     arrow_length_ratio=0.15,
    #     label="Velocity specified at waypoint 1",
    # )
    # ax_3d.set_zlim(8, 12)
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("3D Trajectory", fontsize=12)
    ax_3d.legend(loc="upper right")
    
    # Create subplots for all derivatives (3x2 grid in remaining space)
    axes = []
    axes.append(fig.add_subplot(gs[0, 1]))  # Position
    axes.append(fig.add_subplot(gs[0, 2]))  # Velocity
    axes.append(fig.add_subplot(gs[1, 1]))  # Acceleration
    axes.append(fig.add_subplot(gs[1, 2]))  # Jerk
    axes.append(fig.add_subplot(gs[2, 1]))  # Snap
    axes.append(fig.add_subplot(gs[2, 2]))  # Snap's derivative
    
    fig.suptitle("Trajectory Analysis", fontsize=16, y=0.98)

    # Position
    ax = axes[0]
    ax.plot(t, position[:, 0], label="X", linewidth=2)
    ax.plot(t, position[:, 1], label="Y", linewidth=2)
    ax.plot(t, position[:, 2], label="Z", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[1]
    ax.plot(t, velocity[:, 0], label="X", linewidth=2)
    ax.plot(t, velocity[:, 1], label="Y", linewidth=2)
    ax.plot(t, velocity[:, 2], label="Z", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Acceleration
    ax = axes[2]
    ax.plot(t, acceleration[:, 0], label="X", linewidth=2)
    ax.plot(t, acceleration[:, 1], label="Y", linewidth=2)
    ax.plot(t, acceleration[:, 2], label="Z", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Acceleration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Jerk
    ax = axes[3]
    ax.plot(t, jerk[:, 0], label="X", linewidth=2)
    ax.plot(t, jerk[:, 1], label="Y", linewidth=2)
    ax.plot(t, jerk[:, 2], label="Z", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk (m/s³)")
    ax.set_title("Jerk")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Snap
    ax = axes[4]
    ax.plot(t, snap[:, 0], label="X", linewidth=2)
    ax.plot(t, snap[:, 1], label="Y", linewidth=2)
    ax.plot(t, snap[:, 2], label="Z", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Snap (m/s⁴)")
    ax.set_title("Snap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Snap's derivative (5th order)
    ax = axes[5]
    ax.plot(t, snap_derivative[:, 0], label="X", linewidth=2)
    ax.plot(t, snap_derivative[:, 1], label="Y", linewidth=2)
    ax.plot(t, snap_derivative[:, 2], label="Z", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Snap Derivative (m/s⁵)")
    ax.set_title("Snap's Derivative")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Maximize window (Windows)
    try:
        mngr = plt.get_current_fig_manager()
        mngr.window.state('zoomed')  # Windows
    except:
        try:
            mngr.resize(*mngr.window.maxsize())  # Linux
        except:
            pass  # macOS or other
    
    plt.show()
    # try:
    #     fig.savefig("example/minsnap_trajectories_example.png")
    # except FileNotFoundError:
    #     plt.show()


if __name__ == "__main__":
    main()
