#!/usr/bin/env python3
"""
S500 Trajectory Planning Usage Example

This script demonstrates how to use S500TrajectoryPlanner for trajectory planning
"""

import numpy as np
import os
from s500_trajectory_planner import S500TrajectoryPlanner


def create_square_trajectory():
    """Create square trajectory"""
    # State vector: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    waypoints = []
    
    # Starting point: ground level
    start = np.zeros(13)
    start[6] = 1.0  # qw = 1 (unit quaternion)
    waypoints.append(start)
    
    # Ascend to 2m height
    wp1 = start.copy()
    wp1[2] = 2.0
    waypoints.append(wp1)
    
    # Four corner points of the square
    side_length = 3.0
    height = 2.0
    
    # Corner 1: (side_length, 0, height)
    wp2 = np.zeros(13)
    wp2[0] = side_length
    wp2[2] = height
    wp2[6] = 1.0
    waypoints.append(wp2)
    
    # Corner 2: (side_length, side_length, height)
    wp3 = wp2.copy()
    wp3[1] = side_length
    waypoints.append(wp3)
    
    # Corner 3: (0, side_length, height)
    wp4 = wp3.copy()
    wp4[0] = 0.0
    waypoints.append(wp4)
    
    # Corner 4: (0, 0, height) - return to above starting point
    wp5 = wp1.copy()
    waypoints.append(wp5)
    
    # Landing
    wp6 = start.copy()
    waypoints.append(wp6)
    
    # 每段的持续时间
    durations = [2.0, 4.0, 4.0, 4.0, 4.0, 2.0]  # 总共20秒
    
    return waypoints, durations


def create_figure_eight_trajectory():
    """创建figure-eight轨迹"""
    waypoints = []
    
    # 起始点
    start = np.zeros(13)
    start[6] = 1.0
    waypoints.append(start)
    
    # 上升
    wp1 = start.copy()
    wp1[2] = 1.5
    waypoints.append(wp1)
    
    # figure-eight参数
    center_height = 1.5
    radius = 2.0
    n_points = 8  # figure-eight的点数
    
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        
        # figure-eight参数方程
        x = radius * np.sin(t)
        y = radius * np.sin(2 * t) / 2  # figure-eight
        z = center_height
        
        wp = np.zeros(13)
        wp[0] = x
        wp[1] = y
        wp[2] = z
        wp[6] = 1.0
        waypoints.append(wp)
    
    # 回到中心并Landing
    wp_center = np.zeros(13)
    wp_center[2] = center_height
    wp_center[6] = 1.0
    waypoints.append(wp_center)
    
    # Landing
    waypoints.append(start)
    
    # 持续时间
    durations = [2.0] + [2.0] * n_points + [2.0, 2.0]  # 总共24秒
    
    return waypoints, durations


def main():
    """Main function"""
    print("=" * 60)
    print("S500 Trajectory Planning Example")
    print("=" * 60)
    
    try:
        # Create trajectory planner
        planner = S500TrajectoryPlanner()
        
        # Select trajectory type
        print("Available trajectory types:")
        print("1. Square trajectory")
        print("2. Figure-eight trajectory")
        
        choice = input("Please select trajectory type (1 or 2): ").strip()
        
        if choice == "1":
            waypoints, durations = create_square_trajectory()
            trajectory_name = "square"
            print("Creating square trajectory")
        elif choice == "2":
            waypoints, durations = create_figure_eight_trajectory()
            trajectory_name = "figure_eight"
            print("Creating figure-eight trajectory")
        else:
            print("Invalid selection, using default square trajectory")
            waypoints, durations = create_square_trajectory()
            trajectory_name = "square"
        
        # Display waypoint information
        print(f"\nTrajectory information:")
        print(f"  - Number of waypoints: {len(waypoints)}")
        print(f"  - Total time: {sum(durations):.1f} seconds")
        
        for i, (wp, dur) in enumerate(zip(waypoints, durations + [0])):
            pos = wp[:3]
            print(f"  - Waypoint{i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", end="")
            if i < len(durations):
                print(f", time: {dur:.1f}s")
            else:
                print(" (endpoint)")
        
        # Create trajectory optimization problem
        dt = 0.01  # Time step
        planner.create_trajectory_problem(waypoints, durations, dt=dt)
        
        # Solve
        print(f"\nStarting trajectory optimization...")
        converged = planner.solve_trajectory(max_iter=200, verbose=True)
        
        if converged:
            print(f"\n✓ Trajectory optimization successful!")
            
            # Save results
            results_dir = os.path.join(planner.package_path, 'results', 's500_trajectory_optimization')
            os.makedirs(results_dir, exist_ok=True)
            
            # Plot and save results
            plot_path = os.path.join(results_dir, f's500_{trajectory_name}_trajectory.png')
            planner.plot_trajectory(save_path=plot_path)
            
            data_path = os.path.join(results_dir, f's500_{trajectory_name}_trajectory.npz')
            planner.save_trajectory(data_path)
            
            print(f"\nResults saved to:")
            print(f"  - Plot: {plot_path}")
            print(f"  - Data: {data_path}")
            
        else:
            print(f"\n✗ Trajectory optimization did not converge")
            print("Suggest adjusting parameters or checking waypoint settings")
            
    except KeyboardInterrupt:
        print("\nUser interrupted program")
    except Exception as e:
        print(f"\n✗ Program execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
