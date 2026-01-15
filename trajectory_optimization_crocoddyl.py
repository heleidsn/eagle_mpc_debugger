#!/usr/bin/env python3
"""
基于Crocoddyl的轨迹优化脚本
读取s500_uam_hover.yaml配置文件，创建优化问题并求解

Author: Assistant
Date: 2025-09-12
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import time
import pinocchio as pin
import crocoddyl
import rospkg
from pathlib import Path

class TrajectoryOptimizer:
    def __init__(self, config_file_path):
        """
        初始化轨迹优化器
        
        Args:
            config_file_path: YAML配置文件路径
        """
        self.config_file_path = config_file_path
        self.config = None
        self.robot_model = None
        self.robot_data = None
        self.state = None
        self.actuation = None
        self.problem = None
        self.solver = None
        
        # 加载配置文件
        self.load_config()
        
        # 初始化机器人模型
        self.initialize_robot_model()
        
    def load_config(self):
        """加载YAML配置文件"""
        try:
            with open(self.config_file_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ 成功加载配置文件: {self.config_file_path}")
        except Exception as e:
            print(f"✗ 加载配置文件失败: {e}")
            raise
            
    def initialize_robot_model(self):
        """初始化机器人模型"""
        try:
            # 获取ROS包路径
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('eagle_mpc_debugger')
            
            # 构建URDF文件的绝对路径
            urdf_path = self.config['trajectory']['robot']['urdf']
            if not urdf_path.startswith('/'):
                urdf_path = os.path.join(package_path, urdf_path)
                
            print(f"正在加载URDF文件: {urdf_path}")
            
            # 构建Pinocchio模型
            self.robot_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
            self.robot_data = self.robot_model.createData()
            
            # 创建状态和动作模型
            self.state = crocoddyl.StateMultibody(self.robot_model)
            
            # 加载平台参数
            follow_path = self.config['trajectory']['robot']['follow']
            if not follow_path.startswith('/'):
                follow_path = os.path.join(package_path, follow_path)
                
            with open(follow_path, 'r') as f:
                platform_config = yaml.safe_load(f)
                
            # 创建多旋翼动作模型
            n_rotors = platform_config['platform']['n_rotors']
            cf = platform_config['platform']['cf']
            cm = platform_config['platform']['cm']
            
            # 构建tau_f矩阵 (推力到力/力矩的映射矩阵)
            # tau_f的形状应该是 (6, n_rotors)，表示从n_rotors个推力输入到6个力/力矩输出的映射
            tau_f = np.zeros((6, n_rotors))
            
            # 从配置中提取旋翼位置和方向
            rotors = platform_config['platform']['$rotors']
            for i, rotor in enumerate(rotors):
                pos = np.array(rotor['translation'])
                spin_dir = rotor['spin_direction'][0]
                
                # 力映射 (所有旋翼都向上产生推力)
                tau_f[0, i] = 0  # x方向力
                tau_f[1, i] = 0  # y方向力  
                tau_f[2, i] = 1.0  # z方向推力 (归一化，实际推力由控制输入决定)
                
                # 力矩映射 (使用旋翼位置计算力矩臂)
                tau_f[3, i] = pos[1]  # 绕x轴力矩 (roll) = y * Fz
                tau_f[4, i] = -pos[0]  # 绕y轴力矩 (pitch) = -x * Fz
                tau_f[5, i] = spin_dir * cm / cf  # 绕z轴力矩 (yaw) = 反作用力矩系数比
            
            # 使用ActuationModelMultiCopterBase
            self.actuation = crocoddyl.ActuationModelMultiCopterBase(self.state, tau_f)
            
            print(f"✓ 成功初始化机器人模型")
            print(f"  - 自由度: {self.robot_model.nq}")
            print(f"  - 旋翼数量: {n_rotors}")
            print(f"  - 控制输入维度: {self.actuation.nu}")
            
        except Exception as e:
            print(f"✗ 初始化机器人模型失败: {e}")
            raise
            
    def create_cost_model(self, stage_config, is_terminal=False, waypoint_multiplier=10.0):
        """
        根据配置创建代价模型
        
        Args:
            stage_config: 阶段配置字典
            is_terminal: 是否为终端代价
            waypoint_multiplier: waypoint阶段的权重倍数
            
        Returns:
            代价模型
        """
        # 暂时保持控制维度一致，避免维度不匹配问题
        control_dim = self.actuation.nu
        cost_model = crocoddyl.CostModelSum(self.state, control_dim)
        
        if 'costs' not in stage_config:
            return cost_model
            
        # 检测是否为waypoint阶段（duration为0或阶段名包含"wp_"）
        is_waypoint = False
        stage_name = stage_config.get('name', '').lower()
        stage_duration = stage_config.get('duration', 0)
        
        if stage_duration == 0 or 'wp_' in stage_name:
            is_waypoint = True
            print(f"🎯 检测到waypoint阶段: {stage_name}, 将应用{waypoint_multiplier}x权重倍数")
            
        for cost_config in stage_config['costs']:
            cost_name = cost_config['name']
            cost_type = cost_config['type']
            weight = float(cost_config['weight'])
            
            # 为waypoint阶段的状态cost应用更高权重
            if is_waypoint and cost_type == "ResidualModelState":
                original_weight = weight
                weight = float(weight * waypoint_multiplier)
                print(f"  📈 {cost_name}: {original_weight} -> {weight} (waypoint权重增强)")
            
            if cost_type == "ResidualModelState":
                # 状态正则化代价
                reference = np.array(cost_config['reference'])
                
                # 创建激活函数
                if 'activation' in cost_config and cost_config['activation'] == "ActivationModelWeightedQuad":
                    if 'weights' in cost_config:
                        weights = np.array(cost_config['weights'])
                        activation = crocoddyl.ActivationModelWeightedQuad(weights)
                    else:
                        activation = crocoddyl.ActivationModelQuad(self.state.ndx)
                else:
                    activation = crocoddyl.ActivationModelQuad(self.state.ndx)
                
                # 创建残差模型
                residual = crocoddyl.ResidualModelState(self.state, reference, control_dim)
                
                # 添加代价
                cost_model.addCost(cost_name, crocoddyl.CostModelResidual(self.state, activation, residual), weight)
                
            elif cost_type == "ResidualModelControl" and not is_terminal:
                # 控制输入正则化代价
                reference = np.array(cost_config['reference'])
                
                # 确保reference的维度与控制输入维度匹配
                if len(reference) != self.actuation.nu:
                    # 调整reference维度以匹配actuation.nu
                    if len(reference) > self.actuation.nu:
                        reference = reference[:self.actuation.nu]
                    else:
                        # 如果reference维度较小，用零填充
                        reference_padded = np.zeros(self.actuation.nu)
                        reference_padded[:len(reference)] = reference
                        reference = reference_padded
                
                # 创建激活函数
                if 'activation' in cost_config and cost_config['activation'] == "ActivationModelWeightedQuad":
                    if 'weights' in cost_config:
                        weights = np.array(cost_config['weights'])
                        # 同样调整weights维度
                        if len(weights) != self.actuation.nu:
                            if len(weights) > self.actuation.nu:
                                weights = weights[:self.actuation.nu]
                            else:
                                weights_padded = np.ones(self.actuation.nu)
                                weights_padded[:len(weights)] = weights
                                weights = weights_padded
                        activation = crocoddyl.ActivationModelWeightedQuad(weights)
                    else:
                        activation = crocoddyl.ActivationModelQuad(self.actuation.nu)
                else:
                    activation = crocoddyl.ActivationModelQuad(self.actuation.nu)
                
                
                # 创建残差模型
                residual = crocoddyl.ResidualModelControl(self.state, reference)
                
                # 添加代价
                cost_model.addCost(cost_name, crocoddyl.CostModelResidual(self.state, activation, residual), weight)
                
        return cost_model
        
    def create_problem(self, dt=0.02, waypoint_multiplier=10.0):
        """
        创建轨迹优化问题
        
        Args:
            dt: 时间步长 (秒)
            waypoint_multiplier: waypoint阶段的权重倍数
        """
        try:
            # 获取初始状态
            initial_state = np.array(self.config['trajectory']['initial_state'])
            
            # 获取阶段配置
            stages = self.config['trajectory']['stages']
            
            running_models = []
            
            for stage in stages:
                stage_name = stage['name']
                duration_ms = stage['duration']
                duration_s = duration_ms / 1000.0
                
                print(f"创建阶段: {stage_name}, 持续时间: {duration_s}s")
                
                # 计算该阶段的时间步数
                n_steps = max(1, int(duration_s / dt))
                
                # 创建代价模型
                cost_model = self.create_cost_model(stage, is_terminal=False, waypoint_multiplier=waypoint_multiplier)
                
                # 创建微分动作模型
                diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self.state, self.actuation, cost_model
                )
                
                # 创建积分动作模型
                int_model = crocoddyl.IntegratedActionModelEuler(diff_model, dt)
                
                # 添加到运行模型列表
                for _ in range(n_steps):
                    running_models.append(int_model)
            
            # 创建终端模型 (使用最后一个阶段的配置)
            terminal_cost = self.create_cost_model(stages[-1], is_terminal=True, waypoint_multiplier=waypoint_multiplier)
            
            
            # 创建终端微分动作模型
            terminal_diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, terminal_cost
            )
            # 对于终端模型，使用零时间步长的积分模型
            terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff_model, 0.0)
            
            # 创建射击问题
            self.problem = crocoddyl.ShootingProblem(initial_state, running_models, terminal_model)
            
            print(f"✓ 成功创建轨迹优化问题")
            print(f"  - 初始状态维度: {len(initial_state)}")
            print(f"  - 运行节点数: {len(running_models)}")
            print(f"  - 时间步长: {dt}s")
            print(f"  - 总时间: {len(running_models) * dt:.2f}s")
            
        except Exception as e:
            print(f"✗ 创建轨迹优化问题失败: {e}")
            raise
            
    def solve(self, max_iter=100, verbose=True):
        """
        求解轨迹优化问题
        
        Args:
            max_iter: 最大迭代次数
            verbose: 是否显示详细信息
        """
        try:
            # 创建求解器
            self.solver = crocoddyl.SolverBoxFDDP(self.problem)
            
            # 设置求解器参数
            self.solver.convergence_init = 1e-6
            
            # 设置回调函数
            callbacks = []
            if verbose:
                callbacks.append(crocoddyl.CallbackVerbose())
            
            # 添加日志记录
            logger = crocoddyl.CallbackLogger()
            callbacks.append(logger)
            
            self.solver.setCallbacks(callbacks)
            
            print(f"开始求解轨迹优化问题...")
            print(f"最大迭代次数: {max_iter}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 求解
            converged = self.solver.solve([], [], max_iter)
            
            # 记录结束时间
            end_time = time.time()
            solve_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            print(f"✓ 轨迹优化完成")
            print(f"  - 求解时间: {solve_time:.2f} ms")
            print(f"  - 收敛状态: {'收敛' if converged else '未收敛'}")
            print(f"  - 最终代价: {self.solver.cost:.6f}")
            print(f"  - 迭代次数: {self.solver.iter}")
            
            return converged
            
        except Exception as e:
            print(f"✗ 求解轨迹优化问题失败: {e}")
            raise
            
    def get_trajectory(self):
        """
        获取优化后的轨迹
        
        Returns:
            (states, controls): 状态轨迹和控制轨迹
        """
        if self.solver is None:
            raise RuntimeError("请先求解轨迹优化问题")
            
        return self.solver.xs, self.solver.us
        
    def _identify_waypoint_indices(self, dt=0.02):
        """
        识别waypoint在轨迹中的索引位置
        
        Args:
            dt: 时间步长
            
        Returns:
            List[int]: waypoint索引列表
        """
        if not hasattr(self, 'config') or 'trajectory' not in self.config:
            return []
            
        waypoint_indices = []
        stages = self.config['trajectory']['stages']
        current_index = 0
        
        for stage in stages:
            stage_name = stage.get('name', '').lower()
            duration_ms = stage.get('duration', 0)
            duration_s = duration_ms / 1000.0
            
            # 检测是否为waypoint阶段
            is_waypoint = duration_s == 0 or 'wp_' in stage_name
            
            if is_waypoint:
                waypoint_indices.append(current_index)
                print(f"🎯 发现waypoint: {stage.get('name', 'unnamed')} at index {current_index}")
            
            # 计算该阶段的时间步数
            n_steps = max(1, int(duration_s / dt))
            current_index += n_steps
        
        return waypoint_indices
        
    def plot_results(self, save_path=None, show_waypoints=True):
        """
        绘制优化结果
        
        Args:
            save_path: 保存路径 (可选)
            show_waypoints: 是否显示waypoint标注 (默认True)
        """
        if self.solver is None:
            print("✗ 请先求解轨迹优化问题")
            return
            
        states, controls = self.get_trajectory()
        
        # 时间轴
        dt = 0.02  # 假设时间步长
        time_states = np.arange(len(states)) * dt
        time_controls = np.arange(len(controls)) * dt
        
        # 识别waypoint位置
        waypoint_indices = []
        if show_waypoints:
            waypoint_indices = self._identify_waypoint_indices(dt)
        
        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Trajectory Optimization Results', fontsize=16)
        
        # 提取状态数据
        positions = np.array([x[:3] for x in states])  # x, y, z
        orientations = np.array([x[3:7] for x in states])  # 四元数
        velocities = np.array([x[7:13] for x in states])  # 线速度和角速度
        
        # 绘制位置
        axes[0, 0].plot(time_states, positions[:, 0], 'r-', label='x')
        axes[0, 0].plot(time_states, positions[:, 1], 'g-', label='y')
        axes[0, 0].plot(time_states, positions[:, 2], 'b-', label='z')
        
        # 添加waypoint标注
        if show_waypoints and waypoint_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if wp_idx < len(time_states):
                    axes[0, 0].axvline(x=time_states[wp_idx], color='orange', linestyle='--', alpha=0.7)
                    axes[0, 0].text(time_states[wp_idx], axes[0, 0].get_ylim()[1]*0.9, 
                                   f'WP{i+1}', rotation=90, ha='right', va='top',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Position Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 绘制速度
        axes[0, 1].plot(time_states, velocities[:, 0], 'r-', label='vx')
        axes[0, 1].plot(time_states, velocities[:, 1], 'g-', label='vy')
        axes[0, 1].plot(time_states, velocities[:, 2], 'b-', label='vz')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Linear Velocity (m/s)')
        axes[0, 1].set_title('Linear Velocity Trajectory')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 绘制角速度
        axes[1, 0].plot(time_states, velocities[:, 3], 'r-', label='ωx')
        axes[1, 0].plot(time_states, velocities[:, 4], 'g-', label='ωy')
        axes[1, 0].plot(time_states, velocities[:, 5], 'b-', label='ωz')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
        axes[1, 0].set_title('Angular Velocity Trajectory')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 绘制控制输入
        if len(controls) > 0:
            controls_array = np.array(controls)
            for i in range(min(4, controls_array.shape[1])):  # 最多显示4个旋翼
                axes[1, 1].plot(time_controls, controls_array[:, i], label=f'Rotor{i+1}')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Thrust (N)')
            axes[1, 1].set_title('Control Inputs')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # 绘制3D轨迹
        ax_3d = fig.add_subplot(3, 2, 5, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                     color='g', s=100, label='Start')
        ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                     color='r', s=100, label='End')
        
        # 添加waypoint标注到3D图
        if show_waypoints and waypoint_indices:
            for i, wp_idx in enumerate(waypoint_indices):
                if wp_idx < len(positions):
                    wp_pos = positions[wp_idx]
                    ax_3d.scatter(wp_pos[0], wp_pos[1], wp_pos[2], 
                                 color='orange', s=150, marker='*', 
                                 label='Waypoints' if i == 0 else "", alpha=0.8)
                    ax_3d.text(wp_pos[0], wp_pos[1], wp_pos[2] + 0.1, 
                              f'WP{i+1}', fontsize=10, ha='center',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Trajectory')
        ax_3d.legend()
        
        # 绘制代价函数收敛曲线
        if hasattr(self.solver, 'cost_evolution'):
            axes[2, 1].semilogy(self.solver.cost_evolution)
            axes[2, 1].set_xlabel('Iteration')
            axes[2, 1].set_ylabel('Cost Function Value')
            axes[2, 1].set_title('Convergence Curve')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Results plot saved to: {save_path}")
        
        plt.show()
        
    def save_trajectory(self, save_path):
        """
        保存优化后的轨迹
        
        Args:
            save_path: 保存路径
        """
        if self.solver is None:
            print("✗ 请先求解轨迹优化问题")
            return
            
        states, controls = self.get_trajectory()
        
        # 保存为numpy数组
        np.savez(save_path, 
                states=np.array(states),
                controls=np.array(controls),
                cost=self.solver.cost,
                iterations=self.solver.iter)
        
        print(f"✓ Trajectory data saved to: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("基于Crocoddyl的轨迹优化脚本")
    print("=" * 60)
    
    try:
        # 获取配置文件路径
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('eagle_mpc_debugger')
        config_file = os.path.join(package_path, 'config/yaml/trajectories/s500_uam_hover.yaml')
        
        # 创建轨迹优化器
        optimizer = TrajectoryOptimizer(config_file)
        
        # 创建优化问题
        optimizer.create_problem(dt=0.1)
        
        # 求解
        converged = optimizer.solve(max_iter=100, verbose=True)
        
        if converged:
            print("\n" + "=" * 60)
            print("轨迹优化成功完成！")
            print("=" * 60)
            
            # 绘制结果
            results_dir = os.path.join(package_path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            plot_path = os.path.join(results_dir, 's500_uam_hover_trajectory.png')
            optimizer.plot_results(save_path=plot_path)
            
            # 保存轨迹数据
            data_path = os.path.join(results_dir, 's500_uam_hover_trajectory.npz')
            optimizer.save_trajectory(data_path)
            
        else:
            print("\n" + "=" * 60)
            print("轨迹优化未收敛，请检查参数设置")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n✗ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
