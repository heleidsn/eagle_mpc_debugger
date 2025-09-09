#!/usr/bin/env python3
'''
Author: Lei He
Date: 2025-01-19
Description: L1 Controller Variables Plotter
    - Plots u_b (baseline control), u_ad (adaptive control), z_tilde (state error), sig_hat (uncertainty estimate)
    - Provides comprehensive visualization of L1 adaptive controller performance
    - Supports both time series plots and analysis plots
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import os

class L1ControllerPlotter:
    """
    Plotter class for L1 Adaptive Controller variables visualization
    """
    
    def __init__(self, save_dir="./l1_plots"):
        """
        Initialize L1 Controller Plotter
        
        Args:
            save_dir (str): Directory to save plots
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Plot configuration
        self.figsize = (16, 12)
        self.colors = {
            'u_b': ['#1f77b4', '#ff7f0e'],  # Blue, Orange for joints
            'u_ad': ['#2ca02c', '#d62728'],  # Green, Red for joints  
            'z_tilde': ['#9467bd', '#8c564b'],  # Purple, Brown for joints
            'sig_hat': ['#e377c2', '#7f7f7f'],  # Pink, Gray for joints
            'reference': '#17becf',  # Cyan for reference
            'true_friction': '#000000'  # Black for true friction
        }
        
        # Line styles
        self.line_styles = {
            'u_b': '-',
            'u_ad': '--', 
            'z_tilde': '-.',
            'sig_hat': ':'
        }
        
        print(f"L1 Controller Plotter initialized. Save directory: {save_dir}")
    
    def plot_all_variables(self, time_data, u_b_data, u_ad_data, z_tilde_data, sig_hat_data,
                          joint_names=None, title_suffix="", save_name=None):
        """
        Plot all L1 controller variables in a comprehensive layout
        
        Args:
            time_data (np.array): Time vector
            u_b_data (np.array): Baseline control data [time x joints]
            u_ad_data (np.array): Adaptive control data [time x joints]
            z_tilde_data (np.array): State estimation error data [time x states]
            sig_hat_data (np.array): Uncertainty estimate data [time x states]
            joint_names (list): List of joint names
            title_suffix (str): Additional title information
            save_name (str): Custom save filename
        """
        
        # Determine number of joints from control data
        n_joints = u_b_data.shape[1] if len(u_b_data.shape) > 1 else 1
        if joint_names is None:
            joint_names = [f"Joint {i+1}" for i in range(n_joints)]
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot baseline control u_b
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(n_joints):
            if len(u_b_data.shape) > 1:
                ax1.plot(time_data, u_b_data[:, i], 
                        color=self.colors['u_b'][i % len(self.colors['u_b'])],
                        linestyle=self.line_styles['u_b'],
                        label=f"{joint_names[i]} - u_b", linewidth=2)
            else:
                ax1.plot(time_data, u_b_data, 
                        color=self.colors['u_b'][0],
                        linestyle=self.line_styles['u_b'],
                        label="u_b (baseline)", linewidth=2)
        ax1.set_ylabel('Baseline Control [Nm]')
        ax1.set_title(f'L1 Controller - Baseline Control (u_b) {title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot adaptive control u_ad
        ax2 = fig.add_subplot(gs[1, :])
        for i in range(n_joints):
            if len(u_ad_data.shape) > 1:
                ax2.plot(time_data, u_ad_data[:, i], 
                        color=self.colors['u_ad'][i % len(self.colors['u_ad'])],
                        linestyle=self.line_styles['u_ad'],
                        label=f"{joint_names[i]} - u_ad", linewidth=2)
            else:
                ax2.plot(time_data, u_ad_data, 
                        color=self.colors['u_ad'][0],
                        linestyle=self.line_styles['u_ad'],
                        label="u_ad (adaptive)", linewidth=2)
        ax2.set_ylabel('Adaptive Control [Nm]')
        ax2.set_title('L1 Controller - Adaptive Control (u_ad)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot state estimation error z_tilde (velocity part)
        ax3 = fig.add_subplot(gs[2, 0])
        # Extract velocity error part (assuming second half of z_tilde is velocity)
        vel_start_idx = z_tilde_data.shape[1] // 2 if len(z_tilde_data.shape) > 1 else 0
        
        if len(z_tilde_data.shape) > 1:
            for i in range(min(n_joints, z_tilde_data.shape[1] - vel_start_idx)):
                ax3.plot(time_data, z_tilde_data[:, vel_start_idx + i], 
                        color=self.colors['z_tilde'][i % len(self.colors['z_tilde'])],
                        linestyle=self.line_styles['z_tilde'],
                        label=f"{joint_names[i]} - z_tilde", linewidth=1.5)
        else:
            ax3.plot(time_data, z_tilde_data, 
                    color=self.colors['z_tilde'][0],
                    linestyle=self.line_styles['z_tilde'],
                    label="z_tilde (state error)", linewidth=1.5)
        ax3.set_ylabel('State Error [rad/s]')
        ax3.set_title('State Estimation Error (z_tilde)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot uncertainty estimate sig_hat (velocity part)
        ax4 = fig.add_subplot(gs[2, 1])
        if len(sig_hat_data.shape) > 1:
            for i in range(min(n_joints, sig_hat_data.shape[1] - vel_start_idx)):
                ax4.plot(time_data, sig_hat_data[:, vel_start_idx + i], 
                        color=self.colors['sig_hat'][i % len(self.colors['sig_hat'])],
                        linestyle=self.line_styles['sig_hat'],
                        label=f"{joint_names[i]} - sig_hat", linewidth=1.5)
        else:
            ax4.plot(time_data, sig_hat_data, 
                    color=self.colors['sig_hat'][0],
                    linestyle=self.line_styles['sig_hat'],
                    label="sig_hat (uncertainty)", linewidth=1.5)
        ax4.set_ylabel('Uncertainty Estimate [Nm]')
        ax4.set_title('Uncertainty Estimate (sig_hat)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot combined control signals
        ax5 = fig.add_subplot(gs[3, :])
        for i in range(n_joints):
            if len(u_b_data.shape) > 1 and len(u_ad_data.shape) > 1:
                # Total control
                u_total = u_b_data[:, i] + u_ad_data[:, i]
                ax5.plot(time_data, u_b_data[:, i], 
                        color=self.colors['u_b'][i % len(self.colors['u_b'])],
                        linestyle=self.line_styles['u_b'], alpha=0.7,
                        label=f"{joint_names[i]} - u_b")
                ax5.plot(time_data, u_ad_data[:, i], 
                        color=self.colors['u_ad'][i % len(self.colors['u_ad'])],
                        linestyle=self.line_styles['u_ad'], alpha=0.7,
                        label=f"{joint_names[i]} - u_ad")
                ax5.plot(time_data, u_total, 
                        color='black', linestyle='-', linewidth=2,
                        label=f"{joint_names[i]} - u_total")
            else:
                u_total = u_b_data + u_ad_data
                ax5.plot(time_data, u_b_data, 
                        color=self.colors['u_b'][0],
                        linestyle=self.line_styles['u_b'], alpha=0.7,
                        label="u_b (baseline)")
                ax5.plot(time_data, u_ad_data, 
                        color=self.colors['u_ad'][0],
                        linestyle=self.line_styles['u_ad'], alpha=0.7,
                        label="u_ad (adaptive)")
                ax5.plot(time_data, u_total, 
                        color='black', linestyle='-', linewidth=2,
                        label="u_total")
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Control Torque [Nm]')
        ax5.set_title('Combined Control Signals')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f'L1 Adaptive Controller Analysis {title_suffix}', fontsize=16, fontweight='bold')
        
        # Save figure
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"l1_controller_analysis_{timestamp}.png"
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L1 controller plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_adaptation_performance(self, time_data, z_tilde_data, sig_hat_data, 
                                  reference_data=None, title_suffix="", save_name=None):
        """
        Plot adaptation performance metrics
        
        Args:
            time_data (np.array): Time vector
            z_tilde_data (np.array): State estimation error data
            sig_hat_data (np.array): Uncertainty estimate data
            reference_data (np.array): Reference trajectory data (optional)
            title_suffix (str): Additional title information
            save_name (str): Custom save filename
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot adaptation error magnitude
        if len(z_tilde_data.shape) > 1:
            error_magnitude = np.linalg.norm(z_tilde_data, axis=1)
        else:
            error_magnitude = np.abs(z_tilde_data)
        
        axes[0, 0].plot(time_data, error_magnitude, 'b-', linewidth=2)
        axes[0, 0].set_ylabel('||z_tilde|| [rad/s]')
        axes[0, 0].set_title('Adaptation Error Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot uncertainty estimate magnitude
        if len(sig_hat_data.shape) > 1:
            uncertainty_magnitude = np.linalg.norm(sig_hat_data, axis=1)
        else:
            uncertainty_magnitude = np.abs(sig_hat_data)
        
        axes[0, 1].plot(time_data, uncertainty_magnitude, 'r-', linewidth=2)
        axes[0, 1].set_ylabel('||sig_hat|| [Nm]')
        axes[0, 1].set_title('Uncertainty Estimate Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot individual joint errors
        if len(z_tilde_data.shape) > 1:
            n_joints = min(2, z_tilde_data.shape[1])
            for i in range(n_joints):
                axes[1, 0].plot(time_data, z_tilde_data[:, -(n_joints-i)], 
                              label=f'Joint {i+1}', linewidth=1.5)
        else:
            axes[1, 0].plot(time_data, z_tilde_data, 'b-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('z_tilde [rad/s]')
        axes[1, 0].set_title('Individual Joint State Errors')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot individual joint uncertainty estimates
        if len(sig_hat_data.shape) > 1:
            n_joints = min(2, sig_hat_data.shape[1])
            for i in range(n_joints):
                axes[1, 1].plot(time_data, sig_hat_data[:, -(n_joints-i)], 
                              label=f'Joint {i+1}', linewidth=1.5)
        else:
            axes[1, 1].plot(time_data, sig_hat_data, 'r-', linewidth=1.5)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('sig_hat [Nm]')
        axes[1, 1].set_title('Individual Joint Uncertainty Estimates')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'L1 Adaptation Performance Analysis {title_suffix}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"l1_adaptation_performance_{timestamp}.png"
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L1 adaptation performance plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_control_breakdown(self, time_data, u_b_data, u_ad_data, 
                             friction_data=None, title_suffix="", save_name=None):
        """
        Plot control signal breakdown and friction compensation
        
        Args:
            time_data (np.array): Time vector
            u_b_data (np.array): Baseline control data
            u_ad_data (np.array): Adaptive control data  
            friction_data (np.array): True friction data (optional)
            title_suffix (str): Additional title information
            save_name (str): Custom save filename
        """
        
        n_joints = u_b_data.shape[1] if len(u_b_data.shape) > 1 else 1
        
        fig, axes = plt.subplots(n_joints, 1, figsize=(12, 4*n_joints))
        if n_joints == 1:
            axes = [axes]
        
        for i in range(n_joints):
            ax = axes[i]
            
            # Extract data for this joint
            if len(u_b_data.shape) > 1:
                u_b_joint = u_b_data[:, i]
                u_ad_joint = u_ad_data[:, i]
                u_total_joint = u_b_joint + u_ad_joint
            else:
                u_b_joint = u_b_data
                u_ad_joint = u_ad_data
                u_total_joint = u_b_joint + u_ad_joint
            
            # Plot control components
            ax.plot(time_data, u_b_joint, 'b-', linewidth=2, label='u_b (baseline)')
            ax.plot(time_data, u_ad_joint, 'g--', linewidth=2, label='u_ad (adaptive)')
            ax.plot(time_data, u_total_joint, 'k-', linewidth=2, label='u_total')
            
            # Plot true friction if available
            if friction_data is not None:
                if len(friction_data.shape) > 1 and i < friction_data.shape[1]:
                    ax.plot(time_data, friction_data[:, i], 'r:', linewidth=2, 
                           label='True Friction')
            
            ax.set_ylabel(f'Joint {i+1} Torque [Nm]')
            ax.set_title(f'Joint {i+1} Control Breakdown')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time [s]')
        plt.suptitle(f'Control Signal Breakdown {title_suffix}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"l1_control_breakdown_{timestamp}.png"
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L1 control breakdown plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_phase_portraits(self, z_tilde_data, sig_hat_data, title_suffix="", save_name=None):
        """
        Plot phase portraits of adaptation variables
        
        Args:
            z_tilde_data (np.array): State estimation error data
            sig_hat_data (np.array): Uncertainty estimate data
            title_suffix (str): Additional title information
            save_name (str): Custom save filename
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Phase portrait: z_tilde vs sig_hat
        if len(z_tilde_data.shape) > 1 and len(sig_hat_data.shape) > 1:
            # Use last two joints for phase portrait
            n_states = z_tilde_data.shape[1]
            for i in range(min(2, n_states)):
                idx = -(i+1)
                axes[0].plot(z_tilde_data[:, idx], sig_hat_data[:, idx], 
                           label=f'Joint {i+1}', alpha=0.7)
        else:
            axes[0].plot(z_tilde_data, sig_hat_data, 'b-', alpha=0.7)
        
        axes[0].set_xlabel('z_tilde [rad/s]')
        axes[0].set_ylabel('sig_hat [Nm]')
        axes[0].set_title('Phase Portrait: z_tilde vs sig_hat')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time evolution of adaptation
        if len(z_tilde_data.shape) > 1:
            z_magnitude = np.linalg.norm(z_tilde_data, axis=1)
            sig_magnitude = np.linalg.norm(sig_hat_data, axis=1)
        else:
            z_magnitude = np.abs(z_tilde_data)
            sig_magnitude = np.abs(sig_hat_data)
        
        axes[1].plot(z_magnitude, sig_magnitude, 'r-', alpha=0.7)
        axes[1].set_xlabel('||z_tilde|| [rad/s]')
        axes[1].set_ylabel('||sig_hat|| [Nm]')
        axes[1].set_title('Adaptation Trajectory')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'L1 Phase Analysis {title_suffix}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"l1_phase_portraits_{timestamp}.png"
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L1 phase portraits saved to: {save_path}")
        
        plt.show()
        return fig


def create_sample_data(duration=10.0, dt=0.001, n_joints=2):
    """
    Create sample L1 controller data for demonstration
    
    Args:
        duration (float): Simulation duration
        dt (float): Time step
        n_joints (int): Number of joints
        
    Returns:
        dict: Sample data dictionary
    """
    time_steps = int(duration / dt)
    time_data = np.linspace(0, duration, time_steps)
    
    # Generate synthetic data that mimics L1 controller behavior
    u_b_data = np.zeros((time_steps, n_joints))
    u_ad_data = np.zeros((time_steps, n_joints))
    z_tilde_data = np.zeros((time_steps, n_joints * 2))  # position + velocity errors
    sig_hat_data = np.zeros((time_steps, n_joints * 2))  # uncertainty estimates
    
    for i in range(n_joints):
        # Baseline control (PD-like behavior)
        ref_signal = 0.5 * np.sin(0.5 * time_data) + 0.3 * np.sin(1.2 * time_data)
        u_b_data[:, i] = 2.0 * ref_signal + 0.5 * np.random.normal(0, 0.1, time_steps)
        
        # Adaptive control (starts small, adapts to disturbances)
        disturbance = 0.2 * np.sin(2.0 * time_data) * (1 - np.exp(-time_data / 2.0))
        u_ad_data[:, i] = disturbance + 0.1 * np.random.normal(0, 0.05, time_steps)
        
        # State estimation error (decreases over time as adaptation improves)
        z_tilde_data[:, i] = 0.1 * np.exp(-time_data / 3.0) * np.sin(3.0 * time_data)
        z_tilde_data[:, i + n_joints] = 0.05 * np.exp(-time_data / 2.0) * np.sin(5.0 * time_data)
        
        # Uncertainty estimate (converges to true disturbance)
        sig_hat_data[:, i] = 0.3 * (1 - np.exp(-time_data / 1.5)) * np.sin(2.0 * time_data)
        sig_hat_data[:, i + n_joints] = 0.15 * (1 - np.exp(-time_data / 1.0)) * np.sin(4.0 * time_data)
    
    return {
        'time': time_data,
        'u_b': u_b_data,
        'u_ad': u_ad_data,
        'z_tilde': z_tilde_data,
        'sig_hat': sig_hat_data
    }


def demo_l1_plotter():
    """Demonstration of L1 Controller Plotter"""
    print("Demonstrating L1 Controller Plotter...")
    
    # Create sample data
    sample_data = create_sample_data(duration=10.0, dt=0.01, n_joints=2)
    
    # Create plotter
    plotter = L1ControllerPlotter(save_dir="./demo_l1_plots")
    
    # Plot all variables
    plotter.plot_all_variables(
        sample_data['time'], 
        sample_data['u_b'], 
        sample_data['u_ad'],
        sample_data['z_tilde'], 
        sample_data['sig_hat'],
        joint_names=["Joint 1", "Joint 2"],
        title_suffix="(Demo Data)",
        save_name="demo_l1_all_variables.png"
    )
    
    # Plot adaptation performance
    plotter.plot_adaptation_performance(
        sample_data['time'], 
        sample_data['z_tilde'], 
        sample_data['sig_hat'],
        title_suffix="(Demo Data)",
        save_name="demo_l1_adaptation.png"
    )
    
    # Plot control breakdown
    plotter.plot_control_breakdown(
        sample_data['time'], 
        sample_data['u_b'], 
        sample_data['u_ad'],
        title_suffix="(Demo Data)",
        save_name="demo_l1_breakdown.png"
    )
    
    # Plot phase portraits
    plotter.plot_phase_portraits(
        sample_data['z_tilde'], 
        sample_data['sig_hat'],
        title_suffix="(Demo Data)",
        save_name="demo_l1_phase.png"
    )
    
    print("Demo completed!")


if __name__ == "__main__":
    demo_l1_plotter()
