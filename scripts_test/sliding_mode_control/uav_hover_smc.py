#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV Height Control Comparison: Sliding Mode Control vs PD Control
==================================================================
System Model: Second-order system (height control)
- States: h (height), v_h (vertical velocity)
- Dynamics: h'' = u + d
- Disturbance: d = A*sin(ω*t) (sinusoidal force disturbance)

Comparing three controllers:
1. Sliding Mode Control (SMC - Simple): u = -k * sat(s / phi)
2. Sliding Mode Control (SMC - Eq+Sw): u = u_eq + u_sw, where u_eq = -c1*v, u_sw = -k*sat(s/phi)
3. PD Control: Traditional linear control

Author: Auto
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ======================
# System Parameters
# ======================
# Simulation parameters
dt = 0.001          # Time step [s]
T = 10.0            # Simulation duration [s]
N = int(T / dt)     # Number of simulation steps

# Initial states
h0 = 0.0            # Initial height [m]
v0 = 0.0            # Initial velocity [m/s]

# Reference height
h_ref = 5.0         # Desired height [m]

# Disturbance parameters
A_dist = 2.0        # Disturbance amplitude [m/s²]
omega_dist = 1.0    # Disturbance frequency [rad/s]

# ======================
# Sliding Mode Control Parameters
# ======================
c1_smc = 2.0        # Sliding surface coefficient
k_smc = 20.0         # Switching gain
phi_smc = 0.1       # Boundary layer thickness (reduces chattering)

# ======================
# PD Control Parameters
# ======================
kp_pd = 15.0        # Proportional gain
kd_pd = 8         # Derivative gain

# Control input saturation (same as SMC)
u_max = k_smc       # Maximum control input [m/s²]
u_min = -k_smc      # Minimum control input [m/s²]

# ======================
# Controller Implementation
# ======================

def sat(x, phi):
    """Saturation function for sliding mode control boundary layer"""
    return np.clip(x / phi, -1.0, 1.0)

def sliding_mode_control(h, v, h_ref):
    """
    Sliding Mode Controller (Simple version)
    Sliding surface: s = c1 * (h - h_ref) + v
    Control law: u = -k * sat(s / phi)
    """
    e = h - h_ref
    s = c1_smc * e + v
    u = -k_smc * sat(s, phi_smc)
    return u, s

def sliding_mode_control_eq_sw(h, v, h_ref):
    """
    Sliding Mode Controller (Equivalent + Switching Control)
    Sliding surface: s = c1 * (h - h_ref) + v
    Control law: u = u_eq + u_sw
    where:
    - u_eq = -c1 * v (equivalent control, based on nominal model)
    - u_sw = -k * sat(s / phi) (switching control, handles disturbances)
    """
    e = h - h_ref
    s = c1_smc * e + v
    
    # Equivalent control: makes s_dot = 0 in nominal case (no disturbance)
    u_eq = -c1_smc * v
    
    # Switching control: handles disturbances and uncertainties
    u_sw = -k_smc * sat(s, phi_smc)
    
    # Total control
    u = u_eq + u_sw
    return u, s, u_eq, u_sw

def pd_control(h, v, h_ref):
    """
    PD Controller with saturation
    Control law: u = -kp * (h - h_ref) - kd * v
    Saturated to match SMC control input limits
    """
    e = h - h_ref
    u = -kp_pd * e - kd_pd * v
    # Saturate control input to match SMC limits
    u = np.clip(u, u_min, u_max)
    return u

def disturbance(t):
    """Sinusoidal disturbance"""
    return A_dist * np.sin(omega_dist * t)

# ======================
# Simulation Functions
# ======================

def simulate_smc():
    """Sliding mode control simulation (simple version)"""
    h = h0
    v = v0
    
    t_list = []
    h_list = []
    v_list = []
    u_list = []
    s_list = []
    d_list = []
    e_list = []
    
    for i in range(N):
        t = i * dt
        
        # Disturbance
        d = disturbance(t)
        
        # Sliding mode controller
        u, s = sliding_mode_control(h, v, h_ref)
        
        # System dynamics: h'' = u + d
        # State space: h' = v, v' = u + d
        h_dot = v
        v_dot = u + d
        
        # Euler integration
        h += h_dot * dt
        v += v_dot * dt
        
        # Record data
        t_list.append(t)
        h_list.append(h)
        v_list.append(v)
        u_list.append(u)
        s_list.append(s)
        d_list.append(d)
        e_list.append(h - h_ref)
    
    return {
        't': np.array(t_list),
        'h': np.array(h_list),
        'v': np.array(v_list),
        'u': np.array(u_list),
        's': np.array(s_list),
        'd': np.array(d_list),
        'e': np.array(e_list)
    }

def simulate_smc_eq_sw():
    """Sliding mode control simulation (equivalent + switching control version)"""
    h = h0
    v = v0
    
    t_list = []
    h_list = []
    v_list = []
    u_list = []
    u_eq_list = []
    u_sw_list = []
    s_list = []
    d_list = []
    e_list = []
    
    for i in range(N):
        t = i * dt
        
        # Disturbance
        d = disturbance(t)
        
        # Sliding mode controller (equivalent + switching)
        u, s, u_eq, u_sw = sliding_mode_control_eq_sw(h, v, h_ref)
        
        # System dynamics: h'' = u + d
        # State space: h' = v, v' = u + d
        h_dot = v
        v_dot = u + d
        
        # Euler integration
        h += h_dot * dt
        v += v_dot * dt
        
        # Record data
        t_list.append(t)
        h_list.append(h)
        v_list.append(v)
        u_list.append(u)
        u_eq_list.append(u_eq)
        u_sw_list.append(u_sw)
        s_list.append(s)
        d_list.append(d)
        e_list.append(h - h_ref)
    
    return {
        't': np.array(t_list),
        'h': np.array(h_list),
        'v': np.array(v_list),
        'u': np.array(u_list),
        'u_eq': np.array(u_eq_list),
        'u_sw': np.array(u_sw_list),
        's': np.array(s_list),
        'd': np.array(d_list),
        'e': np.array(e_list)
    }

def simulate_pd():
    """PD control simulation"""
    h = h0
    v = v0
    
    t_list = []
    h_list = []
    v_list = []
    u_list = []
    d_list = []
    e_list = []
    
    for i in range(N):
        t = i * dt
        
        # Disturbance
        d = disturbance(t)
        
        # PD controller
        u = pd_control(h, v, h_ref)
        
        # System dynamics: h'' = u + d
        # State space: h' = v, v' = u + d
        h_dot = v
        v_dot = u + d
        
        # Euler integration
        h += h_dot * dt
        v += v_dot * dt
        
        # Record data
        t_list.append(t)
        h_list.append(h)
        v_list.append(v)
        u_list.append(u)
        d_list.append(d)
        e_list.append(h - h_ref)
    
    return {
        't': np.array(t_list),
        'h': np.array(h_list),
        'v': np.array(v_list),
        'u': np.array(u_list),
        'd': np.array(d_list),
        'e': np.array(e_list)
    }

# ======================
# Run Simulation
# ======================
print("Running sliding mode control simulation (simple version)...")
data_smc = simulate_smc()

print("Running sliding mode control simulation (equivalent + switching)...")
data_smc_eq_sw = simulate_smc_eq_sw()

print("Running PD control simulation...")
data_pd = simulate_pd()

# ======================
# Calculate Performance Metrics
# ======================
def calculate_metrics(data):
    """Calculate control performance metrics"""
    e = data['e']
    u = data['u']
    
    # Steady-state error (average of last 2 seconds)
    steady_start = int(0.8 * len(e))
    steady_error = np.abs(e[steady_start:]).mean()
    
    # Maximum error
    max_error = np.abs(e).max()
    
    # Control input RMS
    u_rms = np.sqrt(np.mean(u**2))
    
    # Maximum control input
    u_max = np.abs(u).max()
    
    return {
        'steady_error': steady_error,
        'max_error': max_error,
        'u_rms': u_rms,
        'u_max': u_max
    }

metrics_smc = calculate_metrics(data_smc)
metrics_smc_eq_sw = calculate_metrics(data_smc_eq_sw)
metrics_pd = calculate_metrics(data_pd)

print("\nPerformance Comparison:")
print("=" * 80)
print(f"{'Metric':<20} {'SMC (Simple)':<20} {'SMC (Eq+Sw)':<20} {'PD Control':<20}")
print("=" * 80)
print(f"{'Steady Error [m]':<20} {metrics_smc['steady_error']:<20.4f} {metrics_smc_eq_sw['steady_error']:<20.4f} {metrics_pd['steady_error']:<20.4f}")
print(f"{'Max Error [m]':<20} {metrics_smc['max_error']:<20.4f} {metrics_smc_eq_sw['max_error']:<20.4f} {metrics_pd['max_error']:<20.4f}")
print(f"{'Control RMS':<20} {metrics_smc['u_rms']:<20.4f} {metrics_smc_eq_sw['u_rms']:<20.4f} {metrics_pd['u_rms']:<20.4f}")
print(f"{'Max Control':<20} {metrics_smc['u_max']:<20.4f} {metrics_smc_eq_sw['u_max']:<20.4f} {metrics_pd['u_max']:<20.4f}")
print("=" * 80)

# ======================
# Plotting
# ======================

plt.figure(figsize=(14, 10))

# 1) Height response comparison
plt.subplot(3, 2, 1)
plt.plot(data_smc['t'], data_smc['h'], 'b-', linewidth=2, label='SMC (Simple)')
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['h'], 'g-', linewidth=2, label='SMC (Eq+Sw)')
plt.plot(data_pd['t'], data_pd['h'], 'r--', linewidth=2, label='PD Control')
plt.plot(data_smc['t'], h_ref * np.ones_like(data_smc['t']), 'k:', linewidth=1.5, label=f'Reference = {h_ref} m')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Height [m]', fontsize=11)
plt.title('Height Response Comparison', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 2) Height error comparison
plt.subplot(3, 2, 2)
plt.plot(data_smc['t'], data_smc['e'], 'b-', linewidth=2, label='SMC (Simple)')
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['e'], 'g-', linewidth=2, label='SMC (Eq+Sw)')
plt.plot(data_pd['t'], data_pd['e'], 'r--', linewidth=2, label='PD Control')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Height Error [m]', fontsize=11)
plt.title('Height Error Comparison', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 3) Velocity response comparison
plt.subplot(3, 2, 3)
plt.plot(data_smc['t'], data_smc['v'], 'b-', linewidth=2, label='SMC (Simple)')
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['v'], 'g-', linewidth=2, label='SMC (Eq+Sw)')
plt.plot(data_pd['t'], data_pd['v'], 'r--', linewidth=2, label='PD Control')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Velocity [m/s]', fontsize=11)
plt.title('Velocity Response Comparison', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 4) Control input comparison
plt.subplot(3, 2, 4)
plt.plot(data_smc['t'], data_smc['u'], 'b-', linewidth=2, label='SMC (Simple)')
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['u'], 'g-', linewidth=2, label='SMC (Eq+Sw)')
plt.plot(data_pd['t'], data_pd['u'], 'r--', linewidth=2, label='PD Control')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Control Input [m/s²]', fontsize=11)
plt.title('Control Input Comparison', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 5) Disturbance signal
plt.subplot(3, 2, 5)
plt.plot(data_smc['t'], data_smc['d'], 'g-', linewidth=2, label=f'Disturbance: {A_dist}*sin({omega_dist}*t)')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Disturbance [m/s²]', fontsize=11)
plt.title('External Disturbance', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 6) Control decomposition (SMC Eq+Sw only)
plt.subplot(3, 2, 6)
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['u_eq'], 'c-', linewidth=2, label='Equivalent Control u_eq')
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['u_sw'], 'm-', linewidth=2, label='Switching Control u_sw')
plt.plot(data_smc_eq_sw['t'], data_smc_eq_sw['u'], 'g-', linewidth=2, label='Total Control u')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Control Input [m/s²]', fontsize=11)
plt.title('SMC Control Decomposition (Eq+Sw)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uav_height_control_comparison.png', dpi=150, bbox_inches='tight')
print("\nFigure saved as: uav_height_control_comparison.png")
plt.show()

# ======================
# Zoomed View (Steady-state Region)
# ======================
t_start_zoom = 8.0
t_end_zoom = 10.0
idx_start = int(t_start_zoom / dt)
idx_end = int(t_end_zoom / dt)

plt.figure(figsize=(12, 6))

# Height response zoom
plt.subplot(1, 2, 1)
plt.plot(data_smc['t'][idx_start:idx_end], data_smc['h'][idx_start:idx_end], 
         'b-', linewidth=2, label='SMC (Simple)')
plt.plot(data_smc_eq_sw['t'][idx_start:idx_end], data_smc_eq_sw['h'][idx_start:idx_end], 
         'g-', linewidth=2, label='SMC (Eq+Sw)')
plt.plot(data_pd['t'][idx_start:idx_end], data_pd['h'][idx_start:idx_end], 
         'r--', linewidth=2, label='PD Control')
plt.plot(data_smc['t'][idx_start:idx_end], 
         h_ref * np.ones(len(data_smc['t'][idx_start:idx_end])), 
         'k:', linewidth=1.5, label=f'Reference = {h_ref} m')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Height [m]', fontsize=11)
plt.title(f'Height Response Zoom ({t_start_zoom}-{t_end_zoom} s)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# Error zoom
plt.subplot(1, 2, 2)
plt.plot(data_smc['t'][idx_start:idx_end], data_smc['e'][idx_start:idx_end], 
         'b-', linewidth=2, label='SMC (Simple)')
plt.plot(data_smc_eq_sw['t'][idx_start:idx_end], data_smc_eq_sw['e'][idx_start:idx_end], 
         'g-', linewidth=2, label='SMC (Eq+Sw)')
plt.plot(data_pd['t'][idx_start:idx_end], data_pd['e'][idx_start:idx_end], 
         'r--', linewidth=2, label='PD Control')
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Height Error [m]', fontsize=11)
plt.title(f'Height Error Zoom ({t_start_zoom}-{t_end_zoom} s)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uav_height_control_zoom.png', dpi=150, bbox_inches='tight')
print("Zoomed figure saved as: uav_height_control_zoom.png")
plt.show()

print("\nSimulation completed!")

