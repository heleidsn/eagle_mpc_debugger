import numpy as np
import matplotlib.pyplot as plt

# ========== System and Control Parameters ==========
m = 1.5
zeta = 0.7
omega_n = 4.0

Kp = m * omega_n**2
Kd = 2.0 * m * zeta * omega_n

print("Kp =", Kp, "Kd =", Kd)

# Reference signal: analytical sine
A = 1.0        # Amplitude
f = 0.3        # Hz
omega_ref = 2.0 * np.pi * f

# =====================================================================
# MPC trajectory dt > Controller dt (demonstrating time step mismatch effects)
# Scenario: MPC trajectory generated with dt_mpc=0.01s, controller runs with dt_ctrl=0.001s
# =====================================================================
print("\n" + "="*60)
print("MPC Trajectory dt > Controller dt (Time Step Mismatch Analysis)")
print("="*60)

dt_mpc = 0.01      # MPC trajectory time step
dt_ctrl = 0.001   # Controller time step
T_sim = 0.1        # Simulation duration

# Generate MPC trajectory (using larger dt)
t_mpc = np.arange(0, T_sim, dt_mpc)
N_mpc = len(t_mpc)
z_ref_mpc = np.zeros(N_mpc)
z_ref_dot_mpc = np.zeros(N_mpc)
z_ref_ddot_mpc = np.zeros(N_mpc)

# MPC trajectory: generated using Euler integration (simulating MPC optimized trajectory)
z_ref_mpc[0] = 0.0
z_ref_dot_mpc[0] = 0.0
for k in range(N_mpc - 1):
    tk = t_mpc[k]
    # Drive with analytical acceleration (simulating MPC reference acceleration)
    zrddot = -A * omega_ref**2 * np.sin(omega_ref * tk)
    z_ref_ddot_mpc[k] = zrddot
    
    # Euler integration to generate MPC trajectory
    z_ref_dot_mpc[k+1] = z_ref_dot_mpc[k] + zrddot * dt_mpc
    z_ref_mpc[k+1] = z_ref_mpc[k] + z_ref_dot_mpc[k] * dt_mpc

# Controller runs (using smaller dt)
t_ctrl = np.arange(0, T_sim, dt_ctrl)
N_ctrl = len(t_ctrl)

# =====================================================================
# Method 1: Zero-Order Hold (ZOH) - directly use nearest MPC trajectory point
# =====================================================================
z_ctrl_zoh = 0.0
zdot_ctrl_zoh = 0.0
z_hist_zoh = np.zeros(N_ctrl)
err_hist_zoh = np.zeros(N_ctrl)
z_ref_zoh = np.zeros(N_ctrl)

for k in range(N_ctrl):
    tk = t_ctrl[k]
    
    # Find corresponding MPC trajectory index (zero-order hold)
    mpc_idx = min(int(tk / dt_mpc), N_mpc - 1)
    zr = z_ref_mpc[mpc_idx]
    zrddot = z_ref_ddot_mpc[mpc_idx]
    z_ref_zoh[k] = zr
    
    # Feedforward control
    u = m * zrddot
    
    # Controller dynamics (using smaller dt)
    zddot = u / m
    zdot_ctrl_zoh += zddot * dt_ctrl
    z_ctrl_zoh += zdot_ctrl_zoh * dt_ctrl
    
    z_hist_zoh[k] = z_ctrl_zoh
    err_hist_zoh[k] = zr - z_ctrl_zoh

# =====================================================================
# Method 2: Linear Interpolation - interpolate between MPC trajectory points
# =====================================================================
z_ctrl_interp = 0.0
zdot_ctrl_interp = 0.0
z_hist_interp = np.zeros(N_ctrl)
err_hist_interp = np.zeros(N_ctrl)
z_ref_interp = np.zeros(N_ctrl)

for k in range(N_ctrl):
    tk = t_ctrl[k]
    
    # Linear interpolation to find reference value
    mpc_idx = tk / dt_mpc
    idx_low = int(mpc_idx)
    idx_high = min(idx_low + 1, N_mpc - 1)
    alpha = mpc_idx - idx_low
    
    if idx_low < N_mpc - 1:
        zr = (1 - alpha) * z_ref_mpc[idx_low] + alpha * z_ref_mpc[idx_high]
        zrddot = (1 - alpha) * z_ref_ddot_mpc[idx_low] + alpha * z_ref_ddot_mpc[idx_high]
    else:
        zr = z_ref_mpc[idx_low]
        zrddot = z_ref_ddot_mpc[idx_low]
    
    z_ref_interp[k] = zr
    
    # Feedforward control
    u = m * zrddot
    
    # Controller dynamics (using smaller dt)
    zddot = u / m
    zdot_ctrl_interp += zddot * dt_ctrl
    z_ctrl_interp += zdot_ctrl_interp * dt_ctrl
    
    z_hist_interp[k] = z_ctrl_interp
    err_hist_interp[k] = zr - z_ctrl_interp

# =====================================================================
# Method 3: Ideal case - Controller dt = MPC dt (as comparison baseline)
# =====================================================================
z_ctrl_match = 0.0
zdot_ctrl_match = 0.0
z_hist_match = np.zeros(N_mpc)
err_hist_match = np.zeros(N_mpc)

for k in range(N_mpc):
    zr = z_ref_mpc[k]
    zrddot = z_ref_ddot_mpc[k]
    
    # Feedforward control
    u = m * zrddot
    
    # Controller dynamics (using same dt)
    zddot = u / m
    zdot_ctrl_match += zddot * dt_mpc
    z_ctrl_match += zdot_ctrl_match * dt_mpc
    
    z_hist_match[k] = z_ctrl_match
    err_hist_match[k] = zr - z_ctrl_match

# =====================================================================
# Plotting Comparison
# =====================================================================
plt.figure(figsize=(16, 10))

# Position comparison
plt.subplot(3, 2, 1)
plt.plot(t_mpc, z_ref_mpc, 'ko-', markersize=4, label='MPC Trajectory (dt=0.01s)', linewidth=2)
plt.plot(t_ctrl, z_ref_zoh, 'r--', alpha=0.5, label='ZOH Reference', linewidth=1)
plt.plot(t_ctrl, z_hist_zoh, 'r-', label='ZOH Control', linewidth=2)
plt.plot(t_ctrl, z_hist_interp, 'b-', label='Interpolated Control', linewidth=2)
plt.title("Position Tracking: Controller dt < MPC Trajectory dt", fontsize=12, fontweight='bold')
plt.xlabel("t (s)")
plt.ylabel("z (m)")
plt.legend()
plt.grid(True)

# Error comparison
plt.subplot(3, 2, 2)
plt.plot(t_ctrl, err_hist_zoh, 'r-', label='ZOH Error', linewidth=2)
plt.plot(t_ctrl, err_hist_interp, 'b-', label='Interpolation Error', linewidth=2)
plt.plot(t_mpc, err_hist_match, 'g-', label='dt Matched Error', linewidth=2)
plt.title("Tracking Error Comparison", fontsize=12, fontweight='bold')
plt.xlabel("t (s)")
plt.ylabel("e (m)")
plt.legend()
plt.grid(True)

# Reference trajectory comparison (showing interpolation effect)
plt.subplot(3, 2, 3)
plt.plot(t_mpc, z_ref_mpc, 'ko-', markersize=6, label='MPC Trajectory Points', linewidth=2)
plt.plot(t_ctrl[::10], z_ref_zoh[::10], 'r.', markersize=3, label='ZOH Reference (sampled)', alpha=0.7)
plt.plot(t_ctrl[::10], z_ref_interp[::10], 'b.', markersize=3, label='Interpolated Reference (sampled)', alpha=0.7)
plt.title("Reference Trajectory: ZOH vs Linear Interpolation", fontsize=12, fontweight='bold')
plt.xlabel("t (s)")
plt.ylabel("z_ref (m)")
plt.legend()
plt.grid(True)

# Error statistics
plt.subplot(3, 2, 4)
methods = ["ZOH", "Linear Interpolation", "dt Matched"]
rmse_zoh = np.sqrt(np.mean(err_hist_zoh**2))
rmse_interp = np.sqrt(np.mean(err_hist_interp**2))
rmse_match = np.sqrt(np.mean(err_hist_match**2))
max_err_zoh = np.max(np.abs(err_hist_zoh))
max_err_interp = np.max(np.abs(err_hist_interp))
max_err_match = np.max(np.abs(err_hist_match))

x_pos = np.arange(len(methods))
width = 0.35

plt.bar(x_pos - width/2, [rmse_zoh, rmse_interp, rmse_match], width, label='RMSE', alpha=0.8)
plt.bar(x_pos + width/2, [max_err_zoh, max_err_interp, max_err_match], width, label='Max |Error|', alpha=0.8)
plt.xlabel("Method")
plt.ylabel("Error (m)")
plt.title("Error Statistics Comparison", fontsize=12, fontweight='bold')
plt.xticks(x_pos, methods)
plt.legend()
plt.grid(True, axis='y')

# Add value labels
for i, (rmse, max_err) in enumerate([(rmse_zoh, max_err_zoh), (rmse_interp, max_err_interp), (rmse_match, max_err_match)]):
    plt.text(i - width/2, rmse, f'{rmse:.6f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, max_err, f'{max_err:.6f}', ha='center', va='bottom', fontsize=8)

# 控制量对比
plt.subplot(3, 2, 5)
u_zoh = np.zeros(N_ctrl)
u_interp = np.zeros(N_ctrl)
for k in range(N_ctrl):
    tk = t_ctrl[k]
    mpc_idx = min(int(tk / dt_mpc), N_mpc - 1)
    u_zoh[k] = m * z_ref_ddot_mpc[mpc_idx]
    
    mpc_idx_float = tk / dt_mpc
    idx_low = int(mpc_idx_float)
    idx_high = min(idx_low + 1, N_mpc - 1)
    alpha = mpc_idx_float - idx_low
    if idx_low < N_mpc - 1:
        u_interp[k] = m * ((1 - alpha) * z_ref_ddot_mpc[idx_low] + alpha * z_ref_ddot_mpc[idx_high])
    else:
        u_interp[k] = m * z_ref_ddot_mpc[idx_low]

plt.plot(t_ctrl[::10], u_zoh[::10], 'r.', markersize=3, label='ZOH Control', alpha=0.7)
plt.plot(t_ctrl[::10], u_interp[::10], 'b.', markersize=3, label='Interpolated Control', alpha=0.7)
plt.plot(t_mpc, m * z_ref_ddot_mpc, 'ko-', markersize=4, label='MPC Reference Control', linewidth=2)
plt.title("Control Input Comparison", fontsize=12, fontweight='bold')
plt.xlabel("t (s)")
plt.ylabel("u (N)")
plt.legend()
plt.grid(True)

# Absolute error over time
plt.subplot(3, 2, 6)
plt.plot(t_ctrl, np.abs(err_hist_zoh), 'r-', label='|ZOH Error|', linewidth=2, alpha=0.7)
plt.plot(t_ctrl, np.abs(err_hist_interp), 'b-', label='|Interpolation Error|', linewidth=2, alpha=0.7)
plt.plot(t_mpc, np.abs(err_hist_match), 'g-', label='|dt Matched Error|', linewidth=2, alpha=0.7)
plt.title("Absolute Error Over Time", fontsize=12, fontweight='bold')
plt.xlabel("t (s)")
plt.ylabel("|e| (m)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================================================================
# Print Statistics
# =====================================================================
print(f"\nMPC Trajectory dt: {dt_mpc*1000:.1f} ms")
print(f"Controller dt: {dt_ctrl*1000:.1f} ms")
print(f"Time Step Ratio: {dt_mpc/dt_ctrl:.0f}:1")
print("\nError Statistics:")
print(f"  ZOH Method:")
print(f"    RMSE: {rmse_zoh:.6f} m")
print(f"    Max |Error|: {max_err_zoh:.6f} m")
print(f"  Linear Interpolation Method:")
print(f"    RMSE: {rmse_interp:.6f} m")
print(f"    Max |Error|: {max_err_interp:.6f} m")
print(f"  dt Matched (Ideal Case):")
print(f"    RMSE: {rmse_match:.6f} m")
print(f"    Max |Error|: {max_err_match:.6f} m")
print("\nMain Issues:")
print("  1. Control update lag: ZOH method keeps control constant between MPC trajectory points")
print("  2. Interpolation error: Linear interpolation may introduce high-frequency components")
print("  3. Numerical integration mismatch: Controller uses smaller dt for integration, but reference is based on larger dt")
print("  4. May lead to tracking error accumulation and system instability")
print("\nSolutions:")
print("  1. Use linear interpolation: Better than ZOH, but may still have errors")
print("  2. Use higher-order interpolation: e.g., cubic spline, but more computationally complex")
print("  3. Match time steps: Let controller dt = MPC trajectory dt (best solution)")
print("  4. Add feedback control: Use PD+FF combination to compensate errors with feedback")
print("="*60)

