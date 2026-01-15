import numpy as np
import matplotlib.pyplot as plt

# ========== System and control parameters ==========
m = 1.5
zeta = 0.7
omega_n = 4.0

Kp = m * omega_n**2
Kd = 2.0 * m * zeta * omega_n

# Reference signal parameters
A = 1.0        # Amplitude
f = 0.3        # Hz
omega_ref = 2.0 * np.pi * f

# Simulation settings
dt = 0.001
T = 10.0
N = int(T / dt)
t = np.linspace(0, T, N)

# =====================================================================
# Method 3: Polynomial trajectory (5th order) - matches IC and final conditions
# =====================================================================
# Design polynomial: z(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
# IC: z(0)=0, zdot(0)=0, zddot(0)=0
# Final: z(T_trans)=A*sin(ωT_trans), zdot(T_trans)=A*ω*cos(ωT_trans), zddot(T_trans)=-A*ω^2*sin(ωT_trans)
T_transition = 2.0

# Target at transition end
z_target = A * np.sin(omega_ref * T_transition)
zdot_target = A * omega_ref * np.cos(omega_ref * T_transition)
zddot_target = -A * omega_ref**2 * np.sin(omega_ref * T_transition)

# Solve for polynomial coefficients
# z(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
# z(0) = 0, zdot(0) = 0, zddot(0) = 0
# z(T_trans) = z_target, zdot(T_trans) = zdot_target, zddot(T_trans) = zddot_target
T_tr = T_transition
A_mat = np.array([
    [1, 0, 0, 0, 0, 0],  # z(0) = 0
    [0, 1, 0, 0, 0, 0],  # zdot(0) = 0
    [0, 0, 2, 0, 0, 0],  # zddot(0) = 0
    [1, T_tr, T_tr**2, T_tr**3, T_tr**4, T_tr**5],  # z(T_trans)
    [0, 1, 2*T_tr, 3*T_tr**2, 4*T_tr**3, 5*T_tr**4],  # zdot(T_trans)
    [0, 0, 2, 6*T_tr, 12*T_tr**2, 20*T_tr**3]  # zddot(T_trans)
])
b_vec = np.array([0, 0, 0, z_target, zdot_target, zddot_target])
coeffs = np.linalg.solve(A_mat, b_vec)

# Compute polynomial trajectory
z_ref_poly = np.zeros(N)
z_ref_dot_poly = np.zeros(N)
z_ref_ddot_poly = np.zeros(N)

for i, ti in enumerate(t):
    if ti <= T_transition:
        z_ref_poly[i] = coeffs[0] + coeffs[1]*ti + coeffs[2]*ti**2 + \
                        coeffs[3]*ti**3 + coeffs[4]*ti**4 + coeffs[5]*ti**5
        z_ref_dot_poly[i] = coeffs[1] + 2*coeffs[2]*ti + 3*coeffs[3]*ti**2 + \
                           4*coeffs[4]*ti**3 + 5*coeffs[5]*ti**4
        z_ref_ddot_poly[i] = 2*coeffs[2] + 6*coeffs[3]*ti + \
                            12*coeffs[4]*ti**2 + 20*coeffs[5]*ti**3
    else:
        # After transition, use sinusoidal
        z_ref_poly[i] = A * np.sin(omega_ref * ti)
        z_ref_dot_poly[i] = A * omega_ref * np.cos(omega_ref * ti)
        z_ref_ddot_poly[i] = -A * omega_ref**2 * np.sin(omega_ref * ti)

# =====================================================================
# Simulate PD only control
# =====================================================================
z_pd = 0.0
zdot_pd = 0.0
z_hist_pd = np.zeros(N)
err_hist_pd = np.zeros(N)
u_pd_hist = np.zeros(N)

for k in range(N):
    zr = z_ref_poly[k]
    zrdot = z_ref_dot_poly[k]
    
    e = zr - z_pd
    edot = zrdot - zdot_pd
    
    # PD only: no feedforward
    u_pd = Kp * e + Kd * edot
    u_pd_hist[k] = u_pd
    
    zddot_pd = u_pd / m
    zdot_pd += zddot_pd * dt
    z_pd += zdot_pd * dt
    
    z_hist_pd[k] = z_pd
    err_hist_pd[k] = e

# =====================================================================
# Simulate PD+FF control
# =====================================================================
z_pdff = 0.0
zdot_pdff = 0.0
z_hist_pdff = np.zeros(N)
err_hist_pdff = np.zeros(N)
u_ff_hist = np.zeros(N)
u_pd_comp_hist = np.zeros(N)
u_total_hist = np.zeros(N)

for k in range(N):
    zr = z_ref_poly[k]
    zrdot = z_ref_dot_poly[k]
    zrddot = z_ref_ddot_poly[k]
    
    e = zr - z_pdff
    edot = zrdot - zdot_pdff
    
    # PD + FF
    u_ff = m * zrddot
    u_pd_comp = Kp * e + Kd * edot
    u_total = u_ff + u_pd_comp
    
    u_ff_hist[k] = u_ff
    u_pd_comp_hist[k] = u_pd_comp
    u_total_hist[k] = u_total
    
    zddot_pdff = u_total / m
    zdot_pdff += zddot_pdff * dt
    z_pdff += zdot_pdff * dt
    
    z_hist_pdff[k] = z_pdff
    err_hist_pdff[k] = e

# =====================================================================
# Visualization
# =====================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Row 1: Reference and tracking
ax = axes[0, 0]
ax.plot(t, z_ref_poly, 'k-', label='Reference (Polynomial)', linewidth=2)
ax.plot(t, z_hist_pd, 'r--', label='PD only', linewidth=2)
ax.plot(t, z_hist_pdff, 'b-.', label='PD + FF', linewidth=2)
ax.set_title('Position Tracking', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.grid(True)

ax = axes[0, 1]
ax.plot(t, z_ref_dot_poly, 'k-', label='Reference velocity', linewidth=2)
# Compute actual velocities
zdot_hist_pd = np.gradient(z_hist_pd, dt)
zdot_hist_pdff = np.gradient(z_hist_pdff, dt)
ax.plot(t, zdot_hist_pd, 'r--', label='PD only', linewidth=2)
ax.plot(t, zdot_hist_pdff, 'b-.', label='PD + FF', linewidth=2)
ax.set_title('Velocity Tracking', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('zdot (m/s)')
ax.legend()
ax.grid(True)

# Row 2: Errors
ax = axes[1, 0]
ax.plot(t, err_hist_pd, 'r--', label='PD only', linewidth=2)
ax.plot(t, err_hist_pdff, 'b-.', label='PD + FF', linewidth=2)
ax.set_title('Position Error', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('e = z_ref - z (m)')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

ax = axes[1, 1]
ax.plot(t, np.abs(err_hist_pd), 'r--', label='|Error| PD only', linewidth=2)
ax.plot(t, np.abs(err_hist_pdff), 'b-.', label='|Error| PD + FF', linewidth=2)
ax.set_title('Absolute Position Error', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('|e| (m)')
ax.legend()
ax.grid(True)

# Row 3: Control inputs
ax = axes[2, 0]
ax.plot(t, u_pd_hist, 'r--', label='u (PD only)', linewidth=2)
ax.plot(t, u_total_hist, 'b-', label='u_total (PD + FF)', linewidth=2)
ax.plot(t, u_ff_hist, 'g:', label='u_FF component', linewidth=1.5, alpha=0.7)
ax.set_title('Control Inputs', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('u (N)')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

ax = axes[2, 1]
ax.plot(t, u_ff_hist, 'g-', label='u_FF = m * z_ref_ddot', linewidth=2)
ax.plot(t, u_pd_comp_hist, 'm--', label='u_PD = Kp*e + Kd*edot', linewidth=2)
ax.set_title('PD+FF Control Components', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('u (N)')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================================
# Detailed comparison: Initial period
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Initial period zoom
t_zoom = 1.0
idx_zoom = int(t_zoom / dt)

ax = axes[0, 0]
ax.plot(t[:idx_zoom], z_ref_poly[:idx_zoom], 'k-', label='Reference', linewidth=2)
ax.plot(t[:idx_zoom], z_hist_pd[:idx_zoom], 'r--', label='PD only', linewidth=2)
ax.plot(t[:idx_zoom], z_hist_pdff[:idx_zoom], 'b-.', label='PD + FF', linewidth=2)
ax.set_title('Position Tracking (Initial 1s)', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.grid(True)

ax = axes[0, 1]
ax.plot(t[:idx_zoom], err_hist_pd[:idx_zoom], 'r--', label='PD only', linewidth=2)
ax.plot(t[:idx_zoom], err_hist_pdff[:idx_zoom], 'b-.', label='PD + FF', linewidth=2)
ax.set_title('Position Error (Initial 1s)', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('e (m)')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

ax = axes[1, 0]
ax.plot(t[:idx_zoom], u_pd_hist[:idx_zoom], 'r--', label='u (PD only)', linewidth=2)
ax.plot(t[:idx_zoom], u_total_hist[:idx_zoom], 'b-', label='u_total (PD + FF)', linewidth=2)
ax.plot(t[:idx_zoom], u_ff_hist[:idx_zoom], 'g:', label='u_FF', linewidth=1.5, alpha=0.7)
ax.set_title('Control Inputs (Initial 1s)', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('u (N)')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

# Error statistics
ax = axes[1, 1]
methods = ['PD only', 'PD + FF']
rmse_pd = np.sqrt(np.mean(err_hist_pd**2))
rmse_pdff = np.sqrt(np.mean(err_hist_pdff**2))
max_err_pd = np.max(np.abs(err_hist_pd))
max_err_pdff = np.max(np.abs(err_hist_pdff))

x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x_pos - width/2, [rmse_pd, rmse_pdff], width, label='RMSE', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, [max_err_pd, max_err_pdff], width, label='Max |Error|', alpha=0.8)
ax.set_xlabel('Control Method')
ax.set_ylabel('Error (m)')
ax.set_title('Error Statistics Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, axis='y')

# Add value labels
for i, (rmse, max_err) in enumerate([(rmse_pd, max_err_pd), (rmse_pdff, max_err_pdff)]):
    ax.text(i - width/2, rmse, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, max_err, f'{max_err:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# =====================================================================
# Statistics
# =====================================================================
print("="*70)
print("PD vs PD+FF Comparison (Polynomial Reference Trajectory)")
print("="*70)

print("\nInitial Conditions:")
print(f"  System: z(0) = 0.0, zdot(0) = 0.0")
print(f"  Reference: z_ref(0) = {z_ref_poly[0]:.6f}, zdot_ref(0) = {z_ref_dot_poly[0]:.6f}")
print(f"  ✓ Perfect initial condition matching!")

print("\n" + "-"*70)
print("Error Statistics (Full Simulation):")
print("-"*70)
print(f"PD only control:")
print(f"  RMSE: {rmse_pd:.6f} m")
print(f"  Max |Error|: {max_err_pd:.6f} m")
print(f"  Mean |Error|: {np.mean(np.abs(err_hist_pd)):.6f} m")

print(f"\nPD+FF control:")
print(f"  RMSE: {rmse_pdff:.6f} m")
print(f"  Max |Error|: {max_err_pdff:.6f} m")
print(f"  Mean |Error|: {np.mean(np.abs(err_hist_pdff)):.6f} m")

print("\n" + "-"*70)
print("Improvement with Feedforward:")
print("-"*70)
rmse_improvement = (1 - rmse_pdff/rmse_pd) * 100
max_err_improvement = (1 - max_err_pdff/max_err_pd) * 100
mean_err_improvement = (1 - np.mean(np.abs(err_hist_pdff))/np.mean(np.abs(err_hist_pd))) * 100

print(f"  RMSE improvement: {rmse_improvement:.2f}%")
print(f"  Max error improvement: {max_err_improvement:.2f}%")
print(f"  Mean error improvement: {mean_err_improvement:.2f}%")

print("\n" + "-"*70)
print("Error Statistics (Initial 1 second):")
print("-"*70)
rmse_pd_init = np.sqrt(np.mean(err_hist_pd[:idx_zoom]**2))
rmse_pdff_init = np.sqrt(np.mean(err_hist_pdff[:idx_zoom]**2))
max_err_pd_init = np.max(np.abs(err_hist_pd[:idx_zoom]))
max_err_pdff_init = np.max(np.abs(err_hist_pdff[:idx_zoom]))

print(f"PD only (initial 1s):")
print(f"  RMSE: {rmse_pd_init:.6f} m")
print(f"  Max |Error|: {max_err_pd_init:.6f} m")

print(f"\nPD+FF (initial 1s):")
print(f"  RMSE: {rmse_pdff_init:.6f} m")
print(f"  Max |Error|: {max_err_pdff_init:.6f} m")

print("\n" + "-"*70)
print("Control Effort Comparison:")
print("-"*70)
u_rms_pd = np.sqrt(np.mean(u_pd_hist**2))
u_rms_pdff = np.sqrt(np.mean(u_total_hist**2))
u_max_pd = np.max(np.abs(u_pd_hist))
u_max_pdff = np.max(np.abs(u_total_hist))

print(f"PD only:")
print(f"  RMS control: {u_rms_pd:.4f} N")
print(f"  Max |control|: {u_max_pd:.4f} N")

print(f"\nPD+FF:")
print(f"  RMS control: {u_rms_pdff:.4f} N")
print(f"  Max |control|: {u_max_pdff:.4f} N")
print(f"  RMS FF component: {np.sqrt(np.mean(u_ff_hist**2)):.4f} N")
print(f"  RMS PD component: {np.sqrt(np.mean(u_pd_comp_hist**2)):.4f} N")

print("\n" + "="*70)
print("Key Observations:")
print("="*70)
print("1. With matched initial conditions, both controllers start with zero error")
print("2. FF component provides reference acceleration, reducing tracking error")
print("3. PD component in PD+FF is smaller (only corrects deviations)")
print("4. PD+FF achieves better tracking performance with lower steady-state error")
print("="*70)

