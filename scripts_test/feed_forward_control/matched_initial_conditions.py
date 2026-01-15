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
# Method 1: Cosine-based reference (1 - cos) - matches IC perfectly
# z(0) = 0, zdot(0) = 0
# =====================================================================
z_ref_cos = A * (1 - np.cos(omega_ref * t))
z_ref_dot_cos = A * omega_ref * np.sin(omega_ref * t)
z_ref_ddot_cos = A * omega_ref**2 * np.cos(omega_ref * t)

# =====================================================================
# Method 2: Smooth start with sigmoid transition
# =====================================================================
tau = 0.5  # Transition time
smooth_start = 1 / (1 + np.exp(-10 * (t - tau) / tau))
z_ref_smooth = A * np.sin(omega_ref * t) * smooth_start
# Compute derivatives numerically for smooth version
z_ref_dot_smooth = np.gradient(z_ref_smooth, dt)
z_ref_ddot_smooth = np.gradient(z_ref_dot_smooth, dt)

# =====================================================================
# Method 3: Polynomial trajectory (5th order) - matches IC and final conditions
# =====================================================================
# Design polynomial: z(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
# IC: z(0)=0, zdot(0)=0
# Final: z(T)=A*sin(ωT), zdot(T)=A*ω*cos(ωT), zddot(T)=-A*ω^2*sin(ωT)
# Use smooth transition to sinusoidal after initial period
T_transition = 2.0
t_transition = t[t <= T_transition]
N_trans = len(t_transition)

# For transition period, use polynomial
# After transition, smoothly connect to sinusoidal
z_ref_poly = np.zeros(N)
z_ref_dot_poly = np.zeros(N)
z_ref_ddot_poly = np.zeros(N)

# Target at transition end
z_target = A * np.sin(omega_ref * T_transition)
zdot_target = A * omega_ref * np.cos(omega_ref * T_transition)
zddot_target = -A * omega_ref**2 * np.sin(omega_ref * T_transition)

# Solve for polynomial coefficients
# z(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
# z(0) = 0, zdot(0) = 0, z(T_trans) = z_target, zdot(T_trans) = zdot_target
# zddot(0) = 0, zddot(T_trans) = zddot_target
# This gives us 6 conditions for 6 coefficients
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
# Original reference (mismatched IC)
# =====================================================================
z_ref_orig = A * np.sin(omega_ref * t)
z_ref_dot_orig = A * omega_ref * np.cos(omega_ref * t)
z_ref_ddot_orig = -A * omega_ref**2 * np.sin(omega_ref * t)

# =====================================================================
# Simulate PD+FF control for each reference
# =====================================================================
def simulate_pdff(z_ref, z_ref_dot, z_ref_ddot, label):
    z = 0.0
    zdot = 0.0
    z_hist = np.zeros(N)
    err_hist = np.zeros(N)
    
    for k in range(N):
        zr = z_ref[k]
        zrdot = z_ref_dot[k]
        zrddot = z_ref_ddot[k]
        
        e = zr - z
        edot = zrdot - zdot
        
        u = m * zrddot + Kp * e + Kd * edot
        
        zddot = u / m
        zdot += zddot * dt
        z += zdot * dt
        
        z_hist[k] = z
        err_hist[k] = e
    
    return z_hist, err_hist

# Simulate all cases
z_hist_orig, err_hist_orig = simulate_pdff(z_ref_orig, z_ref_dot_orig, z_ref_ddot_orig, "Original")
z_hist_cos, err_hist_cos = simulate_pdff(z_ref_cos, z_ref_dot_cos, z_ref_ddot_cos, "Cosine")
z_hist_smooth, err_hist_smooth = simulate_pdff(z_ref_smooth, z_ref_dot_smooth, z_ref_ddot_smooth, "Smooth")
z_hist_poly, err_hist_poly = simulate_pdff(z_ref_poly, z_ref_dot_poly, z_ref_ddot_poly, "Polynomial")

# =====================================================================
# Visualization
# =====================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Row 1: Reference trajectories
ax = axes[0, 0]
ax.plot(t, z_ref_orig, 'k--', label='Original: sin(ωt)', linewidth=2, alpha=0.7)
ax.plot(t, z_ref_cos, 'b-', label='Method 1: A(1-cos(ωt))', linewidth=2)
ax.plot(t, z_ref_smooth, 'g-', label='Method 2: Smooth start', linewidth=2)
ax.plot(t, z_ref_poly, 'r-', label='Method 3: Polynomial', linewidth=2)
ax.set_title('Reference Trajectories', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('z_ref (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 3.0])

ax = axes[0, 1]
ax.plot(t, z_ref_dot_orig, 'k--', label='Original', linewidth=2, alpha=0.7)
ax.plot(t, z_ref_dot_cos, 'b-', label='Method 1', linewidth=2)
ax.plot(t, z_ref_dot_smooth, 'g-', label='Method 2', linewidth=2)
ax.plot(t, z_ref_dot_poly, 'r-', label='Method 3', linewidth=2)
ax.set_title('Reference Velocities', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('zdot_ref (m/s)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 3.0])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

# Row 2: Tracking performance
ax = axes[1, 0]
ax.plot(t, z_ref_orig, 'k--', label='Reference', linewidth=1, alpha=0.5)
ax.plot(t, z_hist_orig, 'k:', label='Original ref', linewidth=2, alpha=0.7)
ax.plot(t, z_hist_cos, 'b-', label='Method 1', linewidth=2)
ax.plot(t, z_hist_smooth, 'g-', label='Method 2', linewidth=2)
ax.plot(t, z_hist_poly, 'r-', label='Method 3', linewidth=2)
ax.set_title('Position Tracking', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 3.0])

ax = axes[1, 1]
ax.plot(t, err_hist_orig, 'k:', label='Original ref', linewidth=2, alpha=0.7)
ax.plot(t, err_hist_cos, 'b-', label='Method 1', linewidth=2)
ax.plot(t, err_hist_smooth, 'g-', label='Method 2', linewidth=2)
ax.plot(t, err_hist_poly, 'r-', label='Method 3', linewidth=2)
ax.set_title('Tracking Error', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('e (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 3.0])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

# Row 3: Initial period zoom
ax = axes[2, 0]
ax.plot(t, z_ref_orig, 'k--', label='Reference', linewidth=1, alpha=0.5)
ax.plot(t, z_hist_orig, 'k:', label='Original ref', linewidth=2, alpha=0.7)
ax.plot(t, z_hist_cos, 'b-', label='Method 1', linewidth=2)
ax.plot(t, z_hist_smooth, 'g-', label='Method 2', linewidth=2)
ax.plot(t, z_hist_poly, 'r-', label='Method 3', linewidth=2)
ax.set_title('Initial Period (0-1s)', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 1.0])

ax = axes[2, 1]
ax.plot(t, err_hist_orig, 'k:', label='Original ref', linewidth=2, alpha=0.7)
ax.plot(t, err_hist_cos, 'b-', label='Method 1', linewidth=2)
ax.plot(t, err_hist_smooth, 'g-', label='Method 2', linewidth=2)
ax.plot(t, err_hist_poly, 'r-', label='Method 3', linewidth=2)
ax.set_title('Initial Error (0-1s)', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('e (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 1.0])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================================
# Statistics comparison
# =====================================================================
print("="*70)
print("Initial Condition Matching Analysis")
print("="*70)

print("\nInitial Conditions Check:")
print(f"  System IC: z(0) = 0.0, zdot(0) = 0.0")
print(f"\n  Original ref: z_ref(0) = {z_ref_orig[0]:.4f}, zdot_ref(0) = {z_ref_dot_orig[0]:.4f}")
print(f"  Method 1 (cos): z_ref(0) = {z_ref_cos[0]:.4f}, zdot_ref(0) = {z_ref_dot_cos[0]:.4f}")
print(f"  Method 2 (smooth): z_ref(0) = {z_ref_smooth[0]:.4f}, zdot_ref(0) = {z_ref_dot_smooth[0]:.4f}")
print(f"  Method 3 (poly): z_ref(0) = {z_ref_poly[0]:.4f}, zdot_ref(0) = {z_ref_dot_poly[0]:.4f}")

print("\n" + "-"*70)
print("Error Statistics (first 1 second):")
print("-"*70)

def compute_stats(err_hist, t_max=1.0):
    idx_max = int(t_max / dt)
    rmse = np.sqrt(np.mean(err_hist[:idx_max]**2))
    max_err = np.max(np.abs(err_hist[:idx_max]))
    return rmse, max_err

rmse_orig, max_err_orig = compute_stats(err_hist_orig)
rmse_cos, max_err_cos = compute_stats(err_hist_cos)
rmse_smooth, max_err_smooth = compute_stats(err_hist_smooth)
rmse_poly, max_err_poly = compute_stats(err_hist_poly)

print(f"  Original ref:  RMSE = {rmse_orig:.6f} m, Max |Error| = {max_err_orig:.6f} m")
print(f"  Method 1 (cos): RMSE = {rmse_cos:.6f} m, Max |Error| = {max_err_cos:.6f} m")
print(f"  Method 2 (smooth): RMSE = {rmse_smooth:.6f} m, Max |Error| = {max_err_smooth:.6f} m")
print(f"  Method 3 (poly): RMSE = {rmse_poly:.6f} m, Max |Error| = {max_err_poly:.6f} m")

print("\n" + "-"*70)
print("Improvement over original:")
print("-"*70)
print(f"  Method 1: RMSE reduction = {(1-rmse_cos/rmse_orig)*100:.2f}%, Max error reduction = {(1-max_err_cos/max_err_orig)*100:.2f}%")
print(f"  Method 2: RMSE reduction = {(1-rmse_smooth/rmse_orig)*100:.2f}%, Max error reduction = {(1-max_err_smooth/max_err_orig)*100:.2f}%")
print(f"  Method 3: RMSE reduction = {(1-rmse_poly/rmse_orig)*100:.2f}%, Max error reduction = {(1-max_err_poly/max_err_orig)*100:.2f}%")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("Method 1 (A(1-cos(ωt))):")
print("  - Perfect IC matching: z(0)=0, zdot(0)=0")
print("  - Simple and smooth")
print("  - Best for periodic motion starting from rest")
print("\nMethod 2 (Smooth start with sigmoid):")
print("  - Matches IC, then transitions to desired trajectory")
print("  - Flexible for any reference shape")
print("  - Good for connecting to arbitrary trajectories")
print("\nMethod 3 (Polynomial transition):")
print("  - Perfect IC matching with smooth transition")
print("  - Can match position, velocity, and acceleration at transition")
print("  - Best for precise trajectory planning")
print("="*70)

