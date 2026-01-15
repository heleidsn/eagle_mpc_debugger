import numpy as np
import matplotlib.pyplot as plt

# ========== System and control parameters ==========
m = 1.5
zeta = 0.7
omega_n = 4.0

Kp = m * omega_n**2
Kd = 2.0 * m * zeta * omega_n

# Reference signal: analytical sine
A = 1.0        # Amplitude
f = 0.3        # Hz
omega_ref = 2.0 * np.pi * f

# Simulation settings
dt = 0.001
T = 2.0  # Focus on initial period
N = int(T / dt)
t = np.linspace(0, T, N)

# Analytical reference
z_ref_ana      = A * np.sin(omega_ref * t)
z_ref_dot_ana  = A * omega_ref * np.cos(omega_ref * t)
z_ref_ddot_ana = -A * omega_ref**2 * np.sin(omega_ref * t)

# =====================================================================
# PD+FF control with initial condition mismatch
# =====================================================================
z_pdff     = 0.0  # Initial position
zdot_pdff  = 0.0  # Initial velocity
z_hist_pdff    = np.zeros(N)
zdot_hist_pdff = np.zeros(N)
err_hist_pdff  = np.zeros(N)
err_dot_hist_pdff = np.zeros(N)
u_ff_hist = np.zeros(N)
u_pd_hist = np.zeros(N)
u_total_hist = np.zeros(N)

for k in range(N):
    zr     = z_ref_ana[k]
    zrdot  = z_ref_dot_ana[k]
    zrddot = z_ref_ddot_ana[k]

    e    = zr - z_pdff
    edot = zrdot - zdot_pdff

    # Separate FF and PD components
    u_ff = m * zrddot
    u_pd = Kp * e + Kd * edot
    u_total = u_ff + u_pd

    u_ff_hist[k] = u_ff
    u_pd_hist[k] = u_pd
    u_total_hist[k] = u_total

    zddot_pdff = u_total / m
    zdot_pdff += zddot_pdff * dt
    z_pdff    += zdot_pdff * dt

    z_hist_pdff[k]   = z_pdff
    zdot_hist_pdff[k] = zdot_pdff
    err_hist_pdff[k] = e
    err_dot_hist_pdff[k] = edot

# =====================================================================
# Analysis: Why large initial error?
# =====================================================================
print("="*70)
print("Analysis: Why PD+FF shows large initial error?")
print("="*70)
print(f"\nInitial conditions:")
print(f"  System: z(0) = 0.0 m, zdot(0) = 0.0 m/s")
print(f"  Reference: z_ref(0) = {z_ref_ana[0]:.4f} m, z_ref_dot(0) = {z_ref_dot_ana[0]:.4f} m/s")
print(f"\nInitial errors:")
print(f"  Position error e(0) = {err_hist_pdff[0]:.4f} m")
print(f"  Velocity error edot(0) = {err_dot_hist_pdff[0]:.4f} m/s")
print(f"\nInitial control components:")
print(f"  FF component: u_ff(0) = {u_ff_hist[0]:.4f} N")
print(f"  PD component: u_pd(0) = {u_pd_hist[0]:.4f} N")
print(f"  Total control: u(0) = {u_total_hist[0]:.4f} N")
print("\n" + "-"*70)
print("Key insights:")
print("-"*70)
print("1. Initial position error is ZERO (both start at z=0)")
print("2. Initial velocity error is LARGE (system at rest, ref moving)")
print("3. FF provides reference acceleration, but system needs time to")
print("   accelerate from zero velocity to reference velocity")
print("4. PD term tries to correct velocity error, but this takes time")
print("5. During this transient, position error accumulates")
print("="*70)

# =====================================================================
# Visualization
# =====================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Row 1: Position and velocity tracking
ax = axes[0, 0]
ax.plot(t, z_ref_ana, 'k-', label='z_ref', linewidth=2)
ax.plot(t, z_hist_pdff, 'r--', label='z (PD+FF)', linewidth=2)
ax.set_title('Position Tracking (Initial Period)', fontweight='bold')
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 0.5])  # Focus on initial period

ax = axes[0, 1]
ax.plot(t, z_ref_dot_ana, 'k-', label='zdot_ref', linewidth=2)
ax.plot(t, zdot_hist_pdff, 'r--', label='zdot (PD+FF)', linewidth=2)
ax.set_title('Velocity Tracking (Initial Period)', fontweight='bold')
ax.set_xlabel('t (s)')
ax.set_ylabel('zdot (m/s)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 0.5])

# Row 2: Errors
ax = axes[1, 0]
ax.plot(t, err_hist_pdff, 'b-', linewidth=2)
ax.set_title('Position Error e = z_ref - z', fontweight='bold')
ax.set_xlabel('t (s)')
ax.set_ylabel('e (m)')
ax.grid(True)
ax.set_xlim([0, 0.5])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

ax = axes[1, 1]
ax.plot(t, err_dot_hist_pdff, 'b-', linewidth=2)
ax.set_title('Velocity Error edot = zdot_ref - zdot', fontweight='bold')
ax.set_xlabel('t (s)')
ax.set_ylabel('edot (m/s)')
ax.grid(True)
ax.set_xlim([0, 0.5])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

# Row 3: Control components
ax = axes[2, 0]
ax.plot(t, u_ff_hist, 'g-', label='u_FF = m * z_ref_ddot', linewidth=2)
ax.plot(t, u_pd_hist, 'r--', label='u_PD = Kp*e + Kd*edot', linewidth=2)
ax.set_title('Control Components (Initial Period)', fontweight='bold')
ax.set_xlabel('t (s)')
ax.set_ylabel('u (N)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 0.5])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

ax = axes[2, 1]
ax.plot(t, u_total_hist, 'b-', linewidth=2, label='u_total = u_FF + u_PD')
ax.set_title('Total Control Input', fontweight='bold')
ax.set_xlabel('t (s)')
ax.set_ylabel('u (N)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 0.5])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# =====================================================================
# Comparison: Perfect initial conditions vs zero initial conditions
# =====================================================================
print("\n" + "="*70)
print("Comparison: Perfect initial conditions vs zero initial conditions")
print("="*70)

# Case with perfect initial conditions
z_pdff_perfect = z_ref_ana[0]
zdot_pdff_perfect = z_ref_dot_ana[0]
z_hist_pdff_perfect = np.zeros(N)
err_hist_pdff_perfect = np.zeros(N)

for k in range(N):
    zr     = z_ref_ana[k]
    zrdot  = z_ref_dot_ana[k]
    zrddot = z_ref_ddot_ana[k]

    e    = zr - z_pdff_perfect
    edot = zrdot - zdot_pdff_perfect

    u = m * zrddot + Kp * e + Kd * edot

    zddot = u / m
    zdot_pdff_perfect += zddot * dt
    z_pdff_perfect    += zdot_pdff_perfect * dt

    z_hist_pdff_perfect[k]   = z_pdff_perfect
    err_hist_pdff_perfect[k] = e

# Compare initial errors
print(f"\nZero initial conditions:")
print(f"  Max |error| in first 0.5s: {np.max(np.abs(err_hist_pdff[:int(0.5/dt)])):.6f} m")
print(f"  RMSE in first 0.5s: {np.sqrt(np.mean(err_hist_pdff[:int(0.5/dt)]**2)):.6f} m")

print(f"\nPerfect initial conditions:")
print(f"  Max |error| in first 0.5s: {np.max(np.abs(err_hist_pdff_perfect[:int(0.5/dt)])):.6f} m")
print(f"  RMSE in first 0.5s: {np.sqrt(np.mean(err_hist_pdff_perfect[:int(0.5/dt)]**2)):.6f} m")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(t, z_ref_ana, 'k-', label='z_ref', linewidth=2)
ax.plot(t, z_hist_pdff, 'r--', label='z (zero IC)', linewidth=2)
ax.plot(t, z_hist_pdff_perfect, 'b:', label='z (perfect IC)', linewidth=2)
ax.set_title('Position Tracking: Effect of Initial Conditions', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 1.0])

ax = axes[1]
ax.plot(t, err_hist_pdff, 'r--', label='Error (zero IC)', linewidth=2)
ax.plot(t, err_hist_pdff_perfect, 'b:', label='Error (perfect IC)', linewidth=2)
ax.set_title('Tracking Error: Effect of Initial Conditions', fontweight='bold', fontsize=12)
ax.set_xlabel('t (s)')
ax.set_ylabel('e (m)')
ax.legend()
ax.grid(True)
ax.set_xlim([0, 1.0])
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Conclusion:")
print("="*70)
print("The large initial error in PD+FF is due to:")
print("1. Initial condition mismatch (system at rest, reference moving)")
print("2. FF alone cannot instantly match reference velocity")
print("3. PD needs time to correct the velocity error")
print("4. During this transient, position error accumulates")
print("5. Once velocity is matched, error decreases rapidly")
print("="*70)

