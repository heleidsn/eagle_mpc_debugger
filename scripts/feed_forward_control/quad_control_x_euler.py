import numpy as np
import matplotlib.pyplot as plt

# ========= 1. Parameters =========
m = 1.0      # mass (kg)
g = 9.81     # gravity (m/s^2)
Iyy = 0.02   # moment of inertia around y (kg*m^2), arbitrary

# ---- Position loop desired dynamics (for x and z) ----
zeta_pos = 0.9     # damping ratio
omega_n_pos = 2.0  # natural frequency (rad/s)

omega_n_pos_z = 2.0  # natural frequency (rad/s)

Kp_x = omega_n_pos**2
Kd_x = 2 * zeta_pos * omega_n_pos

Kp_z = omega_n_pos_z**2
Kd_z = 2 * zeta_pos * omega_n_pos_z

# ---- Attitude loop desired dynamics ----
zeta_att = 0.9
omega_n_att = 8.0

Kp_th = omega_n_att**2 * Iyy
Kd_th = 2 * zeta_att * omega_n_att * Iyy

print("Kp_x, Kd_x:", Kp_x, Kd_x)
print("Kp_z, Kd_z:", Kp_z, Kd_z)
print("Kp_th, Kd_th:", Kp_th, Kd_th)

# ========= 2. Simulation setup =========
dt = 0.001
Tsim = 10.0
N = int(Tsim / dt)
t = np.linspace(0, Tsim, N)

# Reference: x is sinusoidal, z is constant hover
A_x = 2.0     # 2 m amplitude
f_x = 0.15    # Hz
omega_x = 2 * np.pi * f_x

x_ref      = A_x * np.sin(omega_x * t)
x_ref_dot  = A_x * omega_x * np.cos(omega_x * t)
x_ref_ddot = -A_x * omega_x**2 * np.sin(omega_x * t)

z_ref      = np.zeros_like(t)  # hover at z = 0
z_ref_dot  = np.zeros_like(t)
z_ref_ddot = np.zeros_like(t)

# ========= 3. State arrays =========
x = np.zeros(N)
z = np.zeros(N)
theta = np.zeros(N)

xd = np.zeros(N)
zd = np.zeros(N)
thetad = np.zeros(N)

T_cmd_hist = np.zeros(N)
tau_hist = np.zeros(N)

# ========= 4. Simulation loop =========
for k in range(N - 1):
    # Current states
    xk, zk, thk = x[k], z[k], theta[k]
    xdk, zdk, thdk = xd[k], zd[k], thetad[k]

    # References at time k
    xr, xrd, xrdd = x_ref[k], x_ref_dot[k], x_ref_ddot[k]
    zr, zrd, zrdd = z_ref[k], z_ref_dot[k], z_ref_ddot[k]

    # ---- Outer loop: position PD + FF to get desired accelerations ----
    ex = xr - xk
    evx = xrd - xdk
    ax_cmd = xrdd + Kp_x * ex + Kd_x * evx   # desired horizontal acceleration

    ez = zr - zk
    evz = zrd - zdk
    az_cmd = zrdd + Kp_z * ez + Kd_z * evz   # desired vertical acceleration

    # ---- Convert desired accelerations to attitude & thrust commands (linearized) ----
    # Small-angle approximation:
    #   ax ≈ -g * theta  -> theta_cmd ≈ -ax_cmd / g
    #   az ≈ T/m - g     -> T_cmd ≈ m (az_cmd + g)
    theta_cmd = -ax_cmd / g
    T_cmd = m * (az_cmd + g)

    # ---- Inner loop: attitude PD to get torque ----
    e_th = theta_cmd - thk
    e_thd = 0.0 - thdk
    tau = Kp_th * e_th + Kd_th * e_thd

    # ---- Dynamics (2D quadrotor) ----
    # x_ddot = -(T/m) * sin(theta)
    # z_ddot =  (T/m) * cos(theta) - g
    xdd = -(T_cmd / m) * np.sin(thk)
    zdd =  (T_cmd / m) * np.cos(thk) - g
    thdd = tau / Iyy

    # ---- Integrate (Euler) ----
    xd[k+1] = xdk + xdd * dt
    zd[k+1] = zdk + zdd * dt
    thetad[k+1] = thdk + thdd * dt

    x[k+1] = xk + xd[k+1] * dt
    z[k+1] = zk + zd[k+1] * dt
    theta[k+1] = thk + thetad[k+1] * dt

    T_cmd_hist[k] = T_cmd
    tau_hist[k] = tau

T_cmd_hist[-1] = T_cmd_hist[-2]
tau_hist[-1] = tau_hist[-2]

# ========= 5. Plot results =========
plt.figure(figsize=(10, 12))

# X tracking
plt.subplot(4, 1, 1)
plt.plot(t, x_ref, label="x_ref")
plt.plot(t, x, linestyle="--", label="x")
plt.ylabel("x (m)")
plt.title("2D Quadrotor: Cascaded Position (x,z) + Attitude Control")
plt.legend()
plt.grid(True)

# X velocity
plt.subplot(4, 1, 2)
plt.plot(t, x_ref_dot, label="x_vel_ref")
plt.plot(t, xd, linestyle="--", label="x_vel")
plt.ylabel("x velocity (m/s)")
plt.legend()
plt.grid(True)

# Z tracking
plt.subplot(4, 1, 3)
plt.plot(t, z_ref, label="z_ref")
plt.plot(t, z, linestyle="--", label="z")
plt.ylabel("z (m)")
plt.legend()
plt.grid(True)

# Theta
plt.subplot(4, 1, 4)
plt.plot(t, np.degrees(theta), label="theta")
plt.ylabel("theta (degrees)")
plt.xlabel("time (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
