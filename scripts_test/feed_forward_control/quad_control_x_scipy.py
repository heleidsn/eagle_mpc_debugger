import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ========= 1. 系统与控制参数 =========
m = 1.0      # 质量 (kg)
g = 9.81     # 重力 (m/s^2)
Iyy = 0.02   # 绕 y 轴转动惯量 (kg*m^2) 随便取一个合理值

# ---- 外环：位置期望闭环指标（x 和 z 共用一套）----
zeta_pos = 0.9
omega_n_pos = 2.0   # rad/s

Kp_x = omega_n_pos**2
Kd_x = 2 * zeta_pos * omega_n_pos

Kp_z = omega_n_pos**2
Kd_z = 2 * zeta_pos * omega_n_pos

# ---- 内环：俯仰角闭环指标 ----
zeta_att = 0.9
omega_n_att = 8.0   # rad/s，姿态带宽明显比位置高

Kp_th = omega_n_att**2 * Iyy
Kd_th = 2 * zeta_att * omega_n_att * Iyy

print("Kp_x, Kd_x:", Kp_x, Kd_x)
print("Kp_z, Kd_z:", Kp_z, Kd_z)
print("Kp_th, Kd_th:", Kp_th, Kd_th)

# ========= 2. 参考轨迹定义 =========
A_x = 2.0        # x 方向正弦轨迹振幅 2 m
f_x = 0.15       # Hz
omega_x = 2 * np.pi * f_x

def ref_traj(t):
    """给定时间 t，返回参考 x、z 及其一二阶导数"""
    x_ref     = A_x * np.sin(omega_x * t)
    x_ref_dot = A_x * omega_x * np.cos(omega_x * t)
    x_ref_dd  = -A_x * omega_x**2 * np.sin(omega_x * t)

    # z 方向保持悬停 0
    z_ref     = 0.0
    z_ref_dot = 0.0
    z_ref_dd  = 0.0

    return x_ref, x_ref_dot, x_ref_dd, z_ref, z_ref_dot, z_ref_dd

# ========= 3. 控制律 =========
def controller(t, s):
    """
    controller(t, s) -> (T_cmd, tau)
    s = [x, z, theta, xdot, zdot, thetadot]
    """
    x, z, th, xd, zd, thd = s

    # --- 参考轨迹 ---
    xr, xrd, xrdd, zr, zrd, zrdd = ref_traj(t)

    # --- 外环：位置 PD + 前馈，加速度指令 ---
    ex   = xr  - x
    evx  = xrd - xd
    ax_cmd = xrdd + Kp_x * ex + Kd_x * evx   # 期望水平加速度

    ez   = zr  - z
    evz  = zrd - zd
    az_cmd = zrdd + Kp_z * ez + Kd_z * evz   # 期望竖直加速度

    # --- 将加速度指令转换为姿态 & 总推力命令（小角度近似） ---
    # ax ≈ -g * theta       -> theta_cmd ≈ -ax_cmd / g
    # az ≈ T/m - g          -> T_cmd ≈ m (az_cmd + g)
    theta_cmd = -ax_cmd / g
    T_cmd = m * (az_cmd + g)

    # --- 内环：俯仰角 PD 控制，输出力矩 tau ---
    e_th  = theta_cmd - th
    e_thd = 0.0 - thd
    tau = Kp_th * e_th + Kd_th * e_thd

    return T_cmd, tau

# ========= 4. 动力学方程（给 solve_ivp 用） =========
def quad2d_dynamics(t, s):
    """
    s = [x, z, theta, xdot, zdot, thetadot]
    返回 ds/dt
    """
    x, z, th, xd, zd, thd = s

    # 控制输入
    T_cmd, tau = controller(t, s)

    # 简化 2D 四旋翼动力学
    # x_ddot = -(T/m) * sin(theta)
    # z_ddot =  (T/m) * cos(theta) - g
    xdd  = -(T_cmd / m) * np.sin(th)
    zdd  =  (T_cmd / m) * np.cos(th) - g
    thdd = tau / Iyy

    # 按状态顺序返回导数
    return [xd, zd, thd, xdd, zdd, thdd]

# ========= 5. 使用 solve_ivp 积分 =========
Tsim = 10.0
t_eval = np.linspace(0, Tsim, 10001)

s0 = [0.0,   # x(0)
      0.0,   # z(0)
      0.0,   # theta(0)
      0.0,   # xdot(0)
      0.0,   # zdot(0)
      0.0]   # thetadot(0)

sol = solve_ivp(
    fun=quad2d_dynamics,
    t_span=(0.0, Tsim),
    y0=s0,
    t_eval=t_eval,
    max_step=0.001,
    method="RK45",  # Euler is too slow, RK45 is better
    rtol=1e-6,
    atol=1e-9,
)

t = sol.t
x, z, theta, xd, zd, thetad = sol.y

# 再算一次参考轨迹用于画图
x_ref_plot, _, _, z_ref_plot, _, _ = ref_traj(t)
# z_ref is constant (0.0), so make it an array matching t
z_ref_plot = np.zeros_like(t)

# ========= 6. 画图 =========
plt.figure(figsize=(10, 10))

# X tracking
plt.subplot(3, 1, 1)
plt.plot(t, x_ref_plot, label="x_ref")
plt.plot(t, x, "--", label="x")
plt.ylabel("x (m)")
plt.title("2D Quadrotor with solve_ivp: Cascaded Position (x,z) + Attitude Control")
plt.legend()
plt.grid(True)

# Z tracking
plt.subplot(3, 1, 2)
plt.plot(t, z_ref_plot, label="z_ref")
plt.plot(t, z, "--", label="z")
plt.ylabel("z (m)")
plt.legend()
plt.grid(True)

# Theta
plt.subplot(3, 1, 3)
plt.plot(t, np.degrees(theta), label="theta")
plt.ylabel("theta (degrees)")
plt.xlabel("time (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
