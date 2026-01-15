import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. 系统和控制参数
# ===============================
m = 1.5  # 质量 (kg)

# 选闭环指标：自然频率和阻尼比
zeta = 0.7       # 阻尼比
omega_n = 10    # 自然频率 (rad/s)，可以理解为响应速度

# 根据标准二阶系统公式计算 PD 增益
Kp = m * omega_n**2          # Kp = m * ω_n^2
Kd = 2.0 * m * zeta * omega_n  # Kd = 2 * m * ζ * ω_n

print("Kp =", Kp, "Kd =", Kd)

# 参考轨迹参数：正弦高度
A = 1.0       # 振幅 1 m
f = 1 / (2 * np.pi)        # 频率 0.3 Hz
omega_ref = 2.0 * np.pi * f

# 仿真设置
dt = 0.001      # 步长 (s)
T = 10.0        # 总时间 (s)
N = int(T / dt) # 步数

t = np.linspace(0, T, N)

# ===============================
# 2. 预分配数组
# ===============================
# 状态: z, zdot
z_pd      = np.zeros(N)
zdot_pd   = np.zeros(N)
u_pd      = np.zeros(N)

z_ff      = np.zeros(N)
zdot_ff   = np.zeros(N)
u_ff      = np.zeros(N)

# 参考轨迹及其导数
z_ref      = A * np.sin(omega_ref * t)
z_ref_dot  = A * omega_ref * np.cos(omega_ref * t)
z_ref_ddot = -A * omega_ref**2 * np.sin(omega_ref * t)

# ===============================
# 3. 仿真循环
# ===============================
for k in range(N - 1):
    # 当前参考
    zr     = z_ref[k]
    zrdot  = z_ref_dot[k]
    zrddot = z_ref_ddot[k]

    # ------- 控制器 1：纯 PD -------
    e_pd    = zr - z_pd[k]
    edot_pd = zrdot - zdot_pd[k]
    u_pd[k] = Kp * e_pd + Kd * edot_pd

    # 系统动力学: m * zddot = u
    zddot_pd = u_pd[k] / m

    # 欧拉积分更新状态
    zdot_pd[k+1] = zdot_pd[k] + zddot_pd * dt
    z_pd[k+1]    = z_pd[k]    + zdot_pd[k] * dt

    # ------- 控制器 2：PD + 前馈 -------
    e_ff    = zr - z_ff[k]
    edot_ff = zrdot - zdot_ff[k]

    # 控制律: u = m * z_ref_ddot + Kp * e + Kd * edot
    u_ff[k] = m * zrddot + Kp * e_ff + Kd * edot_ff
    # u_ff[k] = m * zrddot

    zddot_ff = u_ff[k] / m

    zdot_ff[k+1] = zdot_ff[k] + zddot_ff * dt
    z_ff[k+1]    = z_ff[k]    + zdot_ff[k] * dt

# 最后一个控制输入就直接用倒数第二个
u_pd[-1] = u_pd[-2]
u_ff[-1] = u_ff[-2]

# ===============================
# 4. 画图对比
# ===============================
plt.figure(figsize=(10, 10))

# 位置追踪
plt.subplot(4, 1, 1)
plt.plot(t, z_ref, label="z_ref (sin)")
plt.plot(t, z_pd,  linestyle="--", label="PD")
plt.plot(t, z_ff,  linestyle=":",  label="PD + FF")
plt.ylabel("z (m)")
plt.title("Height Tracking: PD vs PD + Feedforward")
plt.legend()
plt.grid(True)

# 速度追踪
plt.subplot(4, 1, 2)
plt.plot(t, z_ref_dot, label="zdot_ref")
plt.plot(t, zdot_pd,  linestyle="--", label="PD")
plt.plot(t, zdot_ff,  linestyle=":",  label="PD + FF")
plt.ylabel("zdot (m/s)")
plt.legend()
plt.grid(True)

# 误差
plt.subplot(4, 1, 3)
plt.plot(t, z_ref - z_pd,  linestyle="--", label="Error (PD)")
plt.plot(t, z_ref - z_ff,  linestyle=":",  label="Error (PD + FF)")
plt.ylabel("Tracking Error (m)")
plt.legend()
plt.grid(True)

# 控制输入
plt.subplot(4, 1, 4)
plt.plot(t, u_pd,  linestyle="--", label="u_PD")
plt.plot(t, u_ff,  linestyle=":",  label="u_PD+FF")
plt.xlabel("Time (s)")
plt.ylabel("u (N)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
