import numpy as np
import matplotlib.pyplot as plt

# ==============
# 参数设置
# ==============
T  = 5.0          # 仿真总时间
dt = 0.001
N  = int(T / dt)

# 参考高度：简单阶跃，0 -> 1 m
def z_ref_func(t):
    return 1.0

def zd_ref_func(t):
    return 0.0

def zdd_ref_func(t):
    return 0.0

# 扰动：有界正弦
def dist_func(t):
    return 0.5 * np.sin(2.0 * np.pi * 0.5 * t)  # 幅值0.5，频率0.5Hz

# 滑模控制参数
lambda_smc = 3.0
lambda_tsm = 2.0
alpha_tsm  = 0.5

eta_smc = 3.0      # 到达律增益
eta_tsm = 3.0

phi_smc = 0.05     # 边界层厚度
phi_tsm = 0.05

def sat(x):
    return np.clip(x, -1.0, 1.0)

# ==============
# 初始化状态
# ==============
# SMC 系统
z_smc  = 0.0
zd_smc = 0.0

# TSM 系统
z_tsm  = 0.0
zd_tsm = 0.0

# 记录
t_list    = []
z_smc_list  = []
z_tsm_list  = []
u_smc_list  = []
u_tsm_list  = []
e_smc_list  = []
e_tsm_list  = []
d_list      = []

for i in range(N):
    t = i * dt

    # 参考信号
    z_ref  = z_ref_func(t)
    zd_ref = zd_ref_func(t)
    zdd_ref = zdd_ref_func(t)

    # 扰动
    d = dist_func(t)

    # ==========
    # 1) 普通 SMC
    # ==========
    e_smc  = z_smc  - z_ref
    ed_smc = zd_smc - zd_ref

    s_smc = ed_smc + lambda_smc * e_smc

    u_smc = -lambda_smc * (zd_smc - zd_ref) \
            - eta_smc * sat(s_smc / phi_smc) \
            + zdd_ref

    # 系统动力学：zdd = u + d
    zdd_smc = u_smc + d
    z_smc  += zd_smc * dt
    zd_smc += zdd_smc * dt

    # ==========
    # 2) TSM
    # ==========
    e_tsm  = z_tsm  - z_ref
    ed_tsm = zd_tsm - zd_ref

    # 避免 e=0 时的0^(-0.5)问题，加一个很小的eps
    eps = 1e-6
    abs_e_tsm = np.abs(e_tsm) + eps

    s_tsm = ed_tsm + lambda_tsm * (abs_e_tsm ** alpha_tsm) * np.sign(e_tsm)

    u_tsm = -lambda_tsm * alpha_tsm * (abs_e_tsm ** (alpha_tsm - 1.0)) * (zd_tsm - zd_ref) \
            - eta_tsm * sat(s_tsm / phi_tsm) \
            + zdd_ref

    zdd_tsm = u_tsm + d
    z_tsm  += zd_tsm * dt
    zd_tsm += zdd_tsm * dt

    # 记录数据
    t_list.append(t)
    z_smc_list.append(z_smc)
    z_tsm_list.append(z_tsm)
    u_smc_list.append(u_smc)
    u_tsm_list.append(u_tsm)
    e_smc_list.append(e_smc)
    e_tsm_list.append(e_tsm)
    d_list.append(d)


# ==============
# 画图
# ==============
t_array    = np.array(t_list)
z_smc_arr  = np.array(z_smc_list)
z_tsm_arr  = np.array(z_tsm_list)
u_smc_arr  = np.array(u_smc_list)
u_tsm_arr  = np.array(u_tsm_list)
e_smc_arr  = np.array(e_smc_list)
e_tsm_arr  = np.array(e_tsm_list)
d_arr      = np.array(d_list)

plt.figure(figsize=(10, 9))

# 1) 高度跟踪
plt.subplot(3, 1, 1)
plt.plot(t_array, z_smc_arr, label="z (SMC)")
plt.plot(t_array, z_tsm_arr, label="z (TSM)")
plt.plot(t_array, np.ones_like(t_array), 'k--', label="z_ref=1")
plt.ylabel("Altitude z [m]")
plt.legend()
plt.grid(True)

# 2) 误差
plt.subplot(3, 1, 2)
plt.plot(t_array, e_smc_arr, label="e (SMC)")
plt.plot(t_array, e_tsm_arr, label="e (TSM)")
plt.ylabel("Tracking error e [m]")
plt.legend()
plt.grid(True)

# 3) 控制输入 & 扰动
plt.subplot(3, 1, 3)
plt.plot(t_array, u_smc_arr, label="u (SMC)")
plt.plot(t_array, u_tsm_arr, label="u (TSM)")
plt.plot(t_array, d_arr, 'k--', label="disturbance d(t)")
plt.xlabel("Time [s]")
plt.ylabel("u, d")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
