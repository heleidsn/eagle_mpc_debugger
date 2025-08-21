import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 模型参数
# ==============================
J = 0.01               # 转动惯量 [kg*m^2]
fric_pos = 0.2         # 正方向静摩擦 [Nm]
fric_neg = 0.25        # 负方向静摩擦 [Nm]
Ts = 0.001             # 采样周期 [s]
T_end = 4.0            # 仿真时长 [s]
noise_std = 0.01       # 测量噪声标准差

# 控制器与估计器参数
alpha = 0.2            # 跟踪更新权重
leak = 1e-3            # 保持时泄露
v_th_entry = 0.01      # rad/s
v_th_exit  = 0.02      # rad/s
u_th_entry = 0.01      # Nm
u_th_exit  = 0.02      # Nm
hold_min_time = 0.05   # s

# 目标轨迹： 0s~1s 正向运动，1s~2s 静止，2s~3s 反向运动，3s~4s 静止
def ref_vel(t):
    if t < 1.0:
        return 1.0
    elif t < 2.0:
        return 0.0
    elif t < 3.0:
        return -1.0
    else:
        return 0.0

# ==============================
# 初始化状态
# ==============================
n_steps = int(T_end / Ts)
theta = 0.0
omega = 0.0
d_hat = 0.0
d_hold = 0.0
in_hold = False
hold_timer = 0.0

# 记录变量
time_hist = []
omega_hist = []
d_hat_hist = []
d_hold_hist = []
u_cmd_hist = []
true_fric_hist = []
in_hold_hist = []

# ==============================
# 仿真循环
# ==============================
for k in range(n_steps):
    t = k * Ts
    v_ref = ref_vel(t)
    
    # 简单 PD 速度控制器（不考虑重力）
    Kp = 1.0
    Kv = 0.1
    tau_ctrl = Kp * (v_ref - omega) - Kv * omega  # 基础控制输出

    # 实际静摩擦力矩
    if abs(omega) < 1e-4:
        if tau_ctrl > 0:
            tau_fric = fric_pos
        elif tau_ctrl < 0:
            tau_fric = -fric_neg
        else:
            tau_fric = 0.0
    else:
        tau_fric = fric_pos * np.sign(omega) if omega > 0 else -fric_neg

    # 系统动力学更新
    domega = (tau_ctrl - tau_fric) / J
    omega += domega * Ts
    theta += omega * Ts

    # 加入速度测量噪声
    omega_meas = omega + np.random.randn() * noise_std

    # L1-like扰动估计（直接用残差近似）
    d_hat = (1 - alpha) * d_hat + alpha * (tau_ctrl - J * (omega_meas - omega) / Ts)

    # 进入保持区判定
    if (abs(omega_meas) < v_th_entry) and (abs(tau_ctrl) < u_th_entry):
        hold_timer += Ts
    else:
        hold_timer = 0.0

    if not in_hold and hold_timer >= hold_min_time:
        in_hold = True
    if in_hold and (abs(omega_meas) > v_th_exit or abs(tau_ctrl) > u_th_exit):
        in_hold = False

    # 更新 d_hold
    if not in_hold:
        d_hold = (1 - alpha) * d_hold + alpha * d_hat
    else:
        d_hold = (1 - leak) * d_hold

    # 记录
    time_hist.append(t)
    omega_hist.append(omega_meas)
    d_hat_hist.append(d_hat)
    d_hold_hist.append(d_hold)
    u_cmd_hist.append(tau_ctrl)
    true_fric_hist.append(tau_fric)
    in_hold_hist.append(int(in_hold))

# ==============================
# 绘图
# ==============================
plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.plot(time_hist, omega_hist, label='Measured velocity')
plt.axhline(v_th_entry, color='r', linestyle='--', alpha=0.5)
plt.axhline(-v_th_entry, color='r', linestyle='--', alpha=0.5)
plt.ylabel('Velocity [rad/s]')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(time_hist, true_fric_hist, label='True friction torque')
plt.plot(time_hist, d_hat_hist, label='d_hat (raw estimate)', alpha=0.6)
plt.plot(time_hist, d_hold_hist, label='d_hold (with hold)', linewidth=2)
plt.ylabel('Torque [Nm]')
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(time_hist, in_hold_hist, label='Hold state')
plt.ylabel('Hold flag')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
