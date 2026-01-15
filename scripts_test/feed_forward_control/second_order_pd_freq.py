import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 系统 & PD 参数 =====
m = 1.5        # 质量 (kg)
zeta = 0.7     # 希望的闭环阻尼比
omega_n = 4.0  # 希望的闭环自然频率 (rad/s)

# 根据二阶系统公式计算 PD 增益
Kp = m * omega_n**2
Kd = 2.0 * m * zeta * omega_n

print("Kp =", Kp, "Kd =", Kd)

# ===== 2. 仿真设置 =====
dt = 0.001           # 仿真步长 (s)
A_ref = 1.0          # 参考正弦的振幅 (m)
freqs_hz = np.linspace(0.05, 2.0, 40)  # 扫描参考频率区间 [0.05, 2] Hz
error_amp_list = []  # 存储每个频率下的稳态误差幅值


def simulate_error_amp(f_hz):
    """
    对于给定参考频率 f_hz，仿真纯 PD 控制下的 tracking，
    返回稳态误差的幅值估计（假定误差近似为正弦）。
    """
    omega_ref = 2.0 * np.pi * f_hz

    # 仿真时长：至少 10 个周期，且不短于 10 秒
    T_period = 1.0 / f_hz
    T_sim = max(10.0 * T_period, 10.0)
    N = int(T_sim / dt)
    t = np.linspace(0, T_sim, N)

    # 状态初始化
    z = 0.0
    zdot = 0.0

    e_hist = np.zeros(N)

    for k in range(N):
        tk = t[k]

        # 参考及其导数（纯 PD 不用加速度，只用 z_ref_dot）
        z_ref = A_ref * np.sin(omega_ref * tk)
        z_ref_dot = A_ref * omega_ref * np.cos(omega_ref * tk)

        # 误差
        e = z_ref - z
        edot = z_ref_dot - zdot

        # 纯 PD 控制（无前馈）
        u = Kp * e + Kd * edot

        # 动力学：m * zddot = u
        zddot = u / m

        # 欧拉积分
        zdot += zddot * dt
        z += zdot * dt

        e_hist[k] = e

    # 丢掉前一半的过渡过程，只看后半段稳态
    steady = e_hist[N // 2 :]

    # 对稳态误差做 RMS，再换算成等效正弦幅值：A = sqrt(2) * RMS
    rms = np.sqrt(np.mean(steady**2))
    amp_est = np.sqrt(2.0) * rms
    return amp_est


# ===== 3. 对每个频率做仿真，记录误差幅值 =====
for f in freqs_hz:
    amp = simulate_error_amp(f)
    error_amp_list.append(amp)
    print(f"f = {f:.2f} Hz, error_amp ≈ {amp:.4f} m")

error_amp_list = np.array(error_amp_list)

# ===== 4. 画图：误差幅值 vs 参考频率 =====
plt.figure(figsize=(8, 5))
plt.plot(freqs_hz, error_amp_list, marker="o")
plt.xlabel("Reference frequency $f$ (Hz)")
plt.ylabel("Steady-state tracking error amplitude |e| (m)")
plt.title("PD tracking error vs reference frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
