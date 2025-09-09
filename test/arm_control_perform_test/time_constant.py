import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fc = np.linspace(0.1, 50, 500)     # 截止频率 1Hz ~ 50Hz
dt = 0.005                       # 采样周期 (s)，例如 200 Hz 采样

# 1) 计算时间常数 tau
tau = 1 / (2 * np.pi * fc)

# 2) 计算对应的滤波系数 alpha (精确离散化)
alpha = 1 - np.exp(-dt / tau)

# 画图
plt.figure(figsize=(10,4))

# 子图1: fc vs tau
plt.subplot(1,2,1)
plt.plot(fc, tau, 'b')
plt.xlabel("Cutoff Frequency $f_c$ (Hz)")
plt.ylabel("Time Constant $\\tau$ (s)")
plt.title("Cutoff Frequency vs Time Constant")
plt.grid(True)

# 子图2: fc vs alpha
plt.subplot(1,2,2)
plt.plot(fc, alpha, 'r')
plt.xlabel("Cutoff Frequency $f_c$ (Hz)")
plt.ylabel("Filter Coefficient $\\alpha$")
plt.title(f"Cutoff Frequency vs Alpha (dt={dt}s)")
plt.grid(True)

plt.tight_layout()
plt.show()
