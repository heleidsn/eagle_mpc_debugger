import numpy as np
import matplotlib.pyplot as plt

# 定义 As 范围
As = np.linspace(-50, 50, 1000)

# 防止除以 0 的问题
As[As == 0] = np.nan

# 定义 Ts 值
Ts_list = [0.02, 0.01, 0.005]

# 绘制图像
plt.figure(figsize=(10, 6))

for Ts in Ts_list:
    Phi = (np.exp(As * Ts) - 1) / As
    adaptation_rate = np.exp(As * Ts) / Phi
    # plt.plot(As, Phi, label=f'Ts = {Ts}')
    plt.plot(As, adaptation_rate, linestyle='--', label=f'Adaptation Rate (Ts = {Ts})')

plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
# plt.title(r'$\Phi = A_s^{-1}(e^{A_s T_s} - 1)$')
plt.xlabel(r'$A_s$')
plt.ylabel(r'$\Phi$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
