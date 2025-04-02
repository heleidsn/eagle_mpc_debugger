import numpy as np
import matplotlib.pyplot as plt

# 参数设置
theta_real = 2.0  # 真实系统参数
gamma = 0.1       # 自适应增益
alpha = 0.5       # 误差反馈增益 (仅用于 NCE-AC)
k = 1.0           # 误差修正项 (仅用于 NCE-AC)
dt = 0.01         # 时间步长
T = 5.0           # 仿真时长

# 初始化变量
time = np.arange(0, T, dt)
x_ce = np.zeros_like(time)  # CE-AC 状态
x_nce = np.zeros_like(time) # NCE-AC 状态
theta_hat_ce = np.zeros_like(time)  # CE-AC 估计参数
theta_hat_nce = np.zeros_like(time) # NCE-AC 估计参数
x_ce[0] = 5.0   # 初始状态
x_nce[0] = 5.0  # 初始状态

# 仿真循环
for i in range(1, len(time)):
    # 误差信号 (用于 NCE-AC)
    e_ce = x_ce[i-1]
    e_nce = x_nce[i-1]

    # CE-AC 控制律
    u_ce = -theta_hat_ce[i-1] * x_ce[i-1]
    theta_hat_ce[i] = theta_hat_ce[i-1] + gamma * x_ce[i-1]**2 * dt  # 参数更新
    x_ce[i] = x_ce[i-1] + (theta_real * x_ce[i-1] + u_ce) * dt  # 系统更新

    # NCE-AC 控制律
    u_nce = -theta_hat_nce[i-1] * x_nce[i-1] + k * e_nce
    theta_hat_nce[i] = theta_hat_nce[i-1] + (gamma * x_nce[i-1]**2 + alpha * e_nce) * dt  # 参数更新
    x_nce[i] = x_nce[i-1] + (theta_real * x_nce[i-1] + u_nce) * dt  # 系统更新

# 画图
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time, x_ce, label="CE-AC", linestyle="--")
plt.plot(time, x_nce, label="NCE-AC")
plt.xlabel("Time (s)")
plt.ylabel("State x")
plt.title("State Response")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, theta_hat_ce, label="Estimated θ (CE-AC)", linestyle="--")
plt.plot(time, theta_hat_nce, label="Estimated θ (NCE-AC)")
plt.xlabel("Time (s)")
plt.ylabel("Estimated Parameter θ")
plt.title("Parameter Estimation")
plt.legend()

plt.tight_layout()
plt.show()
