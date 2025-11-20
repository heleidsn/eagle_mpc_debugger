import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------
# 1. 二阶弹簧阻尼系统参数
# -----------------------------
m = 1.0      # 质量
c = 0.5      # 阻尼
k = 2.0      # 刚度

# 连续时间状态空间
A = np.array([[0,       1],
              [-k/m, -c/m]])
B = np.array([[0],
              [1/m]])
C = np.array([[1, 0]])      # 只量测位移
dt = 0.01                   # 仿真步长
Ad = np.eye(2) + A * dt     # 简单Euler离散化
Bd = B * dt

T = 20.0
N = int(T / dt)
t_vec = np.arange(N) * dt

# -----------------------------
# 2. 真实系统仿真（含输入+扰动+测量噪声）
# -----------------------------
x_true = np.zeros((2, N))
x_true[:, 0] = [0.0, 0.0]

u_hist = np.zeros(N)
d_true = np.zeros(N)
y_meas = np.zeros(N)

np.random.seed(0)

for k in range(N - 1):
    t = k * dt
    # 控制输入：一个低频正弦
    u = 1.0 * math.sin(0.5 * t)
    u_hist[k] = u
    # 扰动：t>3 之后加入一个阶跃扰动
    d = 0.5 if t > 3.0 else 0.0
    d_true[k] = d

    xdot = A @ x_true[:, k] + B.flatten() * (u + d)
    x_true[:, k+1] = x_true[:, k] + dt * xdot

    # 测量：位置 + 高斯噪声
    y_meas[k] = C @ x_true[:, k] + np.random.randn() * 0.00

y_meas[-1] = C @ x_true[:, -1] + np.random.randn() * 0.00
d_true[-1] = d_true[-2]

# -----------------------------
# 3. Luenberger Observer
#    x_{k+1} = Ad x_k + Bd u_k + L (y_k - C x_k)
# -----------------------------
L_luen = np.array([[1.0],
                   [1.0]])      # 观测器增益（手动选一个稳定的）

xhat_L = np.zeros_like(x_true)
for k in range(N - 1):
    u = u_hist[k]
    y = y_meas[k]
    xhat_L[:, k+1] = (
        Ad @ xhat_L[:, k].reshape(-1, 1)
        + Bd * u
        + L_luen * (y - C @ xhat_L[:, k])
    ).flatten()

# -----------------------------
# 4. Kalman Filter
# -----------------------------
Qk = np.diag([1e-4, 1e-3])   # 过程噪声协方差（调参用）
Rk = np.array([[1e-2]])      # 测量噪声协方差

xhat_KF = np.zeros_like(x_true)
P = np.eye(2)

for k in range(N - 1):
    u = u_hist[k]
    y = y_meas[k]

    # 预测
    x_pred = Ad @ xhat_KF[:, k] + Bd.flatten() * u
    P_pred = Ad @ P @ Ad.T + Qk

    # 更新
    y_pred = C @ x_pred
    S = C @ P_pred @ C.T + Rk
    K_gain = P_pred @ C.T @ np.linalg.inv(S)
    xhat_KF[:, k+1] = x_pred + (K_gain * (y - y_pred)).flatten()
    P = (np.eye(2) - K_gain @ C) @ P_pred

# -----------------------------
# 5. Disturbance Observer (扩展状态观测: [x1,x2,d])
#    d 视为常值状态： d_{k+1} = d_k
# -----------------------------
A_e = np.array([[0,       1,    0],
                [-k/m, -c/m,  1/m],
                [0,       0,    0]])
B_e = np.array([[0],
                [1/m],
                [0]])
C_e = np.array([[1, 0, 0]])

Ad_e = np.eye(3) + A_e * dt
Bd_e = B_e * dt
L_e = np.array([[2.0],
                [3.0],
                [1.0]])     # 扩展观测器增益

xhat_DOB = np.zeros((3, N))   # [x1_hat, x2_hat, d_hat]

for k in range(N - 1):
    u = u_hist[k]
    y = y_meas[k]
    xhat_DOB[:, k+1] = (
        Ad_e @ xhat_DOB[:, k].reshape(-1, 1)
        + Bd_e * u
        + L_e * (y - C_e @ xhat_DOB[:, k])
    ).flatten()

dhat_DOB = xhat_DOB[2, :]

# -----------------------------
# 6. 简单 L1-like 自适应观测器
#    用自适应律估计常值扰动 d_hat
# -----------------------------
xhat_L1 = np.zeros_like(x_true)
dhat_L1 = np.zeros(N)
L_L1 = np.array([[2.0],
                 [3.0]])
gamma = 5.0         # 自适应增益

for k in range(N - 1):
    u = u_hist[k]
    y = y_meas[k]

    # 预测 + 输出注入 + 使用 d_hat 作为未知输入补偿
    xhat_L1[:, k+1] = (
        Ad @ xhat_L1[:, k].reshape(-1, 1)
        + Bd * (u + dhat_L1[k])
        + L_L1 * (y - C @ xhat_L1[:, k])
    ).flatten()

    # 非严格的 L1 风格：用输出误差来更新 d_hat
    dhat_L1[k+1] = dhat_L1[k] + gamma * (y - C @ xhat_L1[:, k]) * dt

# -----------------------------
# 7. 从Luenberger Observer估计扰动
#    通过状态估计误差反推扰动
# -----------------------------
# 方法1: 从状态估计误差推导扰动估计
# 真实系统: x_{k+1} = Ad x_k + Bd (u_k + d_k)
# 观测器:   xhat_{k+1} = Ad xhat_k + Bd u_k + L (y_k - C xhat_k)
# 扰动估计: dhat = (x_{k+1} - Ad x_k - Bd u_k) / Bd
# 但由于我们不知道真实状态，使用估计误差来近似

dhat_Luen = np.zeros(N)
for k in range(N - 1):
    # 使用观测器预测与实际测量的差异来估计扰动
    # 观测器预测: xhat_pred = Ad @ xhat_L[:, k] + Bd * u_hist[k]
    xhat_pred = Ad @ xhat_L[:, k] + Bd.flatten() * u_hist[k]
    # 实际更新: xhat_L[:, k+1] = xhat_pred + L_luen * (y_meas[k] - C @ xhat_L[:, k])
    # 扰动导致的额外变化可以通过观测器误差来估计
    # 简化方法：使用状态估计误差的变化率
    if k > 0:
        # 估计扰动：通过观测器增益和输出误差的关系
        # 这是一个近似方法，实际扰动估计需要扩展状态观测器
        output_error = y_meas[k] - C @ xhat_L[:, k]
        # 扰动对输出的影响可以通过观测器增益来估计
        # 这里使用一个简化的估计方法
        dhat_Luen[k] = (output_error * L_luen[0, 0]) / (Bd[1, 0] + 1e-6)  # 避免除零

# 更准确的方法：使用扩展状态观测器（DOB）的结果
# DOB已经实现了扩展的Luenberger Observer，可以直接估计扰动

# -----------------------------
# 8. 绘图对比
# -----------------------------
fig = plt.figure(figsize=(16, 10))

# (a) 位置估计对比
plt.subplot(3, 2, 1)
plt.plot(t_vec, x_true[0, :], 'k-', linewidth=2, label='True position')
plt.plot(t_vec, xhat_L[0, :], 'b--', linewidth=1.5, label='Luenberger Observer')
# plt.plot(t_vec, xhat_DOB[0, :], 'g--', linewidth=1.5, label='DOB (Extended Observer)')
plt.plot(t_vec, xhat_L1[0, :], 'r--', linewidth=1.5, label='L1-like Observer')
plt.ylabel('Position x [m]')
plt.title('Position Estimation Comparison')
plt.legend()
plt.grid(True)

# (b) 速度估计对比
plt.subplot(3, 2, 2)
plt.plot(t_vec, x_true[1, :], 'k-', linewidth=2, label='True velocity')
plt.plot(t_vec, xhat_L[1, :], 'b--', linewidth=1.5, label='Luenberger Observer')
# plt.plot(t_vec, xhat_DOB[1, :], 'g--', linewidth=1.5, label='DOB (Extended Observer)')
plt.plot(t_vec, xhat_L1[1, :], 'r--', linewidth=1.5, label='L1-like Observer')
plt.ylabel('Velocity v [m/s]')
plt.title('Velocity Estimation Comparison')
plt.legend()
plt.grid(True)

# (c) 扰动估计对比 - 主要图表
plt.subplot(3, 2, 3)
plt.plot(t_vec, d_true, 'k-', linewidth=2, label='True disturbance')
# plt.plot(t_vec, dhat_DOB, 'g--', linewidth=2, label='DOB estimate (Extended Luenberger)')
plt.plot(t_vec, dhat_L1, 'r--', linewidth=2, label='L1-like estimate')
plt.plot(t_vec, dhat_Luen, 'b:', linewidth=1.5, alpha=0.7, label='Luenberger derived (approx)')
plt.xlabel('Time [s]')
plt.ylabel('Disturbance d')
plt.title('Disturbance Estimation Comparison')
plt.legend()
plt.grid(True)

# (d) 扰动估计误差
plt.subplot(3, 2, 4)
d_error_DOB = dhat_DOB - d_true
d_error_L1 = dhat_L1 - d_true
d_error_Luen = dhat_Luen - d_true
# plt.plot(t_vec, d_error_DOB, 'g-', linewidth=1.5, label='DOB error')
plt.plot(t_vec, d_error_L1, 'r-', linewidth=1.5, label='L1-like error')
plt.plot(t_vec, d_error_Luen, 'b-', linewidth=1.5, alpha=0.7, label='Luenberger derived error')
plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Estimation Error')
plt.title('Disturbance Estimation Error')
plt.legend()
plt.grid(True)

# (e) 位置估计误差
plt.subplot(3, 2, 5)
pos_error_L = xhat_L[0, :] - x_true[0, :]
pos_error_DOB = xhat_DOB[0, :] - x_true[0, :]
pos_error_L1 = xhat_L1[0, :] - x_true[0, :]
plt.plot(t_vec, pos_error_L, 'b-', linewidth=1.5, label='Luenberger error')
# plt.plot(t_vec, pos_error_DOB, 'g-', linewidth=1.5, label='DOB error')
plt.plot(t_vec, pos_error_L1, 'r-', linewidth=1.5, label='L1-like error')
plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Position Error [m]')
plt.title('Position Estimation Error')
plt.legend()
plt.grid(True)

# (f) 控制输入和扰动
plt.subplot(3, 2, 6)
plt.plot(t_vec[:-1], u_hist[:-1], 'b-', linewidth=1.5, label='Control input u')
plt_twin = plt.gca().twinx()
plt_twin.plot(t_vec, d_true, 'r-', linewidth=2, label='True disturbance')
# plt_twin.plot(t_vec, dhat_DOB, 'g--', linewidth=1.5, label='DOB estimate')
plt.xlabel('Time [s]')
plt.ylabel('Control Input u', color='b')
plt_twin.set_ylabel('Disturbance d', color='r')
plt.title('Control Input and Disturbance')
plt.legend(loc='upper left')
plt_twin.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印统计信息
print("\n" + "=" * 60)
print("Disturbance Estimation Statistics")
print("=" * 60)
print(f"DOB (Extended Luenberger Observer):")
print(f"  RMSE: {np.sqrt(np.mean(d_error_DOB**2)):.4f}")
print(f"  Max error: {np.max(np.abs(d_error_DOB)):.4f}")
print(f"  Mean error: {np.mean(d_error_DOB):.4f}")
print(f"\nL1-like Observer:")
print(f"  RMSE: {np.sqrt(np.mean(d_error_L1**2)):.4f}")
print(f"  Max error: {np.max(np.abs(d_error_L1)):.4f}")
print(f"  Mean error: {np.mean(d_error_L1):.4f}")
print(f"\nLuenberger derived (approximate):")
print(f"  RMSE: {np.sqrt(np.mean(d_error_Luen**2)):.4f}")
print(f"  Max error: {np.max(np.abs(d_error_Luen)):.4f}")
print(f"  Mean error: {np.mean(d_error_Luen):.4f}")
print("=" * 60)
