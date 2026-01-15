import numpy as np
import matplotlib.pyplot as plt

# ======================
# 滑模控制参数
# ======================
c1 = 2.0        # 滑模面系数
k  = 5.0        # 切换增益（越大到达越快，抖振越大）
phi = 0.1       # 边界层厚度（越大抖振越小，但等效精度变差）

x_ref = 1.0     # 期望位置

# 是否使用边界层：True → sat，False → sign
use_boundary_layer = True

def sat(x):
    """饱和函数，|x|<=1 时为线性，之外为 ±1"""
    return np.clip(x, -1.0, 1.0)

def sliding_mode_control(x1, x2):
    # 误差
    e  = x1 - x_ref
    # 滑模面
    s  = c1 * e + x2

    if use_boundary_layer:
        u_eq = -k * sat(s / phi)  # 带边界层，减小抖振
    else:
        u_eq = -k * np.sign(s)    # 纯 sign，抖振更明显

    return u_eq, s

# ======================
# 仿真设置
# ======================
dt   = 0.001
T    = 5.0
N    = int(T / dt)

# 状态初始化
x1 = 0.0   # 位置
x2 = 0.0   # 速度

t_list  = []
x1_list = []
x2_list = []
u_list  = []
s_list  = []
d_list  = []

for i in range(N):
    t = i * dt

    # 外扰（可以随便换，这里用一个有界正弦扰动）
    d = 0.5 * np.sin(2.0 * t)

    # 滑模控制器
    u, s = sliding_mode_control(x1, x2)

    # 系统动力学
    # x1' = x2
    # x2' = u + d
    x1_dot = x2
    x2_dot = u + d

    # 欧拉积分
    x1 += x1_dot * dt
    x2 += x2_dot * dt

    # 记录数据
    t_list.append(t)
    x1_list.append(x1)
    x2_list.append(x2)
    u_list.append(u)
    s_list.append(s)
    d_list.append(d)

# ======================
# 画图
# ======================

t_array  = np.array(t_list)
x1_array = np.array(x1_list)
x2_array = np.array(x2_list)
u_array  = np.array(u_list)
s_array  = np.array(s_list)
d_array  = np.array(d_list)

plt.figure(figsize=(10, 8))

# 1) 位置 & 速度 & 参考
plt.subplot(3, 1, 1)
plt.plot(t_array, x1_array, label="x1 (position)")
plt.plot(t_array, x2_array, label="x2 (velocity)")
plt.plot(t_array, x_ref * np.ones_like(t_array), '--', label="x_ref")
plt.ylabel("x1, x2")
plt.legend()
plt.grid(True)

# 2) 滑模面
plt.subplot(3, 1, 2)
plt.plot(t_array, s_array, label="sliding surface s")
plt.ylabel("s")
plt.legend()
plt.grid(True)

# 3) 控制输入 & 扰动
plt.subplot(3, 1, 3)
plt.plot(t_array, u_array, label="u (control)")
plt.plot(t_array, d_array, '--', label="d (disturbance)")
plt.xlabel("time [s]")
plt.ylabel("u, d")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
