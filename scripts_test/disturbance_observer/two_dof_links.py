import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# 2-link 平面机械臂参数（简化版）
# -----------------------------
m1 = 1.0   # link1 mass
m2 = 1.0   # link2 mass
l1 = 1.0   # link1 length
l2 = 1.0   # link2 length
lc1 = 0.5  # link1 COM
lc2 = 0.5  # link2 COM
I1 = 0.2   # link1 inertia
I2 = 0.2   # link2 inertia
g  = 9.81  # gravity

def manipulator_dynamics(t, x, tau_fun):
    """
    x = [q1, q2, dq1, dq2]
    tau_fun(t) 返回 2x1 torque
    """
    q1, q2, dq1, dq2 = x
    tau = tau_fun(t)  # [tau1, tau2]

    # -------- 惯量矩阵 M(q)（含 cos(q2) → 非线性）--------
    a1 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2)
    a2 = m2*l1*lc2
    a3 = I2 + m2*lc2**2

    M11 = a1 + 2*a2*cos(q2)
    M12 = a3 + a2*cos(q2)
    M21 = M12
    M22 = a3

    M = np.array([[M11, M12],
                  [M21, M22]])

    # -------- 科里奥利/离心项 C(q, dq)*dq（含 dq1*dq2, dq1^2, sin(q2) → 非线性）--------
    h = -a2*sin(q2)  # 方便写法
    C1 = h * dq2 * (2*dq1 + dq2)   # 对应 Spong 里的 C(q,dq)*dq 第一项
    C2 = h * dq1**2                # 第二项

    C_vec = np.array([C1, C2])

    # -------- 重力项 g(q)（含 cos(q1), cos(q1+q2) → 非线性）--------
    g1 = (m1*lc1 + m2*l1)*g*cos(q1) + m2*lc2*g*cos(q1 + q2)
    g2 = m2*lc2*g*cos(q1 + q2)
    G_vec = np.array([g1, g2])

    # 计算关节加速度
    ddq = np.linalg.solve(M, tau - C_vec - G_vec)

    # 状态导数
    dx = np.array([dq1, dq2, ddq[0], ddq[1]])
    return dx

# -----------------------------
# 输入力矩：给一个简单的常值或正弦力矩
# -----------------------------
def tau_const(t):
    # 常值力矩，故意选一个不为 0 的
    return np.array([1.0, 0.5])

# 你也可以换成正弦激励：
# def tau_sin(t):
#     return np.array([1.0 * np.sin(1.0 * t),
#                      0.5 * np.sin(0.5 * t)])

# -----------------------------
# 仿真函数
# -----------------------------
def simulate(x0, tau_fun, t_span=(0.0, 5.0), dt=0.001):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        fun=lambda t, x: manipulator_dynamics(t, x, tau_fun),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )
    return sol.t, sol.y  # t, x(t)

# -----------------------------
# 设置三个初始条件：
# x0c = x0a + x0b，用来测试线性叠加
# -----------------------------
x0a = np.array([0.1, 0.0, 0.0, 0.0])   # 轻微初始角度
x0b = np.array([0.0, 0.2, 0.0, 0.0])   # 另一关节轻微初始角度
x0c = x0a + x0b                        # 线性叠加

# 仿真
t, xa = simulate(x0a, tau_const)
_, xb = simulate(x0b, tau_const)
_, xc = simulate(x0c, tau_const)

# xa, xb, xc 形状均为 (4, N)
# 如果系统是线性的，那么应该有：
# xc ≈ xa + xb （对所有时间 t）
x_linear_comb = xa + xb  # 假如系统是线性时的“预测”

# -----------------------------
# 画图比较 q1, q2：
# xc（真实 x0c） vs xa+xb（线性叠加预测）
# -----------------------------
plt.figure()
plt.title("q1: true x(t; x0c) vs xa+xb (if linear)")
plt.plot(t, xc[0, :], label="q1 true (x0c)")
plt.plot(t, x_linear_comb[0, :], '--', label="q1 xa+xb (linear superposition)")
plt.xlabel("time [s]")
plt.ylabel("q1 [rad]")
plt.legend()
plt.grid(True)

plt.figure()
plt.title("q2: true x(t; x0c) vs xa+xb (if linear)")
plt.plot(t, xc[1, :], label="q2 true (x0c)")
plt.plot(t, x_linear_comb[1, :], '--', label="q2 xa+xb (linear superposition)")
plt.xlabel("time [s]")
plt.ylabel("q2 [rad]")
plt.legend()
plt.grid(True)

plt.show()
