import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 机械臂与摩擦参数 ==========

g = 9.81

# 2-link planar arm parameters
m1, m2 = 2.0, 1.0      # kg
l1, l2 = 1.0, 0.8      # m
lc1, lc2 = 0.5, 0.4    # COM distances
I1, I2 = 0.2, 0.1      # link inertias

# Friction parameters (Coulomb + viscous)
z1 = 0.0541     # N*m
k1 = 0.0076     # N*m/(rad/s)
z2 = 0.0167
k2 = 0.0088

# Revised friction model small parameter
l_fric = 0.001


# ========== 动力学模型：M(q), C(q, qd), G(q) ==========

def M(q):
    """惯性矩阵 M(q), q = [q1, q2]"""
    q1, q2 = q
    c2 = np.cos(q2)

    a11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    a12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    a22 = I2 + m2 * lc2**2
    return np.array([[a11, a12],
                     [a12, a22]])

def C(q, qd):
    """科氏/离心矩阵 C(q, qd)"""
    q1, q2 = q
    qd1, qd2 = qd
    s2 = np.sin(q2)

    h = m2 * l1 * lc2 * s2
    C11 = h * qd2
    C12 = h * (qd1 + qd2)
    C21 = -h * qd1
    C22 = 0.0

    return np.array([[C11, C12],
                     [C21, C22]])

def G_vec(q):
    """重力项 G(q)"""
    q1, q2 = q
    g1 = (m1 * lc1 + m2 * l1) * g * np.cos(q1) + m2 * lc2 * g * np.cos(q1 + q2)
    g2 = m2 * lc2 * g * np.cos(q1 + q2)
    return np.array([g1, g2])


# ========== 摩擦模型：Coulomb + viscous + revised model ==========

def coulomb_viscous_friction(qd):
    """传统 Coulomb + viscous 摩擦模型 d(qd)"""
    qd1, qd2 = qd
    d1 = z1 * np.sign(qd1) + k1 * qd1
    d2 = z2 * np.sign(qd2) + k2 * qd2
    return np.array([d1, d2])

def revised_friction(qd, tau):
    """
    简化版 revised friction:
    速度接近0时，摩擦接近外力矩 tau（静摩擦区），否则趋近于 d(qd)
    """
    d = coulomb_viscous_friction(qd)

    Ta = np.zeros(2)
    for i, (zi, tau_i) in enumerate(zip([z1, z2], tau)):
        if tau_i > zi:
            Ta[i] = zi
        elif tau_i < -zi:
            Ta[i] = -zi
        else:
            Ta[i] = tau_i

    weight = np.exp(- (qd / l_fric)**2)
    return d + (Ta - d) * weight


# ========== 参考轨迹：两个关节阶跃 ==========

def reference(t):
    """
    返回期望 q, qd, qdd
    两个关节给简单方波，用来测试扰动估计和跟踪
    """
    # 关节1: 0~2s 0rad, 2~6s 0.5rad, 6~10s -0.5rad
    if t < 2.0:
        q1_des = 0.0
    elif t < 6.0:
        q1_des = 0.5
    else:
        q1_des = -0.5

    # 关节2: 0~3s 0, 3~7s -0.5, 7~10s 0.5
    if t < 3.0:
        q2_des = 0.0
    elif t < 7.0:
        q2_des = -0.5
    else:
        q2_des = 0.5

    q_des = np.array([q1_des, q2_des])
    qd_des = np.zeros(2)
    qdd_des = np.zeros(2)
    return q_des, qd_des, qdd_des


# ========== Chen 型 NDO / DOB（无加速度版本） ==========

# 设计参数 L = c * I
c = 10.0
L = c * np.eye(2)

def p_func(q, qd):
    """
    这里选用：p(q, qd) = L * M(q) * qd
    这是常见的一种“无加速度 NDO”实现方式：
    - 用到 q, qd, M(q)，但不用 qdd
    - 对应 Chen / Mohammadi 里的变换
    """
    return L @ (M(q) @ qd)

def ndo_update(z, q, qd, tau):
    """
    无加速度 NDO 动态（估计的是 +d，而系统实际是 -d_true）：
    基于：M(q)qdd + C(q,qd)qd + G(q) = tau + d   （理论形式）
    对应的 NDO（无加速度实现）：
        p = L M(q) qd
        ż = -L z + L ( C(q,qd)qd + G(q) - tau - p )
        d_hat (理论里的 d̂) = z + p

    我们的仿真中：
        M qdd + C qd + G + d_true = tau
    所以 d = -d_true
    => 对 d 的估计为 d_hat，则对 d_true 的估计为 d_true_hat = -d_hat
    """
    Cq = C(q, qd)
    Gq = G_vec(q)
    Cqd = Cq @ qd
    p = p_func(q, qd)

    dz = -L @ z + L @ (Cqd + Gq - tau - p)
    d_hat_d = z + p       # 这是对 "d = +扰动" 的估计

    # 转成我们动力学里用的 d_true 对应的估计
    d_hat_true = -d_hat_d
    return dz, d_hat_true, d_hat_d


# ========== 周期阶跃扰动函数 ==========

def periodic_step_disturbance(t, period=2.0, amplitude=0.5):
    """
    生成周期变化的阶跃扰动
    period: 周期（秒）
    amplitude: 扰动幅值（N*m）
    返回: [d1, d2] 两个关节的扰动
    """
    phase = (t % period) / period

    # 关节1: 前半周期 0，后半周期 +amplitude
    if phase < 0.5:
        d1 = 0.0
    else:
        d1 = amplitude

    # 关节2: 前半周期 0，后半周期 +amplitude
    if phase < 0.5:
        d2 = 0.0
    else:
        d2 = amplitude

    return np.array([d1, d2])


# ========== 主仿真循环 ==========

T_end = 10.0
dt = 0.001
N = int(T_end / dt)

# 状态变量
q = np.array([0.0, 0.0])
qd = np.array([0.0, 0.0])

# NDO 状态
z = np.zeros(2)
d_hat = np.zeros(2)          # 对 d_true 的估计
d_hat_d = np.zeros(2)        # 对 “理论 d”（加在右边）的估计

# 控制器增益（位置 PD）
Kp = np.diag([100.0, 100.0])
Kd = np.diag([10.0, 10.0])

# 记录数据
time_hist = []
q_hist = []
qd_hist = []
q_des_hist = []
qd_des_hist = []
tau_hist = []
tau_uncompensated_hist = []
d_true_hist = []
d_hat_hist = []
d_ext_hist = []

for k in range(N):
    t = k * dt

    # 参考信号
    q_des, qd_des, qdd_des = reference(t)

    # 动力学矩阵
    Mq = M(q)
    Cq = C(q, qd)
    Gq = G_vec(q)

    # PD + 前馈控制（这里没有加扰动补偿）
    v = qdd_des + Kd @ (qd_des - qd) + Kp @ (q_des - q)
    tau_uncompensated = Mq @ v + Cq @ qd + Gq
    tau = tau_uncompensated.copy()   # 如果要补偿，可以在这里 tau = tau_uncompensated + d_hat（注意符号）

    # 真实摩擦（可选，看你要不要算进扰动）
    d_friction = revised_friction(qd, tau)

    # 周期阶跃扰动（这里用作主要测试对象）
    d_ext = periodic_step_disturbance(t, period=2.0, amplitude=0.5)

    # 总扰动，这里我们只让 NDO 追 d_ext，方便看效果
    d_true = d_ext
    # 如果希望连摩擦一起估计，可以改成：
    # d_true = d_friction + d_ext

    # 机械臂动力学：Mq * qdd = tau - C qd - G - d_true
    qdd = np.linalg.solve(Mq, tau - Cq @ qd - Gq - d_true)

    # ===== 无加速度 NDO 更新 =====
    dz, d_hat, d_hat_d = ndo_update(z, q, qd, tau_uncompensated)
    z = z + dz * dt

    # 状态积分
    qd = qd + qdd * dt
    q = q + qd * dt

    # 记录
    time_hist.append(t)
    q_hist.append(q.copy())
    qd_hist.append(qd.copy())
    q_des_hist.append(q_des.copy())
    qd_des_hist.append(qd_des.copy())
    tau_hist.append(tau.copy())
    tau_uncompensated_hist.append(tau_uncompensated.copy())
    d_true_hist.append(d_true.copy())
    d_hat_hist.append(d_hat.copy())
    d_ext_hist.append(d_ext.copy())

# 转成数组便于画图
time_hist = np.array(time_hist)
q_hist = np.array(q_hist)
qd_hist = np.array(qd_hist)
q_des_hist = np.array(q_des_hist)
qd_des_hist = np.array(qd_des_hist)
tau_hist = np.array(tau_hist)
tau_uncompensated_hist = np.array(tau_uncompensated_hist)
d_true_hist = np.array(d_true_hist)
d_hat_hist = np.array(d_hat_hist)
d_ext_hist = np.array(d_ext_hist)

# ========== 画图：状态、力矩和扰动 ==========

plt.figure(figsize=(14, 12))

# ========== 左列：Joint 1 ==========
# Joint 1 Position
plt.subplot(4, 2, 1)
plt.plot(time_hist, q_hist[:, 0], label='Actual q1', linewidth=2)
plt.plot(time_hist, q_des_hist[:, 0], '--', label='Reference q1', linewidth=2)
plt.ylabel('Position [rad]')
plt.title('Joint 1: Position')
plt.legend()
plt.grid(True)

# Joint 1 Velocity
plt.subplot(4, 2, 3)
plt.plot(time_hist, qd_hist[:, 0], label='Actual qd1', linewidth=2)
plt.plot(time_hist, qd_des_hist[:, 0], '--', label='Reference qd1', linewidth=2)
plt.ylabel('Velocity [rad/s]')
plt.title('Joint 1: Velocity')
plt.legend()
plt.grid(True)

# Joint 1 Torque
plt.subplot(4, 2, 5)
plt.plot(time_hist, tau_hist[:, 0], label='Control Torque τ1', linewidth=2)
plt.plot(time_hist, tau_uncompensated_hist[:, 0], '--', label='Uncompensated τ1', linewidth=1.5, alpha=0.7)
plt.ylabel('Torque [Nm]')
plt.title('Joint 1: Control Torque')
plt.legend()
plt.grid(True)

# Joint 1 Disturbance
plt.subplot(4, 2, 7)
plt.plot(time_hist, d_true_hist[:, 0], label='True Disturbance', linewidth=2)
plt.plot(time_hist, d_ext_hist[:, 0], ':', label='External Step Disturbance', linewidth=1.5, alpha=0.7)
plt.plot(time_hist, d_hat_hist[:, 0], '--', label='Estimated Disturbance', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('Joint 1: Disturbance Estimation')
plt.legend()
plt.grid(True)

# ========== 右列：Joint 2 ==========
# Joint 2 Position
plt.subplot(4, 2, 2)
plt.plot(time_hist, q_hist[:, 1], label='Actual q2', linewidth=2)
plt.plot(time_hist, q_des_hist[:, 1], '--', label='Reference q2', linewidth=2)
plt.ylabel('Position [rad]')
plt.title('Joint 2: Position')
plt.legend()
plt.grid(True)

# Joint 2 Velocity
plt.subplot(4, 2, 4)
plt.plot(time_hist, qd_hist[:, 1], label='Actual qd2', linewidth=2)
plt.plot(time_hist, qd_des_hist[:, 1], '--', label='Reference qd2', linewidth=2)
plt.ylabel('Velocity [rad/s]')
plt.title('Joint 2: Velocity')
plt.legend()
plt.grid(True)

# Joint 2 Torque
plt.subplot(4, 2, 6)
plt.plot(time_hist, tau_hist[:, 1], label='Control Torque τ2', linewidth=2)
plt.plot(time_hist, tau_uncompensated_hist[:, 1], '--', label='Uncompensated τ2', linewidth=1.5, alpha=0.7)
plt.ylabel('Torque [Nm]')
plt.title('Joint 2: Control Torque')
plt.legend()
plt.grid(True)

# Joint 2 Disturbance
plt.subplot(4, 2, 8)
plt.plot(time_hist, d_true_hist[:, 1], label='True Disturbance', linewidth=2)
plt.plot(time_hist, d_ext_hist[:, 1], ':', label='External Step Disturbance', linewidth=1.5, alpha=0.7)
plt.plot(time_hist, d_hat_hist[:, 1], '--', label='Estimated Disturbance', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('Joint 2: Disturbance Estimation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
