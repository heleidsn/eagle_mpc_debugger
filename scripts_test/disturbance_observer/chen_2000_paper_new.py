import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ========== 机械臂与摩擦参数（可以根据需要改） ==========

g = 9.81

# 2-link planar arm parameters (随便选一组合理的)
m1, m2 = 0.5, 0.3      # kg
l1, l2 = 1.0, 0.8      # m
lc1, lc2 = 0.5, 0.4    # COM distances
I1, I2 = 0.002, 0.001      # link inertias

# Friction parameters from Chen paper (Coulomb + viscous)
# link 1
z1 = 0.0541     # N*m
k1 = 0.0076     # N*m/(rad/s)
# link 2
z2 = 0.0167
k2 = 0.0088

# Revised friction model中那个小参数 l（这里直接用0.001，和文中相近）
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
    # 经典书上常用形式
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
    文中式 (33)(34) 的简化实现：
    当速度接近0时，摩擦等于外加力矩 tau（静摩擦区），否则趋近于 d(qd)
    这里只实现一个近似版本，主要是避免 sign 在0附近的数值问题
    """
    d = coulomb_viscous_friction(qd)

    # 近似的 Ta: 在 [-z, z] 内取 tau，否则饱和为 ±z
    # 为简单起见，对于两个关节分别处理
    Ta = np.zeros(2)
    for i, (zi, tau_i) in enumerate(zip([z1, z2], tau)):
        if tau_i > zi:
            Ta[i] = zi
        elif tau_i < -zi:
            Ta[i] = -zi
        else:
            Ta[i] = tau_i

    # 速度越接近0，越趋向 Ta；速度大时趋向 d
    # 这里给一个简单权重：exp(-(qd/l)^2)
    weight = np.exp(- (qd / l_fric)**2)
    return d + (Ta - d) * weight


# ========== 参考轨迹：两个关节分别给方波 ==========

def reference(t):
    """
    返回期望 qd, q 和 qdd (这里qdd_des设为0，简单一点)
    模仿论文：两个关节分别给方波命令
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
    # q_des = np.array([1.0, 2.0])
    q_des = np.array([0.0, 0.0])
    qd_des = np.zeros(2)
    qdd_des = np.zeros(2)
    return q_des, qd_des, qdd_des


# ========== NDO 设计：Chen 文中的形式 ==========

# 设计参数 L = c * I
# 根据Chen 2000论文，增益应该足够大以保证快速收敛和跟踪阶跃扰动
# 对于阶跃扰动，需要较大的增益来快速响应
c = 100.0  # 增大增益以提高对阶跃扰动的跟踪速度
L = c * np.eye(2)

def p_func(q, qd):
    """
    p(q, qd) = L * M(q) * qd
    这是满足 ∂p/∂qd = L*M(q) 的一个简单选择，
    对应文中设计，能把 LJ*ddq 吃掉
    """
    return L @ (M(q) @ qd)

def ndo_dynamics(z, q, qd, tau, Cq=None):
    """
    NDO动态:  ż = -L z + L( C(q,q̇)q̇ + G(q) - τ - p(q, q̇) )
    根据Chen 2000论文的标准形式
    这里 T = tau
    """
    G = G_vec(q)
    p = p_func(q, qd)
    
    # 计算科氏/离心项 C(q,q̇)q̇
    if Cq is not None:
        Cqd = Cq @ qd
    else:
        Cq_matrix = C(q, qd)
        Cqd = Cq_matrix @ qd
    
    # 标准NDO形式：ż = -L·z + L·(C(q,q̇)q̇ + G(q) - τ - p(q,q̇))
    dz = -L @ z + L @ (Cqd + G - tau - p)
    
    # 当前的 d_hat
    d_hat = z + p
    return dz, d_hat


# ========== 周期阶跃扰动函数 ==========

def periodic_step_disturbance(t, period=2.0, amplitude=0.5):
    """
    生成周期变化的阶跃扰动
    period: 周期（秒）
    amplitude: 扰动幅值（N*m）
    返回: [d1, d2] 两个关节的扰动
    """
    # 计算当前周期内的相位
    phase = (t % period) / period
    
    # 关节1: 前半个周期为0，后半个周期为amplitude
    if phase < 0.5:
        d1 = 0.0
    else:
        d1 = amplitude
    
    # 关节2: 前半个周期为-amplitude，后半个周期为0
    if phase < 0.5:
        d2 = 0.0
    else:
        d2 = amplitude
        
    d1 = 0.0
    
    return np.array([d1, d2])


def L_func(q):
    Mq = M(q)
    return c * np.linalg.inv(Mq)

def ndo_dynamics_full(z, p, q, qd, qdd, tau):
    # N(q, qd) = C(q,qd) qd + G(q)
    Cq = C(q, qd)
    Gq = G_vec(q)
    Nq = Cq @ qd + Gq
    Mq = M(q)
    Lq = L_func(q)

    # ż = -L z + L (N - τ - p)
    dz = -Lq @ z + Lq @ (Nq - tau - p)

    # ṗ = L M q¨
    dp = Lq @ (Mq @ qdd)

    # τ̂_d = z + p
    tau_d_hat = z + p
    return dz, dp, tau_d_hat



# ========== 主仿真循环 ==========

T_end = 10.0
dt = 0.001
N = int(T_end / dt)

# 状态变量
q = np.array([0.0, 0.0])
qd = np.array([0.0, 0.0])

# NDO 状态
z = np.zeros(2)
p = np.zeros(2)
d_hat = np.zeros(2)  # 初始扰动估计为0

tau_d_hat = np.zeros(2)   # 对应论文中的 \hat τ_d

# 控制器增益（计算力矩 + PD）
Kp = np.diag([100.0, 100.0])
Kd = np.diag([10.0, 10.0])

# 记录数据
time_hist = []
q_hist = []
qd_hist = []
q_des_hist = []  # 参考位置
qd_des_hist = []  # 参考速度
tau_hist = []
tau_uncompensated_hist = []  # 未补偿的控制力矩
d_true_hist = []
d_hat_hist = []
d_ext_hist = []  # 外部扰动历史

for k in range(N):
    t = k * dt

    # 参考信号
    q_des, qd_des, qdd_des = reference(t)

    # 计算动力学项
    Mq = M(q)
    Cq = C(q, qd)
    Gq = G_vec(q)

    # 计算力矩控制律（使用上一时刻的扰动估计进行补偿）
    v = qdd_des + Kd @ (qd_des - qd) + Kp @ (q_des - q)
    # 添加扰动补偿：tau = M*v + C*qd + G + d_hat
    tau = Mq @ v + Cq @ qd + Gq + d_hat
    
    # 计算未补偿的控制输入（用于NDO，根据Chen 2000论文）
    # NDO应该基于未补偿的控制输入来估计扰动
    tau_uncompensated = Mq @ v + Cq @ qd + Gq

    # 真实摩擦（采用 revised friction 近似模型）
    d_friction = revised_friction(qd, tau)
    
    # 周期阶跃扰动
    d_ext = periodic_step_disturbance(t, period=4.0, amplitude=0.5)
    
    # 总扰动 = 摩擦 + 外部扰动
    # d_true = d_friction + d_ext
    d_true = d_ext
    # d_true = np.zeros(2)

    # 机械臂动力学：Mq * qdd = tau - C qd - G - d_true
    qdd = np.linalg.solve(Mq, tau - Cq @ qd - Gq - d_true)

    # NDO 更新（使用未补偿的控制输入，根据Chen 2000论文）
    # 观测器基于：d = J(θ)θ̈ + G(θ, θ̇) - T
    # 如果T中已经包含d_hat，观测器无法正确估计
    dz, d_hat = ndo_dynamics(z, q, qd, tau_uncompensated, Cq=Cq)
    
    # ===== Chen 2000 基本 NDO =====
    Nq = Cq @ qd + Gq          # N(q, qd)
    tau_d_hat_dot = -L @ tau_d_hat + L @ (Mq @ qdd + Nq - tau)
    tau_d_hat += tau_d_hat_dot * dt
    
    # 注意：这里 tau_d_hat ≈ τ_d = -d_true
    d_hat = -tau_d_hat         # 和你定义的 d_true 对齐

    # 欧拉积分
    qd = qd + qdd * dt
    q = q + qd * dt
    z = z + dz * dt

    # 记录数据
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
plt.plot(time_hist, tau_hist[:, 0], label='Control Torque τ1 (with d_hat)', linewidth=2)
plt.plot(time_hist, tau_uncompensated_hist[:, 0], '--', label='Uncompensated τ1', linewidth=1.5, alpha=0.7)
plt.plot(time_hist, d_hat_hist[:, 0], ':', label='d_hat contribution', linewidth=1.5, alpha=0.7)
plt.ylabel('Torque [Nm]')
plt.title('Joint 1: Control Torque (d_hat compensation)')
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
plt.plot(time_hist, tau_hist[:, 1], label='Control Torque τ2 (with d_hat)', linewidth=2)
plt.plot(time_hist, tau_uncompensated_hist[:, 1], '--', label='Uncompensated τ2', linewidth=1.5, alpha=0.7)
# plt.plot(time_hist, d_hat_hist[:, 1], ':', label='d_hat contribution', linewidth=1.5, alpha=0.7)
plt.ylabel('Torque [Nm]')
plt.title('Joint 2: Control Torque (d_hat compensation)')
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
