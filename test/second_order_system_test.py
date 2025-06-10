import numpy as np
import matplotlib.pyplot as plt

# 模拟参数
dt = 0.01  # 时间步长
T = 10     # 总时间（秒）
m = 2.0    # 初始质量 kg
g = 9.81   # 重力加速度 m/s^2
max_thrust = 8 * g  # 最大推力 N

# 时间轴
t = np.arange(0, T, dt)

# 初始化状态变量
y = 0.0    # 初始位置
v = 0.0    # 初始速度
y_log = [] # 记录位置
v_log = [] # 记录速度
u_log = [] # 记录未滤波推力
u_filtered_log = []  # 记录实际输出推力
integral_log = []    # 记录积分器状态
m_log = [] # 记录质量变化

# PID参数（可调）
Kp = 200
Ki = 10
Kd = 50

# 一阶系统时间常数
tau = 0.05
u_filtered = 0.0  # 一阶系统的初始推力

# PID状态
integral = 0.0
prev_error = 0.0

# 参考轨迹1（阶跃信号）：0~5s 为0，5s后跳到2
def ref_step(t):
    return 0 if t < 2 else 2

# 参考轨迹2（正弦）：1Hz，幅值1，偏移1
def ref_sine(t):
    return 1 + np.sin(2 * np.pi * 0.5 * t)

# 更换轨迹函数（选择一个）
ref_func = ref_step
# ref_func = ref_sine

# 主仿真循环
for ti in t:
    # 在 t=5s 改变质量
    if ti >= 6.0:
        m = 4.0  # 改变质量

    ref = ref_func(ti)
    error = ref - y

    # PID控制器
    # 加入 anti-windup
    
    
    derivative = (error - prev_error) / dt
    u_unsat = Kp * error +  Kd * derivative
    
    if abs(u_unsat) < max_thrust:
        integral += error * dt  # 正常积分
    else:
        # 控制器饱和时冻结积分器，防止windup
        integral += 0
        
    u = u_unsat + Ki * integral
    
    prev_error = error

    # 加上重力补偿
    u += m * g

    # 饱和限制
    u = np.clip(u, -max_thrust, max_thrust)

    # 一阶系统滤波
    u_filtered += (dt / tau) * (u - u_filtered)

    # 更新系统状态
    a = (u_filtered - m * g) / m  # 受力公式 F=ma => a=(u-mg)/m
    v += a * dt
    y += v * dt

    # 日志记录
    y_log.append(y)
    v_log.append(v)
    u_log.append(u)
    u_filtered_log.append(u_filtered)
    integral_log.append(integral)
    m_log.append(m)

# 绘图
plt.figure(figsize=(10, 12))

# 位置图
plt.subplot(4, 1, 1)
plt.plot(t, [ref_func(ti) for ti in t], 'r--', label='Reference')
plt.plot(t, y_log, 'b', label='Output y')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

# 速度图
plt.subplot(4, 1, 2)
plt.plot(t, v_log, 'm', label='Velocity v')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()

# 控制输入图（积分器状态、未滤波推力、实际推力）
plt.subplot(4, 1, 3)
plt.plot(t, u_log, 'g', label='Raw Control input (Thrust)')
plt.plot(t, u_filtered_log, 'b', label='Filtered Control input (Thrust)')
plt.plot(t, integral_log, 'r', label='Integrator State')
plt.ylabel('Thrust / Integral')
plt.legend()
plt.grid()

# 质量变化图
plt.subplot(4, 1, 4)
plt.plot(t, m_log, 'c', label='Mass m')
plt.ylabel('Mass (kg)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
