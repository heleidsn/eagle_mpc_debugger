import numpy as np
import matplotlib.pyplot as plt

# ========== 0. 系统与控制参数 ==========
m = 1.5
zeta = 0.7
omega_n = 4.0

Kp = m * omega_n**2
Kd = 2.0 * m * zeta * omega_n

print("Kp =", Kp, "Kd =", Kd)

# 参考信号：解析正弦
A = 1.0        # 振幅
f = 0.3        # Hz
omega_ref = 2.0 * np.pi * f

# 仿真设置
dt = 0.001
T = 10.0
N = int(T / dt)
t = np.linspace(0, T, N)

# 解析参考（作为“真”参考）
z_ref_ana      = A * np.sin(omega_ref * t)
z_ref_dot_ana  = A * omega_ref * np.cos(omega_ref * t)
z_ref_ddot_ana = -A * omega_ref**2 * np.sin(omega_ref * t)

# =====================================================================
# Case 1：解析正弦参考 + 只用 FF（理论上理想，数值上有漂移）
# =====================================================================
z_ff1     = 0.0
zdot_ff1  = 0.0
z_hist_ff1    = np.zeros(N)
err_hist_ff1  = np.zeros(N)

for k in range(N):
    zr     = z_ref_ana[k]
    zrddot = z_ref_ddot_ana[k]

    # 只用 FF：u = m * z_ref_ddot
    u = m * zrddot

    # 动力学 & Euler 积分
    zddot_ff1 = u / m
    zdot_ff1 += zddot_ff1 * dt
    z_ff1    += zdot_ff1 * dt

    z_hist_ff1[k]   = z_ff1
    err_hist_ff1[k] = zr - z_ff1

# =====================================================================
# Case 2：用“同样离散动力学”产生参考 + 只用 FF（几乎不漂）
# 思路：参考 z_ref_disc 用和被控对象完全一样的 Euler 积分产生
#      然后 FF 控制器对这个离散参考进行前馈，模型完全匹配 → 误差≈0
# =====================================================================
z_ref_disc    = 0.0   # 离散参考的 z
z_refdot_disc = 0.0
z_refdisc_hist = np.zeros(N)

z_ff2     = 0.0   # 被控对象
zdot_ff2  = 0.0
z_hist_ff2    = np.zeros(N)
err_hist_ff2  = np.zeros(N)

for k in range(N):
    tk = t[k]

    # 仍然用解析 sin 的加速度来驱动“参考系统”
    zrddot_disc = -A * omega_ref**2 * np.sin(omega_ref * tk)

    # 参考系统：Euler 积分
    z_refdot_disc += zrddot_disc * dt
    z_ref_disc    += z_refdot_disc * dt
    z_refdisc_hist[k] = z_ref_disc

    # 控制律：只用 FF，根据这个离散参考的加速度
    u2 = m * zrddot_disc

    zddot_ff2 = u2 / m
    zdot_ff2 += zddot_ff2 * dt
    z_ff2    += zdot_ff2 * dt

    z_hist_ff2[k]   = z_ff2
    err_hist_ff2[k] = z_ref_disc - z_ff2

# =====================================================================
# Case 3：解析正弦参考 + PD + FF（实际推荐做法）
# =====================================================================
z_pdff     = 0.0
zdot_pdff  = 0.0
z_hist_pdff    = np.zeros(N)
err_hist_pdff  = np.zeros(N)

for k in range(N):
    zr     = z_ref_ana[k]
    zrdot  = z_ref_dot_ana[k]
    zrddot = z_ref_ddot_ana[k]

    e    = zr - z_pdff
    edot = zrdot - zdot_pdff

    # PD + FF
    u3 = m * zrddot + Kp * e + Kd * edot

    zddot_pdff = u3 / m
    zdot_pdff += zddot_pdff * dt
    z_pdff    += zdot_pdff * dt

    z_hist_pdff[k]   = z_pdff
    err_hist_pdff[k] = zr - z_pdff

# =====================================================================
# 画图对比
# =====================================================================
plt.figure(figsize=(12, 10))

# ---- Case 1: 解析参考 + FF-only ----
plt.subplot(3, 2, 1)
plt.plot(t, z_ref_ana, label="z_ref (analytic)")
plt.plot(t, z_hist_ff1, "--", label="z (FF only, analytic ref)")
plt.title("Case 1: Analytic ref + FF-only (shows drift)")
plt.xlabel("t (s)")
plt.ylabel("z (m)")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t, err_hist_ff1)
plt.title("Case 1 Error (drift accumulates)")
plt.xlabel("t (s)")
plt.ylabel("e = z_ref - z (m)")
plt.grid(True)

# ---- Case 2: 离散参考 + FF-only ----
plt.subplot(3, 2, 3)
plt.plot(t, z_refdisc_hist, label="z_ref_disc (discrete model)")
plt.plot(t, z_hist_ff2, "--", label="z (FF only, discrete ref)")
plt.title("Case 2: Discrete ref (same dynamics) + FF-only")
plt.xlabel("t (s)")
plt.ylabel("z (m)")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t, err_hist_ff2)
plt.title("Case 2 Error (almost zero)")
plt.xlabel("t (s)")
plt.ylabel("e_disc = z_ref_disc - z (m)")
plt.grid(True)

# ---- Case 3: 解析参考 + PD+FF ----
plt.subplot(3, 2, 5)
plt.plot(t, z_ref_ana, label="z_ref (analytic)")
plt.plot(t, z_hist_pdff, "--", label="z (PD + FF)")
plt.title("Case 3: Analytic ref + PD + FF")
plt.xlabel("t (s)")
plt.ylabel("z (m)")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(t, err_hist_pdff)
plt.title("Case 3 Error (small, bounded)")
plt.xlabel("t (s)")
plt.ylabel("e = z_ref - z (m)")
plt.grid(True)

plt.tight_layout()
plt.show()
