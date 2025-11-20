# Sinusoidal tracking demo: frequency sweep + time-domain with delay/saturation/mismatch
# Requirements: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import pi

wn_nom = 8.0
zeta_nom = 0.7
Kp = wn_nom**2   # 64
Kd = 2*zeta_nom*wn_nom  # 11.2

Kp = 10
Kd = 0.1

def P_freq(w, wn, zeta, delay=0.0):
    s = 1j*w
    G = (wn**2) / (s**2 + 2*zeta*wn*s + wn**2)
    if delay > 0:
        G = G * np.exp(-s*delay)
    return G

def closed_loop_T(w, Kp, Kd, wn_p, zeta_p, F=None, delay=0.0):
    s = 1j*w
    C = Kd*s + Kp
    P = P_freq(w, wn_p, zeta_p, delay=delay)
    Fw = 0.0 if F is None else F(w)
    return (P*C + P*Fw) / (1 + P*C)

def F_inverse_nominal(w):
    s = 1j*w
    Pn = (wn_nom**2) / (s**2 + 2*zeta_nom*wn_nom*s + wn_nom**2)
    return 1.0 / (Pn + 1e-12)

# --- 1) Frequency sweep ---
w = np.logspace(-1, 2, 600)

T_A = closed_loop_T(w, Kp, Kd, wn_nom, zeta_nom, F=None, delay=0.0)
T_B = closed_loop_T(w, Kp, Kd, wn_nom, zeta_nom, F=F_inverse_nominal, delay=0.0)
T_C = closed_loop_T(w, Kp, Kd, wn_nom, zeta_nom, F=F_inverse_nominal, delay=0.03)
T_D = closed_loop_T(w, Kp, Kd, 1.2*wn_nom, zeta_nom, F=F_inverse_nominal, delay=0.0)

E_A = np.abs(1 - T_A)
E_B = np.abs(1 - T_B)
E_C = np.abs(1 - T_C)
E_D = np.abs(1 - T_D)

plt.figure()
plt.semilogx(w, 20*np.log10(E_A), label="PD only")
plt.semilogx(w, 20*np.log10(E_B), label="PD + inverse FF (ideal)")
plt.semilogx(w, 20*np.log10(E_C), label="PD + inverse FF + 30 ms delay")
plt.semilogx(w, 20*np.log10(E_D), label="PD + inverse FF + 20% mismatch")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("|E(jw)| (dB)")
plt.title("Steady-State Error Magnitude vs Frequency")
plt.grid(True)
plt.legend()
plt.show()

# --- 2) Time-domain with delay/saturation/mismatch ---
Tsim = 6.0
dt = 0.0005
t = np.arange(0, Tsim, dt)
wn_p = 1.2*wn_nom
zeta_p = 0.65
delay = 0.00
u_limit = 100

ref_freq_hz = 2.0   # change to Hz
wref = 2 * pi * ref_freq_hz
r   = np.sin(wref * t)
rd  = wref * np.cos(wref * t)
rdd = - (wref**2) * np.sin(wref * t) 
delay_steps = int(np.round(delay / dt))

def plant_step(x, u_applied):
    y, yd = x
    ydd = -2*zeta_p*wn_p*yd - (wn_p**2)*y + (wn_p**2)*u_applied
    yd  = yd + dt * ydd
    y   = y + dt * yd
    return np.array([y, yd]), y, yd, ydd

def saturate(val, limit):
    return np.clip(val, -limit, limit)

def inverse_ff_full(r, rd, rdd):
    return (rdd + 2*zeta_nom*wn_nom*rd + (wn_nom**2)*r) / (wn_nom**2)

def simulate_controller(mode):
    x = np.array([0.0, 0.0])
    buf = [0.0]*(delay_steps+1)
    y_hist = np.zeros_like(t)
    e_hist = np.zeros_like(t)
    u_hist = np.zeros_like(t)
    u_ff_hist = np.zeros_like(t)
    u_fb_hist = np.zeros_like(t)
    d_hat = 0.0
    dob_alpha = 2*pi*6.0*dt  # ~6 Hz

    for k in range(len(t)):
        y, yd = x
        e  = r[k]  - y
        ed = rd[k] - yd
        u_fb = Kp*e + Kd*ed

        if mode == "PD":
            u_ff = 0.0
        elif mode == "PD_INVFF":
            u_ff = inverse_ff_full(r[k], rd[k], rdd[k])
        elif mode == "PD_INVFF_DOB":
            u_ff = inverse_ff_full(r[k], rd[k], rdd[k])
        elif mode == "MPC_PREVIEW":
            kp = min(k + delay_steps, len(t)-1)
            u_ff = inverse_ff_full(r[kp], rd[kp], rdd[kp])
        else:
            raise ValueError

        # u = u_ff + u_fb
        u = u_ff + u_fb

        if mode == "PD_INVFF_DOB":
            ydd_nom = -2*zeta_nom*wn_nom*yd - (wn_nom**2)*y + (wn_nom**2)*u
            ydd_true = -2*zeta_p*wn_p*yd - (wn_p**2)*y + (wn_p**2)*u
            d_tilde = ydd_true - ydd_nom
            d_hat = d_hat + dob_alpha * (d_tilde - d_hat)
            u = u - d_hat/(wn_nom**2)

        u = saturate(u, u_limit)

        u_applied = buf.pop(0)
        buf.append(u)
        x, y, yd, ydd_true = plant_step(x, u_applied)
        y_hist[k] = y
        e_hist[k] = r[k] - y
        u_hist[k] = u_applied
        u_ff_hist[k] = u_ff
        u_fb_hist[k] = u_fb
    return y_hist, e_hist, u_hist, u_ff_hist, u_fb_hist

# modes = ["PD", "PD_INVFF", "PD_INVFF_DOB", "MPC_PREVIEW"]
modes = ["PD", "PD_INVFF_DOB", "MPC_PREVIEW"]
results = {m: simulate_controller(m) for m in modes}

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
ax1, ax2, ax3, ax4 = axes

# 第一个子图：参考信号和跟踪结果
ax1.plot(t, r, label="Reference")
for m in modes:
    ax1.plot(t, results[m][0], label=m.replace("_"," "))
ax1.set_ylabel("Amplitude")
ax1.set_title("Time-Domain Tracking (Delay/Saturation/Mismatch)")
ax1.grid(True)
ax1.legend()

# 第二个子图：跟踪误差
for m in modes:
    ax2.plot(t, results[m][1], label=m.replace("_"," "))
ax2.set_ylabel("Tracking Error")
ax2.set_title("Tracking Error Comparison")
ax2.grid(True)
ax2.legend()

# 第三个子图：前馈控制量 u_ff
for m in modes:
    ax3.plot(t, results[m][3], label=f"{m.replace('_',' ')} - u_ff")
ax3.set_ylabel("u_ff")
ax3.set_title("Feedforward Control Signal")
ax3.grid(True)
ax3.legend()

# 第四个子图：反馈控制量 u_fb
for m in modes:
    ax4.plot(t, results[m][4], label=f"{m.replace('_',' ')} - u_fb")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("u_fb")
ax4.set_title("Feedback Control Signal")
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()
