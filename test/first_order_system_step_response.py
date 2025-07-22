import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step

# Define system parameters
K = 2       # Gain
tau = 3     # Time constant

# Define the transfer function G(s) = K / (tau*s + 1)
num = [K]
den = [tau, 1]
system = TransferFunction(num, den)

# Generate time vector and step response
t, y = step(system)

# Plot the response
plt.figure(figsize=(8, 4))
plt.plot(t, y, label='Step Response')
plt.axhline(K, color='r', linestyle='--', label='Final Value (K)')
plt.title('Step Response of First-Order System')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
