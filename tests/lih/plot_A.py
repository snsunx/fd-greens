import numpy as np
import matplotlib.pyplot as plt

omegas, A_qasm = np.loadtxt('A_qasm.dat').T
omegas, A_noisy = np.loadtxt('A_noisy.dat').T
omegas, A_noisy_new = np.loadtxt('A_noisy_new.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_qasm, label='QASM')
ax.plot(omegas, A_noisy, label='Noisy')
ax.plot(omegas, A_noisy_new, label='Noisy New')
ax.legend()
fig.savefig('A_compare.png')
