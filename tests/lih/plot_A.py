import numpy as np
import matplotlib.pyplot as plt

omegas, A_qasm = np.loadtxt('A_qasm.dat').T
omegas, A_noisy = np.loadtxt('A_noisy.dat').T
omegas, A_noisy1 = np.loadtxt('A_noisy1.dat').T
omegas, A_noisy_new = np.loadtxt('A_noisy_new.dat').T
omegas, A_noisy_cached = np.loadtxt('A_noisy_cached.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_qasm, label='QASM')
ax.plot(omegas, A_noisy, label='Noisy')
ax.plot(omegas, A_noisy1, label='Noisy1')
ax.plot(omegas, A_noisy_new, label='Noisy New')
ax.plot(omegas, A_noisy_cached, label='Noisy Cached')
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.legend()
fig.savefig('A.png')
