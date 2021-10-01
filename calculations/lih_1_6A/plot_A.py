import numpy as np
import matplotlib.pyplot as plt

omegas, A_sv = np.loadtxt('A_sv.dat').T
omegas, A_qasm = np.loadtxt('A_qasm.dat').T
omegas, A_qasm1 = np.loadtxt('A_qasm1.dat').T
omegas, A_noisy = np.loadtxt('A_noisy.dat').T
omegas, A_noisy1 = np.loadtxt('A_noisy1.dat').T
omegas, A_noisy_new = np.loadtxt('A_noisy_new.dat').T
omegas, A_noisy_cached = np.loadtxt('A_noisy_cached.dat').T
omegas, A_noisy_cached1 = np.loadtxt('A_noisy_cached1.dat').T
omegas, A_noisy_cached2 = np.loadtxt('A_noisy_cached2.dat').T
omegas, A_red = np.loadtxt('A_red.dat').T
omegas, A_red_noisy = np.loadtxt('A_red_noisy.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_sv, label='4q, statevector')
#ax.plot(omegas, A_qasm, label='QASM')
#ax.plot(omegas, A_qasm1, label='QASM1')
#ax.plot(omegas, A_noisy, label='Noisy')
#ax.plot(omegas, A_noisy1, label='Noisy1')
#ax.plot(omegas, A_noisy_new, label='Noisy New')
#ax.plot(omegas, A_noisy_cached, label='Noisy Cached')
#ax.plot(omegas, A_noisy_cached1, label='Noisy Cached1')
#ax.plot(omegas, A_noisy_cached2, ls='--', label='Noisy Cached2')
ax.plot(omegas, A_red, ls='--', label="2q, statevector")
ax.plot(omegas, A_red_noisy, ls='--', label="2q, noisy")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-25, 15])
ax.legend()
fig.savefig('A.png', dpi=300)
