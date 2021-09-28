import numpy as np
import matplotlib.pyplot as plt

omegas, A_red_sv = np.loadtxt('A_red_sv.dat').T
omegas, A_red_qasm = np.loadtxt('A_red_qasm.dat').T
omegas, A_red_noisy = np.loadtxt('A_red_noisy.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_red_sv, label="Statevector")
ax.plot(omegas, A_red_qasm, ls='--', label="QASM")
ax.plot(omegas, A_red_noisy, ls='--', label="Noisy")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('A.png', dpi=300)
