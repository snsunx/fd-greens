import numpy as np
import matplotlib.pyplot as plt

omegas, A_red_sv = np.loadtxt('../data/A_red_sv.dat').T
#omegas, A_red_sv_sv_sv = np.loadtxt('../data/A_red_sv-sv-sv.dat').T
omegas, A_red_qasm_sv_qasm = np.loadtxt('../data/A_red_qasm-sv-qasm.dat').T
omegas, A_red_noisy_sv_noisy = np.loadtxt('../data/A_red_noisy-sv-noisy.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_red_sv, label="Statevector")
#ax.plot(omegas, A_red_sv_sv_sv, label="SV, SV, SV")
ax.plot(omegas, A_red_qasm_sv_qasm, ls='--', label="QASM, SV, QASM")
ax.plot(omegas, A_red_noisy_sv_noisy, ls='--', label="Noisy, SV, Noisy")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('A.png', dpi=300)
