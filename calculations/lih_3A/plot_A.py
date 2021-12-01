import numpy as np
import matplotlib.pyplot as plt

omegas, A_red_sv = np.loadtxt('data/A_red_sv.dat').T
omegas, A_red_sve3 = np.loadtxt('data/A_red_sve3.dat').T
omegas, A_red_qasm = np.loadtxt('data/A_red_qasm.dat').T
omegas, A_red_qasme3 = np.loadtxt('data/A_red_qasme3.dat').T
omegas, A_red_noisy = np.loadtxt('data/A_red_noisy.dat').T
omegas, A_red_noisye = np.loadtxt('data/A_red_noisye.dat').T
"""
omegas, A_red_sv_sv_qasm = np.loadtxt('data/A_red_sv-sv-qasm.dat').T
omegas, A_red_sv_qasm_qasm = np.loadtxt('data/A_red_sv-qasm-qasm.dat').T
omegas, A_red_sv_sv_noisy = np.loadtxt('data/A_red_sv-sv-noisy.dat').T
omegas, A_red_qasm_sv_qasm = np.loadtxt('data/A_red_qasm-sv-qasm.dat').T
omegas, A_red_qasm_qasm_qasm = np.loadtxt('data/A_red_qasm-qasm-qasm.dat').T
omegas, A_red_noisy_noisy_noisy = np.loadtxt('data/A_red_noisy-noisy-noisy.dat').T
"""

fig, ax = plt.subplots()
ax.plot(omegas, A_red_sv, label="Exact")
ax.plot(omegas, A_red_sve3, label="Exact, SVE3")
ax.plot(omegas, A_red_qasme3, label="QASM, E3")
#ax.plot(omegas, A_red_qasm, ls='--', label="QASM")
#ax.plot(omegas, A_red_noisy, ls='--', label="Noisy")
#ax.plot(omegas, A_red_sv_sv_qasm, ls='--', label="SV, SV, QASM")
#ax.plot(omegas, A_red_sv_qasm_qasm, ls='--', label="SV, QASM, QASM")
#ax.plot(omegas, A_red_sv_sv_noisy, ls='-.', label="SV, SV, Noisy")
#ax.plot(omegas, A_red_qasm_sv_qasm, ls='--', label="QASM, SV, QASM")
#ax.plot(omegas, A_red_qasm_qasm_qasm, ls='-.', label='QASM, QASM, QASM')
#ax.plot(omegas, A_red_noisy_noisy_noisy, ls='--', label="Noisy, Noisy, Noisy")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('A3.png', dpi=300)
