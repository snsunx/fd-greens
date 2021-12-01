import numpy as np
import matplotlib.pyplot as plt

omegas, reTrS_red_sv, imTrS_red_sv = np.loadtxt('data/TrS_red_sv.dat').T
omegas, reTrS_red_qasm, imTrS_red_qasm = np.loadtxt('data/TrS_red_qasm.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, reTrS_red_sv, label="Real")
ax.plot(omegas, imTrS_red_sv, label="Imag")
ax.plot(omegas, reTrS_red_qasm, ls='--', label="Real")
ax.plot(omegas, imTrS_red_qasm, ls='--', label="Imag")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Tr$\Sigma$ (eV)")
#ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('TrS.png', dpi=300)
