import numpy as np
import matplotlib.pyplot as plt

omegas, A_red_sv = np.loadtxt('data/A_red_sv.dat').T
#omegas, A_red_sve3 = np.loadtxt('../lih_3A/data/A_red_sve3.dat').T
omegas, A_red_qasm = np.loadtxt('data/A_red_qasm.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_red_sv, label="Exact")
ax.plot(omegas, A_red_qasm, label="QASM")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('A.png', dpi=300)
