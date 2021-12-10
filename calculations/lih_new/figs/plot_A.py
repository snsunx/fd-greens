import numpy as np
import matplotlib.pyplot as plt

omegas, A_red_sv = np.loadtxt('../data/A_red_sv.dat').T
#omegas, A_red_sve3 = np.loadtxt('../lih_3A/data/A_red_sve3.dat').T
omegas, lih_tomo_A = np.loadtxt('../data/lih_tomo_A.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, A_red_sv, label="SV")
ax.plot(omegas, lih_tomo_A, ls='--', marker='x', markevery=10, label="QASM")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('A.png', dpi=300)
