import numpy as np
import matplotlib.pyplot as plt

omegas, lih_expt_qasm = np.loadtxt('data/lih_3A_1_A.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, lih_expt_qasm, ls='--', marker='x', markevery=5, label='QASM')

ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
#ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('figs/A.png', dpi=300)
