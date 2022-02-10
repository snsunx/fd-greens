import numpy as np
import matplotlib.pyplot as plt

omegas, lih_expt_qasm, _ = np.loadtxt('data/lih_expt_qasm_TrSigma.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, lih_expt_qasm, ls='--', marker='x', markevery=20, label='QASM')

ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Tr$\Sigma$ (eV)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('figs/TrSigma.png', dpi=300)
