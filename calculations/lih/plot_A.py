import numpy as np
import matplotlib.pyplot as plt

omegas, A_red_sv = np.loadtxt('data/A_red_sv.dat').T
omegas, lih_new_stat = np.loadtxt('data/lih_new_stat_A.dat').T
omegas, lih_new_qasm = np.loadtxt('data/lih_new_qasma_A.dat').T
#omegas, lih_new1_A = np.loadtxt('data/lih_new1_A.dat').T
#omegas, lih_eh_exact_A = np.loadtxt('data/lih_eh_exact_A.dat').T

fig, ax = plt.subplots()
#ax.plot(omegas, A_red_sv, label='Old, SV')
ax.plot(omegas, lih_new_stat - A_red_sv, ls='--', marker='x', markevery=5, label='stat')
ax.plot(omegas, lih_new_qasm - A_red_sv, ls='--', marker='x', markevery=5, label='qasm')
#ax.plot(omegas, lih_new1_A, ls='--', marker='x', markevery=5, label='New, QASM')
#ax.plot(omegas, lih_eh_tomo_A, ls='--', marker='x', markevery=10, label="QASM")
#ax.plot(omegas, lih_eh_exact_A, ls='--', marker='x', markevery=10, label="Exact")

ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('figs/A.png', dpi=300)
