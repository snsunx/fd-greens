import numpy as np
import matplotlib.pyplot as plt

omegas, red_sv, _ = np.loadtxt('data/TrS_red_sv.dat').T
omegas, lih_new_stat, _ = np.loadtxt('data/lih_new_stat_TrSigma.dat').T
omegas, lih_new_qasm, _ = np.loadtxt('data/lih_new_qasm_TrSigma.dat').T
#omegas, reTrS_lih_new1, imTrS_lih_new1 = np.loadtxt('data/lih_new1_TrSigma.dat').T

#omegas, reTrS_red_qasm, imTrS_red_qasm = np.loadtxt('../data/TrS_red_qasm.dat').T
#omegas, re_lih_tomo_TrS, im_lih_tomo_TrS = np.loadtxt('data/lih_eh_exact_TrSigma.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, red_sv, label="Old, SV")
ax.plot(omegas, lih_new_stat, ls='--', marker='x', markevery=20, label='stat')
ax.plot(omegas, lih_new_qasm, ls='--', marker='x', markevery=20, label='qasm')
#ax.plot(omegas, reTrS_lih_new1, ls='--', marker='x', markevery=20, label="New, QASM")
#ax.plot(omegas, imTrS_red_sv, label="Imag, SV")
#ax.plot(omegas, re_lih_tomo_TrS, ls='--', marker='x', markevery=20, label="Real, QASM")
#ax.plot(omegas, im_lih_tomo_TrS, ls='--', marker='x', markevery=20, label="Imag, QASM")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Tr$\Sigma$ (eV)")
#ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('figs/TrSigma.png', dpi=300)
