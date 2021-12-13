import numpy as np
import matplotlib.pyplot as plt

omegas, reTrS_red_sv, imTrS_red_sv = np.loadtxt('../data/TrS_red_sv.dat').T
#omegas, reTrS_red_qasm, imTrS_red_qasm = np.loadtxt('../data/TrS_red_qasm.dat').T
omegas, re_lih_tomo_TrS, im_lih_tomo_TrS = np.loadtxt('../data/lih_eh_exact_TrSigma.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, reTrS_red_sv, label="Real, SV")
#ax.plot(omegas, imTrS_red_sv, label="Imag, SV")
ax.plot(omegas, re_lih_tomo_TrS, ls='--', marker='x', markevery=20, label="Real, QASM")
#ax.plot(omegas, im_lih_tomo_TrS, ls='--', marker='x', markevery=20, label="Imag, QASM")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("Tr$\Sigma$ (eV)")
#ax.set_xlim([-20, 10])
ax.legend()
fig.savefig('TrSigma.png', dpi=300)
