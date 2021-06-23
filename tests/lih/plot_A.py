import numpy as np
import matplotlib.pyplot as plt

omegas_active, As_active= np.loadtxt('A_active.dat').T
omegas_UCC1, As_UCC1 = np.loadtxt('A_UCC1.dat').T
omegas_UCC2, As_UCC2 = np.loadtxt('A_UCC2.dat').T
omegas_qasm, As_qasm = np.loadtxt('A_qasm.dat').T
omegas_recompiled, As_recompiled = np.loadtxt('A_recompiled.dat').T
omegas_noisy, As_noisy = np.loadtxt('A_noisy.dat').T

fig, ax = plt.subplots()
ax.plot(omegas_active, As_active, label='Statevector')
#ax.plot(omegas_qasm, As_qasm, label='QASM', marker='.')
#ax.plot(omegas_UCC1, As_UCC1, label='UCC1')
#ax.plot(omegas_UCC2, As_UCC2, label='UCC2')
#ax.plot(omegas_recompiled, As_recompiled, label='Recompiled', marker='^')
ax.plot(omegas_noisy, As_noisy, label='Noisy')
ax.set_ylim([0, 1.4])
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel(r'$A$ (eV$^{-1})$')
ax.legend()
fig.savefig('A.png', dpi=250)
