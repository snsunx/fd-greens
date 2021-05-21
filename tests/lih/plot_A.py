import numpy as np
import matplotlib.pyplot as plt

omegas_active, As_active= np.loadtxt('A_active.dat').T
omegas_UCC1, As_UCC1 = np.loadtxt('A_UCC1.dat').T
#omegas_UCC2, As_UCC2 = np.loadtxt('A_UCC2.dat').T

fig, ax = plt.subplots()
ax.plot(omegas_active, As_active, label='Active')
ax.plot(omegas_UCC1, As_UCC1, label='UCC1')
#ax.plot(omegas_UCC2, As_UCC2, label='UCC2')
ax.set_ylim([0, 1.4])
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel(r'$A$ (eV$^{-1})$')
ax.legend()
fig.savefig('A.png', dpi=250)
