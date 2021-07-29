import numpy as np
import matplotlib.pyplot as plt

omegas_UCC1, Sigmas_real_UCC1, Sigmas_imag_UCC1 = np.loadtxt('Sigmas_UCC1.dat').T
omegas_UCC2, Sigmas_real_UCC2, Sigmas_imag_UCC2 = np.loadtxt('Sigmas_UCC2.dat').T

fig, ax = plt.subplots()
ax.plot(omegas_UCC1, Sigmas_real_UCC1, label='Real UCC1')
ax.plot(omegas_UCC2, Sigmas_real_UCC2, label='Real UCC2')
#ax.plot(omegas_UCC2, Sigmas_imag_UCC2, label='Imag UCC2')
#ax.set_ylim([0, 1.4])
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel(r'$Tr \Sigma$ (eV$^{-1})$')
ax.legend()
fig.savefig('Sigma.png', dpi=250)
