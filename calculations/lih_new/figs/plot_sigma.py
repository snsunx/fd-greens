import numpy as np
import matplotlib.pyplot as plt

omegas, sigma_real, sigma_imag = np.loadtxt('../data/lih_exc_sigma.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, sigma_real)
#ax.plot(omegas, sigma_imag, label="Imag")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel("$\sigma$ (a.u.)")
#ax.legend()
fig.savefig('sigma.png', dpi=300)
