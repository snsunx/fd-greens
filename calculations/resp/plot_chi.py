import numpy as np
import matplotlib.pyplot as plt

omegas, chi_real, chi_imag = np.loadtxt(f'data/lih_chi10.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, chi_real, label="Real")
ax.plot(omegas, chi_imag, label="Imag")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel(f"$\chi_1$ (eV$^{-1}$)")
ax.legend()
fig.savefig(f'figs/chi10.png', dpi=300)
