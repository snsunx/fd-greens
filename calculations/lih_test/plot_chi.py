import numpy as np
import matplotlib.pyplot as plt

ind = '11'
ystr = '\chi_{' + ind + '}'
fname = 'chi' + ind

omegas, chi_real, chi_imag = np.loadtxt(f'../data/chi{ind}_qasm.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, chi_real, label="Real")
ax.plot(omegas, chi_imag, label="Imag")
ax.set_xlabel('$\omega$ (eV)')
ax.set_ylabel(f"${ystr}$ (eV$^{-1}$)")
ax.legend()
fig.savefig(f'chi{ind}.png', dpi=300)
