import numpy as np
import matplotlib.pyplot as plt

omegas, chi00_real, chi00_imag = np.loadtxt('data/chi00_qasm.dat').T
omegas, chi11_real, chi11_imag = np.loadtxt('data/chi11_qasm.dat').T
omegas, sigma_real, sigma_imag = np.loadtxt('data/sigma_qasm.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, sigma_real, label="Re")
ax.plot(omegas, sigma_imag, label="Im")
ax.set_xlabel('$\omega$ (eV)')
#ax.set_ylabel("$\chi$ (eV$^{-1}$)")
ax.set_ylabel("$\sigma$")
ax.legend()
fig.savefig('sigma.png', dpi=300)
