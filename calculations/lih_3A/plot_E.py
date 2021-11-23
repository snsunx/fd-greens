import numpy as np
import matplotlib.pyplot as plt

#converters = {0: lambda s: complex(s.decode().replace('+-', '-'))}
radii, E1_red_qasm = np.loadtxt('data/E1_red_sv.dat').T#, dtype=complex, converters=converters).T
radii, E2_red_qasm = np.loadtxt('data/E2_red_sv.dat').T#, dtype=complex, converters=converters).T
Ecorr_red_qasm = E1_red_qasm + E2_red_qasm

#radii = radii[:-2]
#E1_red_qasm = E1_red_qasm[:-2]
#E2_red_qasm = E2_red_qasm[:-2]
#Ecorr_red_qasm = Ecorr_red_qasm[:-2]

fig, ax = plt.subplots()
ax.plot(radii, E1_red_qasm, marker='o', ls='--', label="E1")
ax.plot(radii, E2_red_qasm, marker='o', ls='--', label="E2")
ax.plot(radii, Ecorr_red_qasm, marker='o', ls='--', label='Ecorr')
ax.set_xlabel('$r$ (Angstrom)')
ax.set_ylabel('$E$ (eV)')
ax.legend()
fig.savefig('E_sv.png', dpi=300)
