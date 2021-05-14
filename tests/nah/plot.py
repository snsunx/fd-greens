import numpy as np
import matplotlib.pyplot as plt

rs, es_ne1_min = np.loadtxt('es_ne1_min.dat').T
rs, es_ne1_max = np.loadtxt('es_ne1_max.dat').T
rs, es_ne3_min = np.loadtxt('es_ne3_min.dat').T
rs, es_ne3_max = np.loadtxt('es_ne3_max.dat').T
rs, es_gs = np.loadtxt('es_gs.dat').T

fig, ax = plt.subplots()
ax.plot(rs, es_ne1_min, color='xkcd:orange', marker='o', label='$N_e = 1$, min')
ax.plot(rs, es_ne1_max, color='xkcd:light brown', marker='+', label='$N_e = 1$, max')
ax.plot(rs, es_ne3_min, color='xkcd:gold', marker='^', label='$N_e = 3$, min')
ax.plot(rs, es_ne3_max, color='xkcd:black', marker='D', label='$N_e = 3$, max')
ax.plot(rs, es_gs, color='C0', marker='x', label='$N_e = 2$, ground state')
ax.set_xlabel('$r (\AA)$')
ax.set_ylabel('$E$ (Ha)')
ax.legend()
fig.savefig('nah_vqe.png')
