import numpy as np
import matplotlib.pyplot as plt

omegas, As = np.loadtxt('A_H2.dat').T
omegas_0d05, As_0d05 = np.loadtxt('A_0d05.dat').T
omegas_0d1, As_0d1 = np.loadtxt('A_0d1.dat').T

fig, ax = plt.subplots()
ax.plot(omegas, As)
ax.plot(omegas_0d05, As_0d05)
ax.plot(omegas_0d1, As_0d1)
fig.savefig('A_H2.png')
