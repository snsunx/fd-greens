import numpy as np
import matplotlib.pyplot as plt

omegas_0d002, As_0d002 = np.loadtxt('A_0d002.dat').T
omegas_0d01, As_0d01 = np.loadtxt('A_0d01.dat').T
#omegas_0d05, As_0d05 = np.loadtxt('A_0d05.dat').T
omegas_0d1, As_0d1 = np.loadtxt('A_0d1.dat').T

fig, ax = plt.subplots()
ax.plot(omegas_0d002, As_0d002)
ax.set_ylim([0, 1.4])
#ax.plot(omegas_0d01, As_0d01)
#ax.plot(omegas_0d1, As_0d1)
fig.savefig('A.png')
