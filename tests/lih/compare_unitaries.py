import numpy as np

circ_recompiled = np.load('circ_recompiled.npy')
circ_recompiled1 = np.load('circ_recompiled1.npy')
circ_unrecompiled = np.load('circ_unrecompiled.npy')

print(abs(circ_recompiled1.conj().T @ circ_unrecompiled))
