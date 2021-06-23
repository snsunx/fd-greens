import numpy as np

circ_recompiled = np.load('circ_recompiled.npy')
circ_recompiled1 = np.load('circ_recompiled1.npy')
circ_unrecompiled = np.load('circ_unrecompiled.npy')

circ_off_diag_statevector = np.load('circ_off_diag_statevector.npy')
circ_off_diag_recompiled = np.load('circ_off_diag_recompiled.npy')
circ_off_diag_unrecompiled = np.load('circ_off_diag_unrecompiled.npy')

psi1 = circ_off_diag_statevector
psi2 = circ_off_diag_unrecompiled
print(np.linalg.norm(psi1))
print(np.linalg.norm(psi2))
print(abs(psi1.conj().T @ psi2))

