import sys
sys.path.append('../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *
from constants import *

ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian([['H', (0, 0, 0)], ['H', (0, 0, 0.6)]], 'sto3g')
print("n_electrons =", hamiltonian.molecule.n_electrons)
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
print("Ground state finished.")
greens_function.compute_eh_states()
print("eh states finished.")
greens_function.compute_diagonal_amplitudes()
print("Diagonal amplitudes finished.")
exit()
greens_function.compute_off_diagonal_amplitudes()
print("Off-diagonal amplitudes finished.")
omegas = np.arange(-30, 30, 0.1)
As = []
broadening = 0.05j
for omega in omegas:
    A = greens_function.compute_spectral_function(omega + broadening)
    As.append(A)
np.savetxt('A_H2.dat', np.vstack((omegas, As)).T)
