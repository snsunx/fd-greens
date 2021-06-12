from greens_function import *
from ansatze import *
from hamiltonians import *

ansatz = build_two_local_ansatz(4)
hamiltonian = MolecularHamiltonian([['H', (0, 0, 0)], ['H', (0, 0, 1)]], 'sto3g')
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()
A = greens_function.compute_spectral_function(1 + 0.01j)
np.set_printoptions(precision=3)
print(np.abs(greens_function.G_e))
print(np.abs(greens_function.G_h))
print(np.abs(greens_function.G))
