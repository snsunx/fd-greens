import sys
sys.path.append('../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *

ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian('H 0 0 0; H 0 0 1', 'sto3g')
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()
omegas = np.arange(-3, 3, 0.01)
As = []
for omega in omegas:
    A = greens_function.compute_spectral_function(omega + 0.1j)
    As.append(A)
np.savetxt('A_0d1.dat', np.vstack((omegas, As)).T)
