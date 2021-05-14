from greens_function import *
from ansatze import *
from hamiltonians import *

ansatz = build_ne2_ansatz(4)
#ansatz = build_kosugi_lih_ansatz(1)
#hamiltonian = MolecularHamiltonian('Li 0 0 0; H 0 0 1.6', 'sto3g')
hamiltonian = MolecularHamiltonian('Li 0 0 0; H 0 0 3', 'sto3g', occupied_inds=[0], active_inds=[1, 2])
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
#greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()
A = greens_function.compute_spectral_function(1 + 0.01j)
