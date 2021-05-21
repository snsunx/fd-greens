"""Main test script to calculate absorption spectra A(omega) of LiH."""
import sys
sys.path.append('../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *
from constants import *

#ansatz = build_kosugi_lih_ansatz(2)
ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian('Li 0 0 0; H 0 0 1.6', 'sto3g', occupied_inds=[0], active_inds=[1, 2])
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()
exit()
omegas = np.arange(-30, 34, 0.1)
As = []
for omega in omegas:
    A = greens_function.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    As.append(A)
np.savetxt('A_active.dat', np.vstack((omegas, As)).T)
