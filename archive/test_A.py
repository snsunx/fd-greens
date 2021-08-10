"""Main test script to calculate absorption spectra A(omega) of LiH."""
import sys
sys.path.append('../../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *
from constants import *

ansatz = build_kosugi_lih_ansatz(2)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g')
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_Npm1_electron_states()
greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()

omega_arr = np.arange(-30, 34, 0.1)
A_list = []

for omega in omega_arr:
    A = greens_function.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
np.savetxt('A_UCC2_new.dat', np.vstack((omega_arr, A_list)).T)
