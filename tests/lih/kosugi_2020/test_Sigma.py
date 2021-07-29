"""Main test script to calculate absorption spectra A(omega) of LiH."""
import sys
sys.path.append('../../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *
from constants import *

ansatz = build_kosugi_lih_ansatz(ind=1)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g')
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()

omegas = np.arange(-30, 34, 0.1)
Sigmas_real = []
Sigmas_imag = []

for omega in omegas:
    Sigma = greens_function.compute_self_energy(omega + 0.02j * HARTREE_TO_EV)
    Sigmas_real.append(np.trace(Sigma).real)
    Sigmas_imag.append(np.trace(Sigma).imag)
np.savetxt('Sigmas_UCC1.dat', np.vstack((omegas, Sigmas_real, Sigmas_imag)).T)
