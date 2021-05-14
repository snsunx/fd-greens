import sys
sys.path.append('../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *

ansatz = build_kosugi_lih_ansatz(1)
#ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian('Li 0 0 0; H 0 0 1.6', 'sto3g')
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
eigenstates_e = greens_function.eigenstates_e
greens_function.compute_diagonal_amplitudes()
greens_function.compute_off_diagonal_amplitudes()
#B_e = greens_function.B_e
#B_h = greens_function.B_h
#print(np.amax(np.abs(B_e)))
#print(np.amax(np.abs(B_h)))
#print(np.sum(B_e))
#print(np.sum(B_h))
omegas = np.arange(-30, 30, 0.1)
As = []
for omega in omegas:
    print('### omega =', omega)
    A = greens_function.compute_spectral_function(omega + 0.5j)
    As.append(A)
np.savetxt('A_0d002.dat', np.vstack((omegas, As)).T)
