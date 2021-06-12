"""Main test script to calculate absorption spectra A(omega) of LiH."""
import sys
sys.path.append('../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *
from constants import *
from math import pi
import numpy as np

np.set_printoptions(precision=3)

#ansatz = build_kosugi_lih_ansatz(2)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', 
    occupied_inds=[0], active_inds=[1, 2])
qiskit_operator = hamiltonian.to_qiskit_qubit_operator()
openfermion_operator = hamiltonian.to_openfermion_qubit_operator()
# print(qiskit_operator.primitive.coeffs)
# print(type(qiskit_operator))
# print(qiskit_operator)
#print('-'*80)
#q_instance = QuantumInstance(backend=Aer.get_backend('statevector_simulator'))
q_instance = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=8192)
#for tmp in [pi, 2 * pi, 3 * pi, 4 * pi]:
#for tmp in range(0, 200, 20):
for _ in [0]:
    ansatz = build_ne2_ansatz(4)
    hamiltonian = MolecularHamiltonian(
        [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', 
        occupied_inds=[0], active_inds=[1, 2])
    greens_function = GreensFunction(ansatz, hamiltonian, q_instance=q_instance)
    greens_function.compute_ground_state()
    greens_function.compute_eh_states()
    print("h state energies", greens_function.energies_h)
    # print("h state energies mod 2pi", greens_function.energies_h % (2 * pi))
    E1 = greens_function.energies_h[0]
    E2 = greens_function.energies_h[2]
    print("E_high - E_low =", E2 - E1)
    greens_function.compute_diagonal_amplitudes()
exit()
greens_function.compute_off_diagonal_amplitudes()
exit()
omegas = np.arange(-30, 34, 0.1)
As = []
for omega in omegas:
    A = greens_function.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    As.append(A)
np.savetxt('A_active.dat', np.vstack((omegas, As)).T)
