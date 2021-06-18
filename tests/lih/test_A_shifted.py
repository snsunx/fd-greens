"""A script to test phase estimation on the shifted Hamiltonian."""
import sys
sys.path.append('../../src/')
from greens_function import *
from ansatze import *
from hamiltonians import *
from constants import *
from math import pi
import numpy as np
from openfermion.linalg import get_sparse_operator 
from tools import get_quantum_instance

def define_hamiltonian():
    global hamiltonian
    hamiltonian = MolecularHamiltonian(
        [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', 
        occupied_inds=[0], active_inds=[1, 2])

ansatz = build_ne2_ansatz(4) 
define_hamiltonian()
scaling_factor, constant_shift = \
    GreensFunction.get_hamiltonian_shift_parameters(ansatz.copy(), hamiltonian)

# q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
#q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
q_instance = get_quantum_instance(Aer.get_backend('qasm_simulator'), shots=8192, noise_model_name='ibmq_jakarta')
for c in [0.]:
    greens_function = GreensFunction(
        ansatz.copy(), hamiltonian, scaling_factor=scaling_factor, 
        constant_shift=constant_shift + c, q_instance=q_instance)
    greens_function.compute_ground_state()
    greens_function.compute_eh_states()
    print("h state energies", greens_function.energies_h)
    greens_function.compute_diagonal_amplitudes()
