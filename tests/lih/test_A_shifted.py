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

def define_hamiltonian():
    global hamiltonian
    hamiltonian = MolecularHamiltonian(
        [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', 
        occupied_inds=[0], active_inds=[1, 2])

"""
define_hamiltonian()
ansatz = build_ne2_ansatz(4)
greens_function = GreensFunction(ansatz.copy(), hamiltonian)
greens_function.compute_ground_state()
ansatz = greens_function.ansatz
greens_function.compute_eh_states()
print("h state energies", greens_function.energies_h)
E_low = greens_function.energies_h[0]
E_high = greens_function.energies_h[2]
print("E_high - E_low =", E_high - E_low)
scaling_factor = pi / (E_high - E_low)
print("scaling_factor =", scaling_factor)

greens_function = GreensFunction(
    ansatz.copy(), hamiltonian, scaling_factor=scaling_factor)
greens_function.compute_eh_states()
print("h state energies", greens_function.energies_h)
constant_shift = -greens_function.energies_h[0] 
print("constant_shift =", constant_shift)
"""

ansatz = build_ne2_ansatz(4) 
define_hamiltonian()
scaling_factor, constant_shift = \
    GreensFunction.get_hamiltonian_shift_parameters(ansatz.copy(), hamiltonian)

# q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
greens_function = GreensFunction(
    ansatz.copy(), hamiltonian, scaling_factor=scaling_factor, 
    constant_shift=constant_shift, q_instance=q_instance)
greens_function.compute_ground_state()
greens_function.compute_eh_states()
print("h state energies", greens_function.energies_h)
greens_function.compute_diagonal_amplitudes()
