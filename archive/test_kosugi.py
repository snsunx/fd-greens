import sys
sys.path.append('../../src/')
from constants import *
from hamiltonians import *
from ansatze import *
import numpy as np
from scipy.sparse.linalg import eigsh

geometry = 'Li 0 0 0; H 0 0 1.6'
basis = 'sto3g'
hamiltonian = MolecularHamiltonian(geometry, basis)
qubit_op = hamiltonian.to_qiskit_qubit_operator()

ansatz = build_kosugi_lih_ansatz(1)
e, ansatz = run_vqe(ansatz.copy(), qubit_op)
print(e * HARTREE_TO_EV)

ansatz = build_kosugi_lih_ansatz(2)
e, ansatz = run_vqe(ansatz.copy(), qubit_op)
print(e * HARTREE_TO_EV)

hamiltonian_mat = hamiltonian.to_array('sparse')
energies_ne3, states_ne3 = number_state_eigensolver(hamiltonian_mat, 3)
energies_ne5, states_ne5 = number_state_eigensolver(hamiltonian_mat, 5)

