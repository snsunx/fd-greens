from z2_symmetries import apply_cnot_z2, taper
import numpy as np
from qiskit import *
from ansatze import build_ne2_ansatz
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from greens_function import GreensFunction
from tools import get_quantum_instance
from constants import HARTREE_TO_EV
from qiskit.opflow import PauliSumOp, PrimitiveOp, Z2Symmetries
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Pauli, PauliTable
from tools import number_state_eigensolver

ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', 
    occupied_inds=[0], active_inds=[1, 2])
qiskit_op = hamiltonian.qiskit_op

# Convert IZIZ and ZIZI to IIIZ and IIZI
qiskit_op_new = apply_cnot_z2(apply_cnot_z2(qiskit_op, 2, 0), 3, 1)
print(qiskit_op_new[:5])

# Taper off the first two qubits (last two qubits in normal order)
qiskit_op_tapered_00 = taper(qiskit_op_new, [0, 1], init_state=[0, 0])
#qiskit_op_tapered_01 = taper(qiskit_op_new, [2, 3], init_state=[0, 1])
#qiskit_op_tapered_10 = taper(qiskit_op_new, [2, 3], init_state=[1, 0])
#qiskit_op_tapered_11 = taper(qiskit_op_new, [2, 3], init_state=[1, 1])
print(qiskit_op_tapered_00[:5])