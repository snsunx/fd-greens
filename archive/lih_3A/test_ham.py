"""Calculates the spectral function on the active-space LiH Hamilonian 
using the classmethods."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from qiskit.opflow import PauliSumOp
from qiskit.circuit import Barrier
from ansatze import *
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate, UnitaryGate, SwapGate

from functools import partial
from operators import SecondQuantizedOperators
from greens_function_restricted import GreensFunctionRestricted
from utils import get_quantum_instance
from constants import HARTREE_TO_EV

from z2_symmetries import transform_4q_hamiltonian

# User-defined parameters.
bond_length = 3.0
save_params = False 
load_params = True
cache_read = False
cache_write = False


#ansatz = build_two_local_ansatz(2)
ansatz = build_2q_ansatz()
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]], 'sto3g', 
    occ_inds=[0], act_inds=[1, 2])

qiskit_op = transform_4q_hamiltonian(hamiltonian.qiskit_op, init_state=[1, 1])
print(qiskit_op.reduce())

transform_func = partial(transform_4q_hamiltonian, init_state=[1, 1])
second_q_ops = SecondQuantizedOperators(4)
second_q_ops.transform(transform_func)
pauli_op_dict = second_q_ops.get_op_dict(spin='down')

print('')
x0 = PauliSumOp(pauli_op_dict[0][0])
x0H = x0.compose(qiskit_op)
x0H = x0H.reduce()
for x in x0H:
    print(x.primitive.table)
    print('')