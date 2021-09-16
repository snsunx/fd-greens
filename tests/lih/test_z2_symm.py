"""Tests Z2 symmetries using the Z2Symmetries class in Qiskit."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from ansatze import build_ne2_ansatz
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from greens_function import GreensFunction
from tools import get_quantum_instance
from constants import HARTREE_TO_EV
from qiskit.opflow import Z2Symmetries

# User-defined parameters.
bond_length = 1.6
cache_read = True
cache_write = False

ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]], 'sto3g', 
    occupied_inds=[0], active_inds=[1, 2])
qiskit_op = hamiltonian.qiskit_op
print(qiskit_op)
z2_symm = Z2Symmetries.find_Z2_symmetries(qiskit_op)
print(z2_symm)
