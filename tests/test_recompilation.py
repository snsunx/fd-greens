import sys
sys.path.append('../src')

import unittest
import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from recompilation import CircuitRecompiler
from utils import get_statevector


class TestCircuitRecompiler(unittest.TestCase):
    def test_recompile_with_statevector(self):
        """Tests recompilation with respect to a statevector."""
        n_qubits = 6
        n_rounds = 6

        circ = QuantumCircuit(n_qubits)
        statevector = get_statevector(circ)

        A = np.kron(np.eye(2), np.random.random((2**(n_qubits-1), 2**(n_qubits-1))))
        U = expm(-1j * (A + A.T))
        
        circuit_recompiler = CircuitRecompiler(n_rounds=n_rounds)
        quimb_gates = circuit_recompiler.recompile(U, statevector)
        print(quimb_gates)


if __name__ == '__main__':
    unittest.main()
