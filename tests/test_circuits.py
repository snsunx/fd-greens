import unittest
import numpy as np 
from qiskit import *
import sys
sys.path.append('../src/')
from circuits import *
from utils import get_unitary

'''
def diagonal_circuits_test():
    n_qubits = 5
    ansatz = QuantumCircuit(n_qubits)
    backend = Aer.get_backend('unitary_simulator')
    for ind in range(5):
        circ = get_diagonal_circuits(ansatz, ind, measure=False)
        result = execute(circ, backend).result()
        U = result.get_unitary(decimals=3)

        circ1 = get_diagonal_circuits1(ansatz, ind, measure=False)
        result = execute(circ1, backend).result()
        U1 = result.get_unitary(decimals=3)

        assert np.allclose(U, U1)
    
    print("Diagonal circuits test passed.")

def off_diagonal_circuits_test():
    n_qubits = 3
    ansatz = QuantumCircuit(n_qubits)
    backend = Aer.get_backend('unitary_simulator')

    for ind in range(n_qubits - 1):
        print(ind)
        circ = get_off_diagonal_circuits(ansatz, ind, ind + 1, measure=False)
        print(circ)
        result = execute(circ, backend).result()
        U = result.get_unitary(decimals=3)

        circ1 = get_off_diagonal_circuits1(ansatz, ind, ind + 1, measure=False)
        print(circ1)
        result = execute(circ1, backend).result()
        U1 = result.get_unitary(decimals=3)

        print(U[3, 3], U1[3, 3])
        assert np.allclose(U, U1)

    print("Off-diagonal circuits test passed.")
'''

class TestPushSwapGates(unittest.TestCase):
	'''
	def test_push_swap_gates_right(self):
		circ = QuantumCircuit(2)
		circ.swap(0, 1)
		circ.h(0)
		circ.x(1)
		circ.s(0)
		print(circ)

		U = get_unitary(circ)
		circ = push_swap_gates(circ)
		print(circ)

		U_ref = get_unitary(circ)
		np.testing.assert_almost_equal(U, U_ref)
	'''
	def test_push_swap_gates_left(self):
		circ = QuantumCircuit(2)
		circ.h(0)
		circ.x(1)
		circ.s(0)
		circ.swap(0, 1)
		print(circ)

		U = get_unitary(circ)
		circ = push_swap_gates(circ, direcs=['left'])
		print(circ)

		U_ref = get_unitary(circ)
		np.testing.assert_almost_equal(U, U_ref)
		
if __name__ == '__main__':
    unittest.main()
	#diagonal_circuits_test()
    #off_diagonal_circuits_test()
