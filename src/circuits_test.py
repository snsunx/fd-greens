import numpy as np 
from qiskit import *
from circuits import *

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

if __name__ == '__main__':
    #diagonal_circuits_test()
    off_diagonal_circuits_test()
