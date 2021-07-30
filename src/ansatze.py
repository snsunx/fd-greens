import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal

def build_two_local_ansatz(n_qubits):
    """Constructs the ansatz for VQE."""
    ansatz = TwoLocal(n_qubits, ['ry', 'rz'], 'cz', reps=1)
    return ansatz

def build_ne1_ansatz(n_qubits):
    """Constructs a one-electron ansatz for VQE."""
    ansatz = QuantumCircuit(n_qubits)
    theta = Parameter('theta')
    ansatz.x(0)
    ansatz.u(-theta - np.pi / 2, 0, -np.pi, 2)
    ansatz.x(2)
    ansatz.u(theta + np.pi / 2, np.pi, 0, 2)
    ansatz.cx(2, 0)
    return ansatz

def build_ne3_ansatz(n_qubits):
    """Constructs a three-electron ansatz for VQE."""
    ansatz = QuantumCircuit(n_qubits)
    theta = Parameter('theta')
    ansatz.x([0, 1, 2])
    ansatz.u(-theta - np.pi / 2, 0, -np.pi, 3)
    ansatz.x(3)
    ansatz.u(theta + np.pi / 2, np.pi, 0, 3)
    ansatz.cx(3, 1)
    return ansatz

def build_ne2_ansatz(n_qubits):
    ansatz = QuantumCircuit(n_qubits)
    theta = (Parameter('theta1'), Parameter('theta2'), Parameter('theta3'))
    ansatz.x([0, 1, 3])
    ansatz.u(-theta[0] - np.pi / 2, 0, -np.pi, 2)
    ansatz.u(-theta[1] - np.pi / 2, 0, -np.pi, 3)
    ansatz.x([2, 3])
    ansatz.u(theta[0] + np.pi / 2, np.pi, 0, 2)
    ansatz.u(theta[1] + np.pi / 2, np.pi, 0, 3)
    ansatz.cx(3, 1)
    ansatz.z(2)
    ansatz.h(1)
    ansatz.cx(2, 1)
    ansatz.h(1)
    ansatz.u(-theta[2] - np.pi / 2, 0, -np.pi, 2)
    ansatz.x(2)
    ansatz.u(theta[2] + np.pi / 2, np.pi, 0, 2)
    ansatz.cx(2, 0)
    return ansatz

def add_xxxy_term(ansatz, i, j, k, l, angle):
    """An internal function for constructing Kosugi 2020's ansatze."""
    ansatz.h([i, j, k])
    ansatz.rx(np.pi / 2, l)
    ansatz.cx(i, j)
    ansatz.cx(j, k)
    ansatz.cx(k, l)
    ansatz.rz(angle, l)
    ansatz.cx(k, l)
    ansatz.cx(j, k)
    ansatz.cx(i, j)
    ansatz.h([i, j, k])
    ansatz.rx(-np.pi / 2, l)
        
def build_kosugi_lih_ansatz(ind=1):
    """Constructs the UCC1 or UCC2 ansatz for LiH in Kosugi 2020."""
    assert ind in [1, 2]
    ansatz = QuantumCircuit(12)
    theta1 = Parameter('theta1')
    theta2 = Parameter('theta2')

    ansatz.x(range(4))
    ansatz.barrier()
    if ind == 1:
        add_xxxy_term(ansatz, 2, 3, 4, 5, theta1)
        ansatz.barrier()
        add_xxxy_term(ansatz, 2, 3, 10, 11, theta2)
    else:
        add_xxxy_term(ansatz, 2, 3, 6, 7, theta1)
        ansatz.barrier()
        add_xxxy_term(ansatz, 2, 3, 8, 9, theta2)

    return ansatz

def build_kosugi_h2o_ansatz(ind=1):
    """Constructs the UCC1 or UCC2 ansatz for H2O in Kosugi 2020."""
    assert ind in [1, 2]
    ansatz = QuantumCircuit(14)
    angles = [Parameter(chr(i)) for i in range(97, 103)]

    ansatz.x(range(10))
    add_xxxy_term(ansatz, 6, 7, 10, 11, angles[0])
    add_xxxy_term(ansatz, 6, 7, 12, 13, angles[1])
    add_xxxy_term(ansatz, 8, 9, 10, 11, angles[2])
    add_xxxy_term(ansatz, 8, 9, 12, 13, angles[3])

    if ind == 1:
        add_xxxy_term(ansatz, 4, 5, 10, 11, angles[4])
        add_xxxy_term(ansatz, 4, 5, 12, 13, angles[5])

    return ansatz

