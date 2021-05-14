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

def build_kosugi_lih_ansatz(num=1):
    """Constructs the UCC1 or UCC2 ansatz for LiH in Kosugi 2020."""
    assert num in [1, 2]
    ansatz = QuantumCircuit(12)
    theta1 = Parameter('theta1')
    theta2 = Parameter('theta2')

    def add_xxxy_term(i, j, k, l, angle):
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

    ansatz.x([0, 1, 2, 3])
    ansatz.barrier()
    if num == 1:
        add_xxxy_term(2, 3, 4, 5, theta1)
        ansatz.barrier()
        add_xxxy_term(2, 3, 10, 11, theta2)
    else:
        add_xxxy_term(2, 3, 6, 7, theta1)
        ansatz.barrier()
        add_xxxy_term(2, 3, 8, 9, theta2)

    return ansatz