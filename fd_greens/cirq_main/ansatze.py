"""
================================================
Ansatz Functions (:mod:`fd_greens.main.ansatze`)
================================================
"""

from typing import Callable, Sequence

import cirq

# from qiskit import QuantumCircuit, QuantumRegister

AnsatzFunction = Callable[[Sequence[float]], cirq.Circuit]


def build_ansatz_gs(params: Sequence[float]) -> cirq.Circuit:
    r"""Constructs an $N$-electron ansatz of the encoded Hamiltonian.
    
    The ansatz is of the form $$(R_y(\theta_2)\otimes R_y(\theta_3)) CZ
    (R_y(\theta_0)\otimes R_y(\theta_1)) |00\rangle$$.

    Args:
        params: A sequence of the parameters.

    Returns:
        ansatz: The ansatz circuit.
    """
    assert len(params) == 6
    qubits = [cirq.LineQubit(i) for i in range(2)]
    ansatz = cirq.Circuit()
    ansatz += [cirq.Ry(rads=params[0])(qubits[0]), cirq.Ry(rads=params[1])(qubits[1])]
    ansatz += [cirq.CZ(qubits[0], qubits[1])]
    ansatz += [cirq.Ry(rads=params[2])(qubits[0]), cirq.Ry(rads=params[3])(qubits[1])]
    ansatz += [cirq.CZ(qubits[0], qubits[1])]
    ansatz += [cirq.Ry(rads=params[4])(qubits[0]), cirq.Ry(rads=params[5])(qubits[1])]
    return ansatz


def build_ansatz_e(params: Sequence[float]) -> cirq.Circuit:
    r"""Constructs the ansatz for ($N$+1)-electron states of the encoded Hamiltonian.
    The ansatz is of the form :math:`(I \otimes R_y(\theta))|10\rangle`.

    Args:
        params: A sequence of the parameters.

    Returns:
        ansatz: The ansatz circuit.
    """
    assert len(params) == 1
    qubits = [cirq.LineQubits(i) for i in range(2)]
    ansatz = cirq.Circuit()
    ansatz += [cirq.X(qubits[0])]
    ansatz += [cirq.Ry(rads=params[0])(qubits[1])]
    return ansatz


def build_ansatz_h(params: Sequence[float]) -> cirq.Circuit:
    r"""Constructs the ansatz for ($N$-1)-electron states of the encoded Hamiltonian. 
    The ansatz is of the form :math:`(I \otimes R_y(\theta))|00\rangle`.
    
    Args:
        params: A sequence of the parameters.

    Returns:
        ansatz: The ansatz circuit.
    """
    assert len(params) == 1
    qubits = [cirq.LineQubit(i) for i in range(2)]
    ansatz = cirq.Circuit()
    ansatz += [cirq.Ry(rads=params[0])(qubits[1])]
    return ansatz
