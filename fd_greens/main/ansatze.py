"""
================================================
Ansatz Functions (:mod:`fd_greens.main.ansatze`)
================================================
"""

from typing import Callable, Sequence
from qiskit import QuantumCircuit, QuantumRegister

AnsatzFunction = Callable[[Sequence[float]], QuantumCircuit]


def build_ansatz_gs(params: Sequence[float]) -> QuantumCircuit:
    r"""Constructs an N-electron ansatz of the encoded Hamiltonian. The ansatz is of
    the form :math:`(R_y(\theta_2)\otimes R_y(\theta_3)) CZ (R_y(\theta_0)\otimes R_y(\theta_1))
    |00\rangle`.

    Args:
        params: A sequence of the parameters.

    Returns:
        ansatz: The ansatz quantum circuit.
    """
    assert len(params) == 6
    qreg = QuantumRegister(2, name="q")
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cz(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)
    ansatz.cz(0, 1)
    ansatz.ry(params[4], 0)
    ansatz.ry(params[5], 0)
    return ansatz


def build_ansatz_e(params: Sequence[float]) -> QuantumCircuit:
    r"""Constructs the ansatz for (N+1)-electron states of the encoded Hamiltonian.
    The ansatz is of the form :math:`(I \otimes R_y(\theta))|10\rangle`.

    Args:
        params: A sequence of the parameters.

    Returns:
        ansatz: The ansatz quantum circuit.
    """
    assert len(params) == 1
    qreg = QuantumRegister(2, name="q")
    ansatz = QuantumCircuit(qreg)
    ansatz.x(0)
    ansatz.ry(params[0], 1)
    return ansatz


def build_ansatz_h(params: Sequence[float]) -> QuantumCircuit:
    r"""Constructs the ansatz for (N-1)-electron states of the encoded Hamiltonian. 
    The ansatz is of the form :math:`(I \otimes R_y(\theta))|00\rangle`.
    
    Args:
        params: A sequence of the parameters.

    Returns:
        ansatz: The ansatz quantum circuit.
    """
    assert len(params) == 1
    qreg = QuantumRegister(2, name="q")
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 1)
    return ansatz
