from typing import Callable, Sequence
from qiskit import QuantumCircuit, QuantumRegister

AnsatzFunction = Callable[[Sequence[float]], QuantumCircuit]

def build_ansatz_gs(params: Sequence[float]) -> QuantumCircuit:
    """Constructs an N-electron ansatz of the encoded Hamiltonian. The ansatz is of
    the form (Ry(theta2)\otimes Ry(theta3))CZ(Ry(theta0)\otimes Ry(theta1))|00>.

    Args:
        A sequence of the parameters.

    Returns:
        The ansatz quantum circuit.
    """
    assert len(params) == 4
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cz(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)
    return ansatz

def build_ansatz_e(params: Sequence[float]) -> QuantumCircuit:
    """Constructs the ansatz for (N+1)-electron states of the encoded Hamiltonian.
    The ansatz is of the form (I \otimes Ry(theta))|10>.

    Args:
        A sequence of the parameters.

    Returns:
        The ansatz quantum circuit.
    """
    assert len(params) == 1
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)
    ansatz.x(0)    
    ansatz.ry(params[0], 1)
    return ansatz

def build_ansatz_h(params: Sequence[float]) -> QuantumCircuit:
    """Constructs the ansatz for (N-1)-electron states of the encoded Hamiltonian. 
    The ansatz is of the form (I \otimes Ry(theta))|00>.
    
    Args:
        A sequence of the parameters.

    Returns:
        The ansatz quantum circuit.
    """
    assert len(params) == 1
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)  
    ansatz.ry(params[0], 1)
    return ansatz
