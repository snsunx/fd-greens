from typing import Optional

import numpy as np

import cirq
from qiskit import QuantumCircuit

from .general_utils import circuit_equal


class C0iXC0Gate(cirq.Gate):
    """The Berkeley iToffoli gate."""

    def __init__(self):
        super(C0iXC0Gate, self)
        self._name = "C0iXC0"

    def _num_qubits_(self) -> int:
        return 3

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [0, 0, 1j, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def __str__(self) -> str:
        return self._name

    def _circuit_diagram_info_(self, args):
        return "(0)", "iX", "(0)"


C0iXC0 = C0iXC0Gate()


def convert_ccz_to_c0ixc0(circuit: cirq.Circuit) -> cirq.Circuit:
    """Converts CCZ to C0iXC0 on a Cirq circuit.
    
    Args:
        circuit: The circuit to be converted.
    
    Returns:
        circuit_new: The new circuit after conversion.
    """
    circuit_new = cirq.Circuit()

    count = 0
    for moment in circuit.moments:
        for op in moment:
            if op.gate.__str__() == "CCZ":
                qubits = op.qubits
                cizc_ops = [
                    cirq.X(qubits[0]),
                    cirq.H(qubits[1]),
                    cirq.X(qubits[2]),
                    C0iXC0(qubits[0], qubits[1], qubits[2]),
                    cirq.X(qubits[0]),
                    cirq.H(qubits[1]),
                    cirq.X(qubits[2]),
                ]

                cs_ops = [
                    cirq.SWAP(qubits[0], qubits[1]),
                    cirq.CZPowGate(exponent=0.5)(qubits[1], qubits[2]),
                    cirq.SWAP(qubits[0], qubits[1]),
                ]

                if count % 2 == 0:
                    circuit_new.append(cizc_ops)
                    circuit_new.append(cs_ops)
                else:
                    circuit_new.append(cs_ops)
                    circuit_new.append(cizc_ops)
                count += 1

            else:
                circuit_new.append(op)

    assert circuit_equal(circuit, circuit_new)
    return circuit_new


def convert_swap_to_cz(circuit: QuantumCircuit) -> QuantumCircuit:
    """Converts SWAP to CZ on a Cirq circuit.
    
    Args:
        circuit: The circuit to be converted.
        
    Returns:
        circuit_new: The circuit after conversion.
    """
    circuit_new = cirq.Circuit()

    for op in list(circuit.all_operations())[::-1]:
        if str(op.gate) == "SWAP":
            q0, q1 = op.qubits
            if convert_swap:
                circuit_new.insert(
                    0,
                    [
                        cirq.XPowGate(exponent=0.5)(q0),
                        cirq.XPowGate(exponent=0.5)(q1),
                        cirq.CZ(q0, q1),
                    ]
                    * 3,
                )
            else:
                circuit_new.insert(0, op)
        else:
            circuit_new.insert(0, op)
            convert_swap = True

    assert circuit_equal(circuit, circuit_new)
    return circuit_new

def optimize_circuit(circuit: cirq.Circuit) -> None:
    pass