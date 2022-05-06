"""
=============================================================
Transpilation (:mod:`fd_greens.cirq_ver.utils.transpilation`)
=============================================================
"""

import numpy as np
import cirq

from .general_utils import circuit_equal


class C0C0iXGate(cirq.ThreeQubitGate):
    """The Berkeley iToffoli gate."""

    def __init__(self):
        super(C0C0iXGate, self)
        self._name = "C0C0iX"

    def _num_qubits_(self) -> int:
        return 3

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [0, 1j, 0, 0, 0, 0, 0, 0],
                [1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
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
        return "(0)", "(0)", "iX"


C0C0iX = C0C0iXGate()


def convert_ccz_to_c0ixc0(circuit: cirq.Circuit) -> cirq.Circuit:
    """Converts CCZs to C0iXC0s in a Cirq circuit.
    
    Args:
        circuit: The circuit to be converted.
    
    Returns:
        circuit_new: The new circuit after conversion.
    """
    circuit_new = cirq.Circuit()
    print(circuit[:10])
    print(circuit[10:])

    count = 0
    for moment in circuit.moments:
        for op in moment:
            if op.gate.__str__() == "CCZ":
                qubits = op.qubits
                cizc_ops = [
                    cirq.X(qubits[0]),
                    cirq.H(qubits[1]),
                    cirq.X(qubits[2]),
                    C0C0iX(qubits[0], qubits[2], qubits[1]),
                    cirq.X(qubits[0]),
                    cirq.H(qubits[1]),
                    cirq.X(qubits[2]),
                ]

                cs_ops = [
                    cirq.SWAP(qubits[0], qubits[1]),
                    cirq.CZPowGate(exponent=-0.5)(qubits[1], qubits[2]),
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

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(circuit_new[:10])
    print(circuit_new[10:])
    assert circuit_equal(circuit, circuit_new, False)
    return circuit_new


def convert_swap_to_cz(circuit: cirq.Circuit) -> cirq.Circuit:
    """Converts SWAPs to CZs in a Cirq circuit.
    
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

    assert circuit_equal(circuit, circuit_new, False)
    return circuit_new


def convert_phxz_to_xpi2(circuit: cirq.Circuit) -> cirq.Circuit:
    """Converts PhasedXZGate to X(pi/2) and virtual Z gates.
    
    Args:
        circuit: The circuit to be converted.
        
    Returns:
        circuit_new: The circuit after conversion.
    """
    circuit_new = cirq.Circuit()
    for op in list(circuit.all_operations()):
        if isinstance(op.gate, cirq.PhasedXZGate):
            a = op.gate.axis_phase_exponent
            x = op.gate.x_exponent
            z = op.gate.z_exponent
            q = op.qubits[0]
            circuit_new.append(
                [
                    cirq.ZPowGate(exponent=-a - 0.5)(q),
                    cirq.XPowGate(exponent=0.5)(q),
                    cirq.ZPowGate(exponent=1.0 - x)(q),
                    cirq.XPowGate(exponent=0.5)(q),
                    cirq.ZPowGate(exponent=a + z - 0.5)(q),
                ],
            )
        else:
            circuit_new.append(op)

    assert circuit_equal(circuit, circuit_new)
    return circuit_new


def transpile_into_berkeley_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """Transpiles a circuit into native gates on the Berkeley device.
    
    Args:
        circuit: The circuit to be transpiled.
    
    Returns:
        circuit_new: The new circuit after transpilation.
    """
    circuit_new = circuit.copy()
    # Three-qubit transpilation
    # circuit_new = convert_ccz_to_c0ixc0(circuit)

    # Two-qubit transpilation
    circuit_new = convert_swap_to_cz(circuit_new)
    
    # Single-qubit transpilation
    cirq.merge_single_qubit_gates_into_phxz(circuit_new)
    circuit_new = convert_phxz_to_xpi2(circuit_new)
    return circuit_new
