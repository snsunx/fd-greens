"""
==============================================
Transpilation (:mod:`fd_greens.transpilation`)
==============================================
"""

from typing import Tuple
import warnings

import numpy as np
import cirq

from .general_utils import unitary_equal
from .parameters import CircuitConstructionParameters, CHECK_CIRCUITS

class C0C0iXGate(cirq.Gate):
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

    def _circuit_diagram_info_(self, args) -> Tuple[str, str, str]:
        return "(0)", "(0)", "iX"


C0C0iX = C0C0iXGate()


def permute_qubits(circuit: cirq.Circuit) -> cirq.Circuit:
    """Permutes qubits of long-range CZs and CCZs by adding SWAP gates.
    
    Args:
        circuit: The circuit to be converted.
        
    Returns:
        circuit_new: The new circuit after conversion.
    """
    def is_adjacent(qubits):
        indices = [q.x for q in qubits]
        assert indices == sorted(indices)
        return np.all(np.diff(indices) == 1)

    circuit_new = cirq.Circuit()
    for moment in circuit:
        for op in moment:
            gate = op.gate
            qubits = sorted(list(op.qubits)).copy()
            if len(qubits) > 1 and not is_adjacent(qubits):
                # The action is to insert a SWAP gate to permute the qubit after qubits[-2]
                # with qubits[-1]. The additional SWAP gate inserted at the beginning is for
                # easy simplification.
                assert gate in [cirq.CZ, cirq.CCZ]
                warnings.warn("Adding SWAP gates to permute qubits.")
                qubits_swap = (qubits[-2] + 1, qubits[-1])
                qubits_new = qubits[:-1] + [qubits[-2] + 1]
                circuit_new.insert(0, cirq.SWAP(*qubits_swap))
                circuit_new.append(cirq.SWAP(*qubits_swap))
                circuit_new.append(gate(*qubits_new))
                circuit_new.append(cirq.SWAP(*qubits_swap))
            else:
                circuit_new.append(op)

    if CHECK_CIRCUITS:
        assert unitary_equal(circuit, circuit_new)
    return circuit_new

def convert_ccz_to_c0c0ix(circuit: cirq.Circuit, spin: str) -> cirq.Circuit:
    """Converts CCZs to C0C0iXs in a circuit.
    
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

                # CiZC gate from the C0C0iX gate, with XHX gates appended on both sides.
                cizc_ops = [
                    cirq.X(qubits[0]), cirq.H(qubits[1]), cirq.X(qubits[2]),
                    C0C0iX(qubits[0], qubits[2], qubits[1]),
                    cirq.X(qubits[0]), cirq.H(qubits[1]), cirq.X(qubits[2]),
                ]

                # qubits[0] is always Q4, qubits[1] is always Q5.
                if CircuitConstructionParameters.CONSTRAIN_CS_CSD:
                    qubits_csd = [qubits[0], qubits[1]]
                    qubits_swap = [qubits[1], qubits[2]]
                else:
                    qubits_csd = [qubits[1], qubits[2]]
                    qubits_swap = [qubits[0], qubits[1]]

                # TODO: This part can be explained in a better way.
                # SWAP gate without a CZ, equivalent to an iSWAP gate.
                iswap_ops = [
                    cirq.XPowGate(exponent=0.5)(qubits_swap[0]), cirq.XPowGate(exponent=0.5)(qubits_swap[1]),
                    cirq.CZ(*qubits_swap),
                    cirq.XPowGate(exponent=0.5)(qubits_swap[0]), cirq.XPowGate(exponent=0.5)(qubits_swap[1]),
                    cirq.CZ(*qubits_swap),
                    cirq.XPowGate(exponent=0.5)(qubits_swap[0]), cirq.XPowGate(exponent=0.5)(qubits_swap[1])
                ]

                # Long-range CS dagger gate absent an equivalent iSWAP gate.
                cs_ops = [cirq.CZPowGate(exponent=-0.5)(*qubits_csd),
                          cirq.CZ(*qubits_swap),
                          cirq.SWAP(*qubits_swap)]

                # For easy circuit optimization, the extra gates are placed alternately before or after CiZC.
                if count % 2 == (spin == 'u'):
                    circuit_new.append(cizc_ops)
                    circuit_new.append(iswap_ops)
                    circuit_new.append(cs_ops)
                else:
                    circuit_new.append(cs_ops[::-1])
                    circuit_new.append(iswap_ops)
                    circuit_new.append(cizc_ops)
                count += 1

            else:
                circuit_new.append(op)

    if CHECK_CIRCUITS:
        assert unitary_equal(circuit, circuit_new)
    return circuit_new

def convert_ccz_to_cz(circuit: cirq.Circuit) -> cirq.Circuit:
    """Decomposes CCZ gates to CZ gates in a circuit."""
    circuit_new = cirq.Circuit()

    HTGate = cirq.PhasedXZGate(axis_phase_exponent=-0.75, x_exponent=0.5, z_exponent=-0.75) # First T then H
    THGate = cirq.PhasedXZGate(axis_phase_exponent=-0.5,x_exponent=0.5, z_exponent=-0.75) # First H then T
    HTHGate = cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.25, z_exponent=0.0)
    HTdaggerHGate = cirq.PhasedXZGate(axis_phase_exponent=-1.0, x_exponent=0.25, z_exponent=0.0)

    for moment in circuit.moments:
        for op in moment:
            if op.gate.__str__() == "CCZ":
                qubits = op.qubits

                circuit_new.append(cirq.H(qubits[2]))
                circuit_new.append(cirq.CZ(qubits[1], qubits[2]))
                circuit_new.append(HTdaggerHGate(qubits[2]))
                circuit_new.append(cirq.SWAP(qubits[1], qubits[2]))

                circuit_new.append(cirq.CZ(qubits[0], qubits[1]))
                circuit_new.append(HTHGate(qubits[1]))
                circuit_new.append(cirq.CZ(qubits[1], qubits[2]))
                circuit_new.append(HTdaggerHGate(qubits[1]))
                circuit_new.append(cirq.CZ(qubits[0], qubits[1]))

                circuit_new.append(cirq.SWAP(qubits[1], qubits[2]))
                circuit_new.append(HTGate(qubits[1]))
                circuit_new.append(THGate(qubits[2]))
                circuit_new.append(cirq.CZ(qubits[0], qubits[1]))
                circuit_new.append(cirq.T(qubits[0]))
                circuit_new.append(HTdaggerHGate(qubits[1]))
                circuit_new.append(cirq.CZ(qubits[0], qubits[1]))
                circuit_new.append(cirq.H(qubits[1]))

            else:
                circuit_new.append(op)

    if CHECK_CIRCUITS:
        assert unitary_equal(circuit, circuit_new)
    return circuit_new


def convert_swap_to_cz(circuit: cirq.Circuit) -> cirq.Circuit:
    """Converts SWAPs to CZs in a circuit.
    
    Args:
        circuit: The circuit to be converted.
        
    Returns:
        circuit_new: The new circuit after conversion.
    """
    circuit1 = circuit.copy()
    circuit_new = cirq.Circuit()

    # print('In convert_swap_to_cz!!!\n')
    # print_circuit(circuit)

    convert_swap = False
    # for op in list(circuit1.all_operations())[::-1]:
    for moment in circuit1[::-1]:
        for op in moment:
            if str(op.gate) == "SWAP":
                if convert_swap:
                    # TODO: There should be a better way to handle this.
                    for _ in range(3):
                        circuit_new.insert(0, [cirq.XPowGate(exponent=0.5)(op.qubits[0])])
                        circuit_new.insert(0, [cirq.XPowGate(exponent=0.5)(op.qubits[1])])
                        circuit_new.insert(0, cirq.CZ(op.qubits[0], op.qubits[1]))
                else:
                    warnings.warn("Not converting SWAP gates at the end to CZs.")
                    circuit_new.insert(0, op)
            else:
                circuit_new.insert(0, op)
                convert_swap = True
            # print('circuit_new\n', circuit_new)

    if CHECK_CIRCUITS:
        assert unitary_equal(circuit, circuit_new)
    return circuit_new

# TODO: This can be combined with transpile_1q_gates into a single function.
def convert_phxz_to_xpi2(circuit: cirq.Circuit) -> cirq.Circuit:
    """Converts ``PhasedXZGate`` s to X(pi/2) gates and virtual Z gates.
    
    Args:
        circuit: The circuit to be converted.
        
    Returns:
        circuit_new: The new circuit after conversion.
    """
    circuit_new = cirq.Circuit()
    for op in list(circuit.all_operations()):
        if isinstance(op.gate, cirq.PhasedXZGate):
            a = op.gate.axis_phase_exponent
            x = op.gate.x_exponent
            z = op.gate.z_exponent
            q = op.qubits[0]

            # Build an array consisting of the exponents of the three Z gates. 
            # The generic gate sequence is ZXZXZ, but if the middle exponent is 1.0, the gate sequence is Z;
            # if the middle exponent is 0.5 or 1.5, the gate sequence is ZXZ.
            exponents = np.array([-a - 0.5, 1.0 - x, a + z - 0.5]) % 2
            if np.any(np.abs(np.array([0.5, 1.0, 1.5]) - exponents[1]) < 1e-8):
                if abs(exponents[1] - 1.0) < 1e-8:
                    circuit_new.append(cirq.ZPowGate(exponent=np.sum(exponents) % 2)(q))
                    circuit_new.append(cirq.I(q))
                    circuit_new.append(cirq.I(q))
                    circuit_new.append(cirq.I(q))
                    circuit_new.append(cirq.I(q))
                else:
                    circuit_new.append(cirq.ZPowGate(exponent=np.sum(exponents[:2]) % 2)(q))
                    circuit_new.append(cirq.XPowGate(exponent=0.5)(q))
                    circuit_new.append(cirq.ZPowGate(exponent=np.sum(exponents[1:]) % 2)(q))
                    circuit_new.append(cirq.I(q))
                    circuit_new.append(cirq.I(q))
            else:
                circuit_new.append(cirq.ZPowGate(exponent=exponents[0])(q))
                circuit_new.append(cirq.XPowGate(exponent=0.5)(q))
                circuit_new.append(cirq.ZPowGate(exponent=exponents[1])(q))
                circuit_new.append(cirq.XPowGate(exponent=0.5)(q))
                circuit_new.append(cirq.ZPowGate(exponent=exponents[2])(q))
        else:
            circuit_new.append(op)

    if CHECK_CIRCUITS:
        assert unitary_equal(circuit, circuit_new)
    return circuit_new

def transpile_1q_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """Transpiles single-qubit gates."""
    circuit_new = circuit.copy()
    cirq.MergeSingleQubitGates().optimize_circuit(circuit_new)
    cirq.DropEmptyMoments().optimize_circuit(circuit_new)
    cirq.merge_single_qubit_gates_into_phxz(circuit_new)
    circuit_new = convert_phxz_to_xpi2(circuit_new)
    
    # if CHECK_CIRCUITS:
    #     assert unitary_equal(circuit, circuit_new)
    return circuit_new


def transpile_into_berkeley_gates(circuit: cirq.Circuit, spin: str = 'd') -> cirq.Circuit:
    """Transpiles a circuit into native gates on the Berkeley device.
    
    Args:
        circuit: The circuit to be transpiled.
    
    Returns:
        circuit_new: The new circuit after transpilation.
    """
    circuit_new = circuit.copy()

    # Qubit permutation.
    circuit_new = permute_qubits(circuit_new)
    cirq.MergeInteractions(allow_partial_czs=False).optimize_circuit(circuit_new)
    
    # Three-qubit gate transpilation.
    if CircuitConstructionParameters.CONVERT_CCZ_TO_ITOFFOLI:
        circuit_new = convert_ccz_to_c0c0ix(circuit_new, spin)
    else:
        circuit_new = convert_ccz_to_cz(circuit_new)
    cirq.MergeInteractions(allow_partial_czs=True).optimize_circuit(circuit_new)

    # Two-qubit gate transpilation.
    circuit_new = convert_swap_to_cz(circuit_new)
    
    # Single-qubit gate transpilation.
    circuit_new = transpile_1q_gates(circuit_new)
    return circuit_new
