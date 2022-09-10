"""
==============================================
Transpilation (:mod:`fd_greens.transpilation`)
==============================================
"""
import os
from typing import Optional, Tuple
import warnings

import numpy as np
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

from .general_utils import unitary_equal
from .parameters import CircuitConstructionParameters, CHECK_CIRCUITS, SYNTHESIZE_WITH_BQSKIT

class C0C0iXGate(cirq.Gate):
    """The Berkeley iToffoli gate."""

    def __init__(self):
        super(C0C0iXGate, self)
        self._name = "C0C0iX"

    def _num_qubits_(self) -> int:
        return 3

    def _has_unitary_(self) -> bool:
        return True

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

iToffoli = C0C0iX
iToffoliGate = C0C0iXGate


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

def convert_ccz_to_c0c0ix(circuit: cirq.Circuit, spin: str, constrain_cs_csd: bool) -> cirq.Circuit:
    """Converts CCZs to C0C0iXs in a circuit.
    
    Args:
        circuit: The circuit to be converted.
        spin: Spin states of the molecular system.
        constrain_cs_csd: Whether to constrain CSD to be on Q4Q5 etc.
    
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
                if constrain_cs_csd:
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

convert_ccz_to_itoffoli = convert_ccz_to_c0c0ix

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
                if SYNTHESIZE_WITH_BQSKIT:
                    print("> Synthesize with BQskit")
                    qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
u3(-1.5704498526160793, 2.347165319631653, 9.105617284690338) q[0];
u3(-1.570791974014007, 2.8931923819384826, 2.032905420195417) q[1];
u3(-9.111957770878731e-23, 1.166243864141404, 6.337282727827317) q[2];
cz q[0], q[1];
u3(4.699826538690047, 5.047405432461273, -2.3468188332590545) q[0];
u3(1.570449852636876, 2.2775508022122035, 4.960788279773917) q[1];
cz q[1], q[2];
u3(1.5707963267946075, 3.734252996837955, 5.5889932745478506) q[1];
u3(3.141592653589793, 1.122570935823188, 1.8114154066967805) q[2];
cz q[0], q[1];
u3(1.5707963267950922, 6.435940946679477, 4.377373500646986) q[0];
u3(2.3561944301608424, 0.24436817897049987, 5.690871465430451) q[1];
cz q[0], q[1];
u3(1.5707963267948968, 4.435313033435187, 3.7742351774873515) q[0];
u3(5.497787170600082, 4.527145873531501, -0.24387815184137246) q[1];
cz q[1], q[2];
u3(3.926990818484793, 5.395333768572616, 4.897709483745008) q[1];
u3(6.283185307179586, 4.314978661687423, 2.4999895140649824) q[2];
cz q[0], q[1];
u3(4.7123882928914655, 1.049587584556453, 1.8353098312956908) q[0];
u3(4.712388980970318, 2.296970791437661, 0.8879062662019785) q[1];
cz q[1], q[2];
u3(3.1290206591070877, -4.833690399469691, 3.9472290073267575) q[1];
u3(3.141592653589793, 2.8968112698778037, 1.918019666775366) q[2];
cz q[0], q[1];
u3(4.7123342571079325, 4.246151519879995, 3.6632913842920525) q[0];
u3(7.853982321467928, 14.460456011155228, -1.4884780258587462) q[1];
                    """
                    # with open("../../../ccz_circuit.txt", "r") as f:
                    #     qasm_str = f.read()
                    ccz_circuit = circuit_from_qasm(qasm_str)
                    qubit_map = {cirq.NamedQubit(f"q_{i}"): qubits[i] for i in range(3)}
                    ccz_circuit_new = ccz_circuit.transform_qubits(qubit_map)
                    # {cirq.NamedQubit(f"q_{i}"): qubits[i] for i in range(3)})
                    circuit_new += ccz_circuit_new
                    print(circuit_new.all_qubits())
                else:
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


def transpile_into_berkeley_gates(
    circuit: cirq.Circuit,
    spin: str = 'd',
    circuit_params: Optional[CircuitConstructionParameters] = None
) -> cirq.Circuit:
    """Transpiles a circuit into native gates on the Berkeley device.
    
    Args:
        circuit: The circuit to be transpiled.
        spin: Spin states of the molecular system.
        circuit_params: Circuit construction parameters.
    
    Returns:
        circuit_new: The new circuit after transpilation.
    """
    if circuit_params is None:
        circuit_params = CircuitConstructionParameters()
    
    circuit_new = circuit.copy()

    # Qubit permutation.
    circuit_new = permute_qubits(circuit_new)
    cirq.MergeInteractions(allow_partial_czs=False).optimize_circuit(circuit_new)
    
    # Three-qubit gate transpilation.
    if circuit_params.CONVERT_CCZ_TO_ITOFFOLI:
        circuit_new = convert_ccz_to_itoffoli(
            circuit_new, spin, constrain_cs_csd=circuit_params.CONSTRAIN_CS_CSD)
    else:
        circuit_new = convert_ccz_to_cz(circuit_new)
    cirq.MergeInteractions(allow_partial_czs=True).optimize_circuit(circuit_new)

    # Two-qubit gate transpilation.
    circuit_new = convert_swap_to_cz(circuit_new)
    
    # Single-qubit gate transpilation.
    circuit_new = transpile_1q_gates(circuit_new)
    return circuit_new
