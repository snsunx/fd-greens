"""
===============================================================
Circuit Constructor (:mod:`fd_greens.main.circuit_constructor`)
===============================================================
"""

from typing import Tuple, Optional, Union, Sequence, List, Any
from cmath import polar

import numpy as np

import cirq
from qiskit.quantum_info import SparsePauliOp

InstructionTuple = Any


class CircuitConstructor:
    """A class to construct circuits for calculating transition amplitudes."""

    def __init__(self, ansatz: cirq.Circuit) -> None:
        """Initializes a ``CircuitConstructor`` object.

        Args:
            ansatz: The ansatz circuit containing the ground state.
            anc: Indices of the ancilla qubits in the off-diagonal circuits.
        """
        self.ansatz = ansatz.copy()
        self.qubits = [cirq.LineQubit(i) for i in range(4)]

    def build_diagonal(self, op: SparsePauliOp) -> cirq.Circuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            op: The operator in the circuit.

        Returns:
            circuit: A circuit with the operator appended.
        """
        assert len(op) == 2

        # Copy the ansatz circuit into the system qubit indices.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 1 for q in op.qubits]))

        # Append Hadamard gate on the ancilla at the beginning.
        circuit.append(cirq.H(self.qubits[0]))

        # Append the LCU circuit of the operator.
        self.append_controlled_gate(circuit, op[0], ctrl_states=[0])
        self.append_controlled_gate(circuit, op[1], ctrl_states=[1])

        # Append Hadamard gate on the ancilla at the end.
        circuit.append(cirq.H(self.qubits[0]))

        return circuit

    def build_off_diagonal(
        self, first_op: SparsePauliOp, second_op: SparsePauliOp
    ) -> cirq.Circuit:
        """Constructs the circuit to calculate an off-diagonal transition amplitude.

        Args:
            first_op: The first operator in the circuit.
            second_op: The second operator in the circuit.

        Returns:
            circ: The new circuit with the two operators appended.
        """
        assert len(first_op) == len(second_op) == 2

        # Copy the ansatz circuit into the system qubit indices.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 2 for q in op.qubits]))

        # Add Hadamard gates on the ancillas at the beginning.
        # inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        circuit.append(cirq.H(self.qubits[0]))
        circuit.append(cirq.H(self.qubits[1]))

        # Append the LCU circuit of the first operator.
        self.append_controlled_gate(circuit, first_op[0], ctrl_states=[0, 0])
        self.append_controlled_gate(circuit, first_op[1], ctrl_states=[1, 0])

        # Append the phase gate in the middle if the second operator is not all I or all Z.
        if not set("".join(second_op.table.to_labels())).issubset({"I", "Z"}):
            circuit.append(cirq.rz(np.pi / 4)(self.qubits[1]))

        # Append the LCU circuit of the second operator.
        self.append_controlled_gate(circuit, second_op[0], ctrl_states=[0, 1])
        self.append_controlled_gate(circuit, second_op[1], ctrl_states=[1, 1])

        # Append the phase gate after the second operator if it is all I or all Z.
        if set("".join(second_op.table.to_labels())).issubset({"I", "Z"}):
            circuit.append(cirq.rz(np.pi / 4)(self.qubits[1]))

        # Add Hadamard gates on the ancillas at the end.
        circuit.append(cirq.H(self.qubits[0]))
        circuit.append(cirq.H(self.qubits[1]))
        return circuit

    def append_controlled_gate(
        self,
        circuit: cirq.Circuit,
        sparse_pauli_op: SparsePauliOp,
        ctrl_states: Sequence[int] = [1],
    ) -> None:
        """Appends a controlled gate to a circuit.

        Args:
            sparse_pauli_op: The Pauli operator from which the controlled gate is constructed.
            ctrl_states: The qubit states on which the cU gate is controlled on.
        """
        assert len(sparse_pauli_op) == 1  # Only a single Pauli string
        assert set(ctrl_states).issubset({0, 1})  # Control states must be 0 or 1
        n_anc = len(ctrl_states)
        assert n_anc in [1, 2]  # Only single or double controlled gates
        coeff = sparse_pauli_op.coeffs[0]
        label = sparse_pauli_op.table.to_labels()[0][::-1]
        n_sys = len(label)

        # Find the indices to apply X gates (for control on 0), H gates (for Pauli X)
        # and X(pi/2) gates (for Pauli Y) as well as the pivot (target qubit of the
        # multi-controlled gate). The pivot selection is hardcoded: for single-controlled
        # gates, the pivot is the smallest index; for double-controlled gate, the pivot is
        # the largest index.
        if set(label) != {"I"}:
            cnot_inds = [i + n_anc for i in range(n_sys) if label[i] != "I"]
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            h_inds = [i + n_anc for i in range(n_sys) if label[i] == "X"]
            rx_inds = [i + n_anc for i in range(n_sys) if label[i] == "Y"]
        else:  # Identity gate, only need to check the phase below.
            cnot_inds = []
            h_inds = []
            rx_inds = []
        x_inds = [i for i in range(n_anc) if ctrl_states[i] == 0]

        # Apply central CZ in the case of 1 control qubit or CCZ gate in the case of
        # 2 control qubits. When the coefficient is not 1, apply a phase gate in the
        # case of 1 control qubit and a controlled phase gate in the case of 2 control qubits.
        if len(ctrl_states) == 1:
            if coeff != 1:
                angle = polar(coeff)[1]
                circuit.append(cirq.rz(angle)(self.qubits[0]))
            if set(label) != {"I"}:
                circuit.append(cirq.CZ(self.qubits[0], self.qubits[pivot]))
        elif len(ctrl_states) == 2:
            if coeff != 1:
                angle = polar(coeff)[1]
                assert angle in [np.pi / 2, np.pi, -np.pi / 2]
                if abs(angle) == np.pi / 2:
                    circuit.append(
                        cirq.CZPowGate(exponent=angle / np.pi)(
                            self.qubits[0], self.qubits[1]
                        )
                    )
                else:
                    circuit.append(cirq.CZ(self.qubits[0], self.qubits[1]))
            if set(label) != {"I"}:
                circuit.append(
                    cirq.CCZ(self.qubits[0], self.qubits[1], self.qubits[pivot])
                )

        # Wrap CNOT gates around for multi-qubit Pauli string. Note that these CNOTs are applied
        # in the reverse direction because the controlled gate is Z.
        for ind in cnot_inds:
            cx_ops = [
                cirq.H(self.qubits[pivot]),
                cirq.CZ(self.qubits[pivot], self.qubits[ind]),
                cirq.H(self.qubits[pivot]),
            ]
            circuit.insert(0, cx_ops)
            circuit.append(cx_ops)
            pivot = ind

        # Wrap X gates around when controlled on 0.
        for ind in x_inds:
            circuit.insert(0, cirq.X(self.qubits[ind]))
            circuit.append(cirq.X(self.qubits[ind]))

        # Wrap H gates around for Pauli X.
        for ind in h_inds:
            circuit.insert(0, cirq.H(self.qubits[ind]))
            circuit.append(cirq.H(self.qubits[ind]))

        # Wrap X(pi/2) gate in front and X(-pi/2) at the end when applying Pauli Y.
        for ind in rx_inds:
            circuit.insert(0, cirq.rx(np.pi / 2)(self.qubits[ind]))
            circuit.append(
                [
                    cirq.rz(np.pi)(self.qubits[ind]),
                    cirq.rx(np.pi / 2)(self.qubits[ind]),
                    cirq.rz(np.pi)(self.qubits[ind]),
                ]
            )
