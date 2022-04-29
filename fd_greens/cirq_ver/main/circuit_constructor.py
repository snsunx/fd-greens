"""
===============================================================
Circuit Constructor (:mod:`fd_greens.main.circuit_constructor`)
===============================================================
"""

from typing import Sequence, Optional
from cmath import polar

import numpy as np

import cirq
from qiskit.quantum_info import SparsePauliOp


class CircuitConstructor:
    """Constructor of transition amplitude circuits."""

    def __init__(
        self, ansatz: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]] = None
    ) -> None:
        """Initializes a ``CircuitConstructor`` object.

        Args:
            ansatz: The ansatz circuit containing the ground state.
            qubits: The qubits on which the circuits are to be constructed.
        """
        self.ansatz = ansatz.copy()
        if qubits is None:
            self.qubits = [cirq.LineQubit(i) for i in range(4)]
        else:
            self.qubits = qubits

    def build_diagonal(self, pauli_op: SparsePauliOp) -> cirq.Circuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            pauli_op: The Pauli operator to be applied.

        Returns:
            circuit: A circuit with the operator appended.
        """
        assert len(pauli_op) == 2

        # Copy the ansatz circuit into the system qubit indices.
        circuit = cirq.Circuit()
        for pauli_op in self.ansatz.all_operations():
            circuit.append(pauli_op.gate(*[q + 1 for q in pauli_op.qubits]))

        # Append Hadamard gate on the ancilla at the beginning.
        circuit.append(cirq.H(self.qubits[0]))

        # Append the LCU circuit of the operator.
        self.append_controlled_gate(circuit, pauli_op[0], ctrl_states=[0])
        self.append_controlled_gate(circuit, pauli_op[1], ctrl_states=[1])

        # Append Hadamard gate on the ancilla at the end.
        circuit.append(cirq.H(self.qubits[0]))

        return circuit

    def build_off_diagonal(
        self, first_pauli_op: SparsePauliOp, second_pauli_op: SparsePauliOp
    ) -> cirq.Circuit:
        """Constructs the circuit to calculate an off-diagonal transition amplitude.

        Args:
            first_op: The first Pauli operator to be applied.
            second_op: The second Pauli operator to be applied.

        Returns:
            circ: The new circuit with the two operators appended.
        """
        assert len(first_pauli_op) == len(second_pauli_op) == 2

        # Copy the ansatz circuit into the system qubit indices.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 2 for q in op.qubits]))

        # Add Hadamard gates on the ancillas at the beginning.
        # inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        circuit.append(cirq.H(self.qubits[0]))
        circuit.append(cirq.H(self.qubits[1]))

        # Append the LCU circuit of the first operator.
        circuit.append(
            self._get_controlled_gate_operations(first_pauli_op[0], ctrl_states=[0, 0])
        )
        circuit.append(
            self._get_controlled_gate_operations(first_pauli_op[1], ctrl_states=[1, 0])
        )

        # Append the phase gate in the middle if the second operator is not all I or all Z.
        if not set("".join(second_pauli_op.table.to_labels())).issubset({"I", "Z"}):
            circuit.append(cirq.rz(np.pi / 4)(self.qubits[1]))

        # Append the LCU circuit of the second operator.
        circuit.append(
            self._get_controlled_gate_operations(second_pauli_op[0], ctrl_states=[0, 1])
        )
        circuit.append(
            self._get_controlled_gate_operations(second_pauli_op[1], ctrl_states=[1, 1])
        )

        # Append the phase gate after the second operator if it is all I or all Z.
        if set("".join(second_pauli_op.table.to_labels())).issubset({"I", "Z"}):
            circuit.append(cirq.rz(np.pi / 4)(self.qubits[1]))

        # Add Hadamard gates on the ancillas at the end.
        circuit.append(cirq.H(self.qubits[0]))
        circuit.append(cirq.H(self.qubits[1]))
        return circuit

    def _get_controlled_gate_operations(
        self, sparse_pauli_op: SparsePauliOp, ctrl_states: Sequence[int] = [1]
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
        operations = []

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
                operations += [cirq.rz(angle)(self.qubits[0])]
            if set(label) != {"I"}:
                operations += [cirq.CZ(self.qubits[0], self.qubits[pivot])]
        elif len(ctrl_states) == 2:
            if coeff != 1:
                angle = polar(coeff)[1]
                assert angle in [np.pi / 2, np.pi, -np.pi / 2]
                if abs(angle) == np.pi / 2:
                    operations += [
                        cirq.CZPowGate(exponent=angle / np.pi)(
                            self.qubits[0], self.qubits[1]
                        )
                    ]
                else:
                    operations += [cirq.CZ(self.qubits[0], self.qubits[1])]
            if set(label) != {"I"}:
                operations += [
                    cirq.CCZ(self.qubits[0], self.qubits[1], self.qubits[pivot])
                ]

        # Wrap CNOT gates around for multi-qubit Pauli string. Note that these CNOTs are applied
        # in the reverse direction because the controlled gate is Z.
        for ind in cnot_inds:
            cx_ops = [
                cirq.H(self.qubits[pivot]),
                cirq.CZ(self.qubits[pivot], self.qubits[ind]),
                cirq.H(self.qubits[pivot]),
            ]
            operations = cx_ops + operations
            operations = operations + cx_ops
            pivot = ind

        # Wrap X gates around when controlled on 0.
        for ind in x_inds:
            x_ops = [cirq.X(self.qubits[ind])]
            operations = x_ops + operations
            operations = operations + x_ops

        # Wrap H gates around for Pauli X.
        for ind in h_inds:
            h_ops = [cirq.H(self.qubits[ind])]
            operations = h_ops + operations
            operations = operations + h_ops

        # Wrap X(pi/2) gate in front and X(-pi/2) at the end when applying Pauli Y.
        for ind in rx_inds:
            operations = [cirq.XPowGate(exponent=0.5)(self.qubits[ind])] + operations
            operations = operations + [cirq.XPowGate(exponent=-0.5)(self.qubits[ind])]

        return operations