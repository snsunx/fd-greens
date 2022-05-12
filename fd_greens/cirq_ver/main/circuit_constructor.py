"""
========================================================================
Circuit Constructor (:mod:`fd_greens.cirq_ver.main.circuit_constructor`)
========================================================================
"""

from itertools import product
from typing import Mapping, Sequence, Optional, Tuple
from cmath import polar

import numpy as np

import cirq


class CircuitConstructor:
    """Constructor of transition amplitude circuits."""

    def __init__(self, ansatz: cirq.Circuit, qubits: Sequence[cirq.Qid]) -> None:
        """Initializes a ``CircuitConstructor`` object.

        Args:
            ansatz: The ansatz circuit containing the ground state.
            qubits: The qubits on which the circuits are to be constructed.
        """
        self.ansatz = ansatz.copy()
        self.qubits = qubits

    def _get_controlled_gate_operations(
        self, dense_pauli_string: cirq.DensePauliString, ctrl_states: Sequence[int] = [1]
    ) -> None:
        """Appends a controlled gate to a circuit.

        Args:
            sparse_pauli_op: The Pauli operator from which the controlled gate is constructed.
            ctrl_states: The qubit states on which the cU gate is controlled on.
        """
        assert set(ctrl_states).issubset({0, 1})  # Control states must be 0 or 1
        coeff = dense_pauli_string.coefficient
        pauli_mask = dense_pauli_string.pauli_mask
        # print(f'{coeff = }')
        # print(f'{pauli_mask = }')

        operations = []

        # Find the indices to apply X gates (for control on 0), H gates (for Pauli X)
        # and X(pi/2) gates (for Pauli Y) as well as the pivot (target qubit of the
        # multi-controlled gate). The pivot selection is hardcoded: for single-controlled
        # gates, the pivot is the smallest index; for double-controlled gate, the pivot is
        # the largest index.
        if set(pauli_mask) != {0}:
            cnot_inds = [i + len(ctrl_states) for i, c in enumerate(pauli_mask) if c != 0]
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            h_inds = [i + len(ctrl_states) for i, c in enumerate(pauli_mask) if c == 1]
            rx_inds = [i + len(ctrl_states) for i, c in enumerate(pauli_mask) if c == 2]
        else:  # Identity gate, only need to check the phase below.
            cnot_inds = []
            h_inds = []
            rx_inds = []
        x_inds = [i for i, j in enumerate(ctrl_states) if j == 0]

        # Apply central CZ in the case of 1 control qubit or CCZ gate in the case of
        # 2 control qubits. When the coefficient is not 1, apply a phase gate in the
        # case of 1 control qubit and a controlled phase gate in the case of 2 control qubits.
        if len(ctrl_states) == 1:
            if coeff != 1:
                angle = polar(coeff)[1]
                operations += [cirq.ZPowGate(exponent=angle / np.pi)(self.qubits[0])]
            if set(pauli_mask) != {0}:
                operations += [cirq.CZ(self.qubits[0], self.qubits[pivot])]
        elif len(ctrl_states) == 2:
            if coeff != 1:
                angle = polar(coeff)[1]
                assert angle in [np.pi / 2, np.pi, -np.pi / 2]
                if abs(angle) == np.pi / 2:
                    operations += [cirq.CZPowGate(exponent=angle / np.pi)(self.qubits[0], self.qubits[1])]
                else:
                    operations += [cirq.CZ(self.qubits[0], self.qubits[1])]
            if set(pauli_mask) != {0}:
                operations += [cirq.CCZ(self.qubits[0], self.qubits[1], self.qubits[pivot])]
        else:
            raise ValueError("Number of control qubits can only be 1 or 2.")

        # Wrap CNOT gates around for multi-qubit Pauli string. Note that these CNOTs are applied
        # in the reverse direction because the controlled gate is Z.
        for ind in cnot_inds:
            cx_ops = [cirq.H(self.qubits[pivot]), 
                      cirq.CZ(self.qubits[pivot], self.qubits[ind]),
                      cirq.H(self.qubits[pivot])]
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

        # print(f'{operations = }')
        return operations

    def build_diagonal(
        self, 
        dense_pauli_string_x: cirq.DensePauliString,
        dense_pauli_string_y: cirq.DensePauliString
    ) -> cirq.Circuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            pauli_op: The Pauli operator to be applied.

        Returns:
            circuit: A circuit with the operator appended.
        """
        # Copy the ansatz circuit with offset 1.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 1 for q in op.qubits]))        

        # Append Hadamard gate on the ancilla at the beginning.
        circuit.append(cirq.H(self.qubits[0]))

        # Append the LCU circuit of the operator.
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_x, ctrl_states=[0]))
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_y, ctrl_states=[1]))

        # Append Hadamard gate on the ancilla at the end.
        circuit.append(cirq.H(self.qubits[0]))

        return circuit

    build_diagonal_circuit = build_diagonal

    def build_off_diagonal(
        self,
        dense_pauli_string_x1: cirq.DensePauliString,
        dense_pauli_string_y1: cirq.DensePauliString,
        dense_pauli_string_x2: cirq.DensePauliString,
        dense_pauli_string_y2: cirq.DensePauliString
    ) -> cirq.Circuit:
        """Constructs the circuit to calculate an off-diagonal transition amplitude.

        Args:
            first_op: The first Pauli operator to be applied.
            second_op: The second Pauli operator to be applied.

        Returns:
            circ: The new circuit with the two operators appended.
        """
        # Copy the ansatz circuit with offset 2.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 2 for q in op.qubits]))

        # Add Hadamard gates on the ancillas at the beginning.
        # inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        circuit.append(cirq.H(self.qubits[0]))
        circuit.append(cirq.H(self.qubits[1]))

        # Append the LCU circuit of the first operator.
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_x1, ctrl_states=[0, 0]))
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_y1, ctrl_states=[1, 0]))

        # Append the phase gate in the middle if the second operator is not all I or all Z.
        # if not set(dense_pauli_).issubset({0, 3}):
        circuit.append(cirq.ZPowGate(exponent=1.0 / 4)(self.qubits[1]))

        # Append the LCU circuit of the second operator.
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_x2, ctrl_states=[0, 1]))
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_y2, ctrl_states=[1, 1]))

        # Append the phase gate after the second operator if it is all I or all Z.
        # if set("".join(second_pauli_op.table.to_labels())).issubset({"I", "Z"}):
        #     circuit.append(cirq.ZPowGate(exponent=1.0 / 4)(self.qubits[1]))

        # Add Hadamard gates on the ancillas at the end.
        circuit.append(cirq.H(self.qubits[0]))
        circuit.append(cirq.H(self.qubits[1]))
        return circuit

    build_off_diagonal_circuit = build_off_diagonal

    @staticmethod
    def build_tomography_circuits(
        circuit: cirq.Circuit, qubits: Sequence[cirq.Qid], append_measure: bool = True
    ) -> Mapping[str, cirq.Circuit]:
        """Constructs tomography circuits on a given circuit.
        
        Args:
            circuit: The circuit to be tomographed.
            qubits: The qubits to be tomographed.
            append_measure: Whether to append measurement gates.
        
        Returns:
            tomography_circuits: A dictionary with tomography labels as keys and tomography 
                circuits as values."""
        tomography_labels = ["".join(x) for x in product("xyz", repeat=len(qubits))]
        tomography_circuits = dict()

        # TODO: Take swap gates into account.
        for label in tomography_labels:
            tomography_circuit = circuit.copy()
            for q, s in zip(qubits, label):
                if s == "x":
                    tomography_circuit.append(
                        [
                            cirq.ZPowGate(exponent=0.5)(q),
                            cirq.XPowGate(exponent=0.5)(q),
                            cirq.ZPowGate(exponent=0.5)(q),
                        ]
                    )
                elif s == "y":
                    tomography_circuit.append(
                        [cirq.XPowGate(exponent=0.5)(q), cirq.ZPowGate(exponent=0.5)(q)]
                    )

                if append_measure:
                    tomography_circuit.append(cirq.measure(q))

            tomography_circuits[label] = tomography_circuit

        return tomography_circuits
