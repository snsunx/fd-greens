"""
==========================================================
Circuit Constructor (:mod:`fd_greens.circuit_constructor`)
==========================================================
"""

from itertools import product
from typing import Mapping, Sequence, Optional, List
from cmath import polar

import numpy as np
import cirq

from .transpilation import transpile_1q_gates
from .general_utils import get_gate_counts


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
        self, dense_pauli_string: cirq.DensePauliString, control_states: Sequence[int] = [1]
    ) -> List[cirq.Operation]:
        """Returns the controlled gate operations corresponding to a Pauli string."""
        assert set(control_states).issubset({0, 1})  # Control states must be 0 or 1
        coeffient = dense_pauli_string.coefficient
        pauli_mask = dense_pauli_string.pauli_mask

        operations = []

        # Find the indices to apply X gates (for control on 0), H gates (for Pauli X)
        # and X(pi/2) gates (for Pauli Y) as well as the pivot (target qubit of the
        # multi-controlled gate). 
        # The pivot selection is hardcoded: for single-controlled
        # gates, the pivot is the smallest index; for double-controlled gate, the pivot is
        # the largest index.
        if np.any(pauli_mask):
            cnot_indices = [i + len(control_states) for i, x in enumerate(pauli_mask) if x != 0]
            pivot = min(cnot_indices)
            cnot_indices.remove(pivot)
            h_indices = [i + len(control_states) for i, x in enumerate(pauli_mask) if x == 1]
            rx_indices = [i + len(control_states) for i, x in enumerate(pauli_mask) if x == 2]
        else:  # Identity gate, only need to check the phase below.
            cnot_indices = []
            h_indices = []
            rx_indices = []
        x_indices = [i for i, j in enumerate(control_states) if j == 0]

        # Apply central CZ in the case of 1 control qubit or CCZ gate in the case of
        # 2 control qubits. When the coefficient is not 1, apply a phase gate in the
        # case of 1 control qubit and a controlled phase gate in the case of 2 control qubits.
        if len(control_states) == 1:
            if coeffient != 1:
                angle = polar(coeffient)[1]
                operations += [cirq.ZPowGate(exponent=angle / np.pi)(self.qubits[0])]
            if np.any(pauli_mask):
                operations += [cirq.CZ(self.qubits[0], self.qubits[pivot])]
        elif len(control_states) == 2:
            if coeffient != 1:
                angle = polar(coeffient)[1]
                # assert abs(angle) in [np.pi / 2, np.pi]
                if abs(angle) == np.pi / 2:
                    operations += [cirq.CZPowGate(exponent=angle / np.pi)(self.qubits[0], self.qubits[1])]
                elif abs(angle) == np.pi:
                    operations += [cirq.CZ(self.qubits[0], self.qubits[1])]
                else:
                    raise ValueError("Pauli string coefficient must be ±1, ±1j.")

            if np.any(pauli_mask):
                operations += [cirq.CCZ(self.qubits[0], self.qubits[1], self.qubits[pivot])]
        else:
            raise ValueError("Number of control qubits can only be 1 or 2.")

        # Wrap CNOT gates around for multi-qubit Pauli string. Note that these CNOTs are applied
        # in the reverse direction because the controlled gate is Z.
        for ind in cnot_indices:
            cx_operations = [cirq.H(self.qubits[pivot]), 
                             cirq.CZ(self.qubits[pivot], self.qubits[ind]),
                             cirq.H(self.qubits[pivot])]
            operations = cx_operations + operations + cx_operations
            # operations = operations + cx_operations
            pivot = ind

        # Wrap X gates around when controlled on 0.
        for ind in x_indices:
            x_operations = [cirq.X(self.qubits[ind])]
            operations = x_operations + operations + x_operations
            # operations = operations + x_operations

        # Wrap H gates around for Pauli X.
        for ind in h_indices:
            h_operations = [cirq.H(self.qubits[ind])]
            operations = h_operations + operations + h_operations
            # operations = operations + h_operations

        # Wrap X(pi/2) gate in front and X(-pi/2) at the end when applying Pauli Y.
        for ind in rx_indices:
            rx_operations_front = [cirq.XPowGate(exponent=0.5)(self.qubits[ind])]
            rx_operations_back = [cirq.XPowGate(exponent=-0.5)(self.qubits[ind])]
            operations = rx_operations_front + operations + rx_operations_back

        return operations

    def build_diagonal_circuit(
        self, 
        dense_pauli_string_x: cirq.DensePauliString,
        dense_pauli_string_y: cirq.DensePauliString
    ) -> cirq.Circuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            dense_pauli_string_x: The X part of the Jordan-Wigner transformed operator.
            dense_pauli_string_y: The Y part of the Jordan-Wigner transformed operator.

        Returns:
            circuit: A circuit with the second quantized operator appended.
        """
        # Copy the ansatz circuit with offset 1.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 1 for q in op.qubits]))

        # Append Hadamard gate on the ancilla at the beginning.
        circuit.append(cirq.H(self.qubits[0]))

        # Append the LCU circuit of the operator.
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_x, control_states=[0]))
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_y, control_states=[1]))

        # Append Hadamard gate on the ancilla at the end.
        circuit.append(cirq.H(self.qubits[0]))

        return circuit

    def build_off_diagonal_circuit(
        self,
        dense_pauli_string_x1: cirq.DensePauliString,
        dense_pauli_string_y1: cirq.DensePauliString,
        dense_pauli_string_x2: cirq.DensePauliString,
        dense_pauli_string_y2: cirq.DensePauliString
    ) -> cirq.Circuit:
        """Constructs the circuit to calculate an off-diagonal transition amplitude.

        Args:
            dense_pauli_string_x1: The X part of the first Jordan-Wigner transformed operator.
            dense_pauli_string_y1: The Y part of the first Jordan-Wigner transformed operator.
            dense_pauli_string_x2: The X part of the second Jordan-Wigner transformed operator.
            dense_pauli_string_y2: The Y part of the second Jordan-Wigner transformed operator.

        Returns:
            circuit: The new circuit with the two second-quantized operators appended.
        """
        # Copy the ansatz circuit with offset 2.
        circuit = cirq.Circuit()
        for op in self.ansatz.all_operations():
            circuit.append(op.gate(*[q + 2 for q in op.qubits]))

        # Add Hadamard gates on the ancillas at the beginning.
        circuit.append([cirq.H(self.qubits[0]), cirq.H(self.qubits[1])])

        # Append the LCU circuit of the first operator.
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_x1, control_states=[0, 0]))
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_y1, control_states=[1, 0]))

        # Append the phase gate in the middle if the second operator is not all I or all Z.
        # if not set(dense_pauli_).issubset({0, 3}):
        circuit.append(cirq.ZPowGate(exponent=1.0 / 4)(self.qubits[1]))

        # Append the LCU circuit of the second operator.
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_x2, control_states=[0, 1]))
        circuit.append(self._get_controlled_gate_operations(dense_pauli_string_y2, control_states=[1, 1]))

        # Append the phase gate after the second operator if it is all I or all Z.
        # if set("".join(second_pauli_op.table.to_labels())).issubset({"I", "Z"}):
        #     circuit.append(cirq.ZPowGate(exponent=1.0 / 4)(self.qubits[1]))

        # Add Hadamard gates on the ancillas at the end.
        circuit.append([cirq.H(self.qubits[0]), cirq.H(self.qubits[1])])
        return circuit

    @staticmethod
    def build_tomography_circuits(
        circuit: cirq.Circuit, 
        tomographed_qubits: Sequence[cirq.Qid],
        measured_qubits: Optional[Sequence[cirq.Qid]] = None
    ) -> Mapping[str, cirq.Circuit]:
        """Constructs tomography circuits on a given circuit.
        
        Args:
            circuit: The circuit to be tomographed.
            tomographed_qubits: The qubits to be tomographed.
            measured_qubits: The qubits to be measured.
        
        Returns:
            tomography_circuits: A dictionary with tomography labels as keys and tomography circuits as values.
        """
        if measured_qubits is None:
            measured_qubits = tomographed_qubits
        
        tomography_labels = ["".join(x) for x in product("xyz", repeat=len(tomographed_qubits))]
        tomography_circuits = dict()

        # TODO: Take swap gates at the end of the circuit into account.
        for label in tomography_labels:
            # Look for the position where the first multi-qubit gate is encountered.
            # This is general because in the last transpilation step the circuits have been converted into
            # the form with alternating single- and multi-qubit gates.
            position = 0
            gate_count = 0
            while gate_count == 0:
                position -= 1
                gate_count = get_gate_counts(cirq.Circuit(circuit[position]), criterion=lambda op: op.gate.num_qubits() > 1)
            
            # Break the circuit at the position where the multi-qubit gate is encountered.
            if position != -1:
                tomography_circuit = circuit[:position + 1]
                measurement_circuit = circuit[position + 1:]
            else:
                tomography_circuit = circuit.copy()
                measurement_circuit = cirq.Circuit()
                    
            # Append H or X(pi/2) to the measurement circuit.
            for q, s in zip(tomographed_qubits, label):
                if s == "x":
                    measurement_circuit.append(cirq.ZPowGate(exponent=0.5)(q))
                    measurement_circuit.append(cirq.XPowGate(exponent=0.5)(q))
                    measurement_circuit.append(cirq.ZPowGate(exponent=0.5)(q))
                elif s == "y":
                    measurement_circuit.append(cirq.I(q))
                    measurement_circuit.append(cirq.XPowGate(exponent=0.5)(q))
                    measurement_circuit.append(cirq.ZPowGate(exponent=0.5)(q))
            
            measurement_circuit = transpile_1q_gates(measurement_circuit)
            for q in measured_qubits:
                measurement_circuit.append(cirq.measure(q))

            tomography_circuit = tomography_circuit + measurement_circuit
            tomography_circuits[label] = tomography_circuit

        return tomography_circuits
