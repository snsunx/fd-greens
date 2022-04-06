"""
===============================================================
Circuit Constructor (:mod:`fd_greens.main.circuit_constructor`)
===============================================================
"""

from typing import Tuple, Optional, Union, Sequence, List
from cmath import polar

import numpy as np
from permutation import Permutation

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Qubit, Clbit, Barrier
from qiskit.extensions import XGate, HGate, RXGate, RZGate, CPhaseGate, CZGate
from qiskit.quantum_info import SparsePauliOp

from .params import CCZGate
from ..utils import create_circuit_from_inst_tups

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]


class CircuitConstructor:
    """A class to construct circuits for calculating transition amplitudes."""

    def __init__(
        self,
        ansatz: QuantumCircuit,
        add_barriers: bool = True,
        anc: Sequence[int] = [0, 1],
    ) -> None:
        """Creates a CircuitConstructor object.

        Args:
            ansatz: The ansatz quantum circuit containing the ground state.
            add_barriers: Whether to add barriers to the circuit.
            ccx_inst_tups: The instruction tuples for customized CCX gate.
            anc: Indices of the ancilla qubits in the off-diagonal circuits.
        """
        self.ansatz = ansatz.copy()
        self.add_barriers = add_barriers
        # self.ccx_inst_tups = ccx_inst_tups
        # TODO: anc is not necessary. Can possibly deprecate this variable.
        self.anc = anc
        self.sys = [i for i in range(4) if i not in anc]  # Total # of qubits = 4

    def build_diagonal(self, a_op: SparsePauliOp) -> QuantumCircuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            a_op: The creation/annihilation operator of the circuit.

        Returns:
            circ: The new circuit with the creation/annihilation operator added.
        """
        assert len(a_op) == 2
        n_qubits = 3  # Hardcoded
        barrier_inst_tups = [(Barrier(n_qubits), range(n_qubits), [])]

        # Copy the ansatz circuit into the system qubit indices.
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [q._index + 1 for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)

        # Add Hadamard gate on the ancilla at the beginning.
        inst_tups += [(HGate(), [0], [])]
        if self.add_barriers:
            inst_tups += barrier_inst_tups

        # Add C-gate of the first operator.
        inst_tups += self._get_controlled_gate_inst_tups(a_op[0], ctrl_states=[0])
        if self.add_barriers:
            inst_tups += barrier_inst_tups

        # Add C-gate of the second operator.
        inst_tups += self._get_controlled_gate_inst_tups(a_op[1], ctrl_states=[1])
        if self.add_barriers:
            inst_tups += barrier_inst_tups

        # Add Hadamard gate on the ancilla at the end.
        inst_tups += [(HGate(), [0], [])]

        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    def build_off_diagonal(
        self, a_op_m: SparsePauliOp, a_op_n: SparsePauliOp
    ) -> QuantumCircuit:
        """Constructs the circuit to calculate an off-diagonal transition amplitude.

        Args:
            a_op_m: The first operator in the circuit.
            a_op_n: The second operator in the circuit.

        Returns:
            circ: The new circuit with the two operators appended.
        """
        assert len(a_op_m) == len(a_op_n) == 2
        n_qubits = 4  # Hardcoded
        barrier_inst_tups = [(Barrier(n_qubits), range(n_qubits), [])]

        # Copy the ansatz circuit into the system qubit indices.
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [q._index + 2 for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)

        # Add Hadamard gates on the ancillas at the beginning.
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        if self.add_barriers:
            inst_tups += barrier_inst_tups

        # Add the CC-gate of the first operator.
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[0], ctrl_states=[0, 0])
        if self.add_barriers:
            inst_tups += barrier_inst_tups
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[1], ctrl_states=[1, 0])
        if self.add_barriers:
            inst_tups += barrier_inst_tups

        # Add the phase gate in the middle if the second operator is not all I or all Z.
        if not set("".join(a_op_n.table.to_labels())).issubset({"I", "Z"}):
            inst_tups += [(RZGate(np.pi / 4), [self.anc[1]], [])]
            if self.add_barriers:
                inst_tups += barrier_inst_tups

        # Add the CC-gate of the second operator.
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[0], ctrl_states=[0, 1])
        if self.add_barriers:
            inst_tups += barrier_inst_tups
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[1], ctrl_states=[1, 1])
        if self.add_barriers:
            inst_tups += barrier_inst_tups

        # Add the phase gate after the second operator if it is all I or all Z.
        if set("".join(a_op_n.table.to_labels())).issubset({"I", "Z"}):
            inst_tups += [(RZGate(np.pi / 4), [self.anc[1]], [])]
            if self.add_barriers:
                inst_tups += barrier_inst_tups

        # Add Hadamard gates on the ancillas at the end.
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    def _get_controlled_gate_inst_tups(
        self, sparse_pauli_op: SparsePauliOp, ctrl_states: Sequence[int] = [1]
    ) -> Sequence[InstructionTuple]:
        """Obtains the instruction tuple corresponding to a controlled-U gate.

        Args:
            sparse_pauli_op: The Pauli operator from which the controlled gate is constructed.
            ctrl_states: The qubit states on which the cU gate is controlled on.

        Returns:
            inst_tups: Instruction tuples corresponding to the controlled-U gate.
        """
        assert len(sparse_pauli_op) == 1  # Only a single Pauli string
        assert set(ctrl_states).issubset({0, 1})
        n_anc = len(ctrl_states)
        assert n_anc in [1, 2]  # Only single or double controlled gates
        coeff = sparse_pauli_op.coeffs[0]
        label = sparse_pauli_op.table.to_labels()[0][::-1]
        n_sys = len(label)
        inst_tups = []

        # Find the indices to apply X gates (for control on 0), H gates (for Pauli X)
        # and X(pi/2) gates (for Pauli Y) as well as the pivot (target qubit of the
        # multi-controlled gate). The pivot selection is hardcoded: for single-controlled
        # gates, the pivot is the smallest index; for double-controlled gate, the pivot is
        # the largest index.
        if set(label) != {"I"}:
            cnot_inds = [i + n_anc for i in range(n_sys) if label[i] != "I"]
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            # x_inds = [i for i in range(n_anc) if ctrl_states[i] == 0]
            h_inds = [i + n_anc for i in range(n_sys) if label[i] == "X"]
            rx_inds = [i + n_anc for i in range(n_sys) if label[i] == "Y"]
        else:  # Identity gate, only need to check the phase below.
            cnot_inds = []
            # x_inds = []
            h_inds = []
            rx_inds = []
        x_inds = [i for i in range(n_anc) if ctrl_states[i] == 0]

        # Apply the phase gate and controlled-Z gate.
        if len(ctrl_states) == 1:
            if coeff != 1:
                angle = polar(coeff)[1]
                inst_tups += [(RZGate(angle), [0], [])]
            if set(label) != {"I"}:
                inst_tups += [(CZGate(), [0, pivot], [])]
        elif len(ctrl_states) == 2:
            if coeff != 1:
                angle = polar(coeff)[1]
                assert angle in [np.pi / 2, np.pi, -np.pi / 2]
                if abs(angle) == np.pi / 2:
                    inst_tups += [(CPhaseGate(angle), [self.anc[0], self.anc[1]], [])]
                else:
                    inst_tups += [(CZGate(), [self.anc[0], self.anc[1]], [])]
            if set(label) != {"I"}:
                inst_tups += [(CCZGate, [self.anc[0], self.anc[1], pivot], [])]

        # Wrap CNOT gates around for multi-qubit Pauli string. Note that these CNOTs are applied
        # in the reverse direction because the controlled gate is Z.
        for ind in cnot_inds:
            cx_inst_tups = [
                (HGate(), [pivot], []),
                (CZGate(), [pivot, ind], []),
                (HGate(), [pivot], []),
            ]
            inst_tups = cx_inst_tups + inst_tups
            inst_tups = inst_tups + cx_inst_tups
            pivot = ind

        # Wrap X gates around for control on 0.
        for ind in x_inds:
            inst_tups = [(XGate(), [ind], [])] + inst_tups
            inst_tups = inst_tups + [(XGate(), [ind], [])]

        # Wrap H gates around for Pauli X.
        for ind in h_inds:
            inst_tups = [(HGate(), [ind], [])] + inst_tups
            inst_tups = inst_tups + [(HGate(), [ind], [])]

        # Wrap X(pi/2) gates around for Pauli Y.
        for ind in rx_inds:
            inst_tups = [(RXGate(np.pi / 2), [ind], [])] + inst_tups
            inst_tups = inst_tups + [
                (RZGate(np.pi), [ind], []),
                (RXGate(np.pi / 2), [ind], []),
                (RZGate(np.pi), [ind], []),
            ]

        return inst_tups
