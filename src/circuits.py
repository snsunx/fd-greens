"""Circuit construction module."""

from typing import Tuple, Optional, Iterable, Union, Sequence, List
from cmath import polar

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction, Qubit, Clbit, Barrier
from qiskit.extensions import CCXGate, XGate, HGate, RXGate, CXGate, RZGate, CPhaseGate, CZGate, PhaseGate
from qiskit.quantum_info import SparsePauliOp

import params
from params import CCZGate
from utils import (get_unitary, create_circuit_from_inst_tups, split_circuit_across_barriers)

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]

class CircuitConstructor:
    """A class to construct circuits for calculating transition amplitudes in Green's functions."""

    def __init__(self,
                 ansatz: QuantumCircuit,
                 add_barriers: bool = True,
                 ccx_inst_tups: Iterable[InstructionTuple] = [(CCXGate(), [0, 1, 2], [])],
                 anc: Sequence[int] = [0, 1]
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
        self.ccx_inst_tups = ccx_inst_tups
        self.anc = anc
        self.sys = [i for i in range(4) if i not in anc]

        ccx_inst_tups_matrix = get_unitary(ccx_inst_tups)
        ccx_matrix = CCXGate().to_matrix()
        assert np.allclose(ccx_inst_tups_matrix, ccx_matrix)

        # TODO: Forgot why need this part. Probably can remove
        self.ccx_angle = 0
        # self.ccx_angle = polar(ccx_inst_tups_matrix[3, 7])[1]
        # ccx_inst_tups_matrix[3, 7] /= np.exp(1j * self.ccx_angle)
        # ccx_inst_tups_matrix[7, 3] /= np.exp(1j * self.ccx_angle)

    def build_eh_diagonal(self, a_op: List[SparsePauliOp]) -> QuantumCircuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            The creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the creation/annihilation operator appended.
        """
        # Copy the circuit with empty ancilla positions
        circ = self._copy_circuit_with_ancilla([0])

        # Apply the gates corresponding to the creation/annihilation terms
        if self.add_barriers: circ.barrier()
        circ.h(0)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op[0], ctrl=[0], n_anc=1)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op[1], ctrl=[1], n_anc=1)
        if self.add_barriers: circ.barrier()
        circ.h(0)
        if self.add_barriers: circ.barrier()

        return circ

    def build_eh_diagonal1(self, a_op: SparsePauliOp) -> QuantumCircuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            The creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the creation/annihilation operator added.
        """
        assert len(a_op) == 2

        # Copy the ansatz circuit into the system qubit indices
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [q._index + 1 for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)
        
        # Main part of the circuit to add creation/annihilation operator
        inst_tups += [(HGate(), [0], [])]
        if self.add_barriers: inst_tups += [(Barrier(3), range(3), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op[0], ctrl_states=[0])
        if self.add_barriers: inst_tups += [(Barrier(3), range(3), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op[1], ctrl_states=[1])
        if self.add_barriers: inst_tups += [(Barrier(3), range(3), [])]
        inst_tups += [(HGate(), [0], [])]

        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    def build_eh_off_diagonal(self,
                              a_op_m: List[SparsePauliOp],
                              a_op_n: List[SparsePauliOp]
                              ) -> QuantumCircuit:
        """Constructs the circuit to calculate off-diagonal transition amplitudes.

        Args:
            a_op_m: The first creation/annihilation operator of the circuit.
            a_op_n: The second creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the two creation/annihilation operators added.
        """
        # Copy the circuit with empty ancilla positions
        circ = self._copy_circuit_with_ancilla([0, 1])

        # Apply the gates corresponding to the creation/annihilation terms
        # if self.add_barriers: circ.barrier()
        circ.h([0, 1])
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_m[0], ctrl=(0, 0), n_anc=2)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_m[1], ctrl=(1, 0), n_anc=2)
        if self.add_barriers: circ.barrier()
        circ.rz(np.pi / 4, 1)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_n[0], ctrl=(0, 1), n_anc=2)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_n[1], ctrl=(1, 1), n_anc=2)
        if self.add_barriers: circ.barrier()
        circ.h([0, 1])

        return circ

    def build_eh_off_diagonal1(self,
                               a_op_m: SparsePauliOp,
                               a_op_n: SparsePauliOp
                               ) -> QuantumCircuit:
        """Constructs the circuit to calculate off-diagonal transition amplitudes.

        Args:
            a_op_m: The first creation/annihilation operator of the circuit.
            a_op_n: The second creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the two creation/annihilation operators appended.
        """
        assert len(a_op_m) == len(a_op_n) == 2

        # Copy the ansatz circuit into the system qubit indices
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [self.sys[q._index] for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)

        # Add the first creation/annihilation operator
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[0], ctrl_states=[0, 0])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[1], ctrl_states=[1, 0])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]

        # Add the phase gate in the middle
        inst_tups += [(RZGate(np.pi/4), [self.anc[1]], [])]
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]

        # Add the second creation/annihilation operator
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[0], ctrl_states=[0, 1])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[1], ctrl_states=[1, 1])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        
        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    def build_charge_diagonal(self, U_op: SparsePauliOp) -> QuantumCircuit:
        """Constructs the circuit to calculate diagonal charge-charge transition amplitudes."""
        circ = self._copy_circuit_with_ancilla([0, 1])
        
        # Apply the gates corresponding to a charge operator
        if self.add_barriers: circ.barrier()
        circ.h([0, 1])
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, -U_op, ctrl=(1, 0), n_anc=2) # iXY = -Z
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, U_op, ctrl=(0, 1), n_anc=2) # iYX = Z
        if self.add_barriers: circ.barrier()
        circ.cz(0, 1)
        if self.add_barriers: circ.barrier()
        circ.h([0, 1])

        return circ

    def _copy_circuit_with_ancilla(self, anc: Sequence[int]) -> QuantumCircuit:
        """Copies a circuit with specific indices for ancillas.

        Args:
            Indices of the ancilla qubits.

        Returns:
            The new quantum circuit with empty ancilla positions.
        """
        # Create a new circuit along with the quantum registers
        n_sys = self.ansatz.num_qubits
        n_anc = len(anc)
        n_qubits = n_sys + n_anc
        inds_new = [i for i in range(n_qubits) if i not in anc]
        qreg_new = QuantumRegister(n_qubits, name='q')
        circ_new = QuantumCircuit(qreg_new)

        # Copy instructions from the ansatz circuit to the new circuit
        for inst, qargs, cargs in self.ansatz.data:
            qargs = [inds_new[q._index] for q in qargs]
            circ_new.append(inst, qargs, cargs)
        return circ_new

    def _apply_controlled_gate(self,
                              circ: QuantumCircuit,
                              op: SparsePauliOp,
                              ctrl: Sequence[int] = [1],
                              n_anc: int = 1) -> None:
        """Applies a controlled-U gate to a quantum circuit.

        Args:
            circ: The quantum circuit on which the controlled-U gate is applied.
            op: The operator from which the controlled gate is constructed.
            ctrl: The qubit state on which the cU gate is controlled on.
            n_anc: Number of ancilla qubits.

        Raises:
            NotImplementedError: Control on more than two qubits is not implemented.
        """
        # Sanity check
        assert set(ctrl).issubset({0, 1})
        assert len(ctrl) <= 2
        assert len(op.coeffs) == 1
        coeff = op.coeffs[0]
        label = op.table.to_labels()[0]
        amp, angle = polar(coeff)
        assert amp == 1

        ind_max = len(label) - 1
        label_tmp = label
        for i in range(len(label)):
            if label_tmp[0] == 'I':
                label_tmp = label_tmp[1:]
                ind_max -= 1

        ind_min = 0
        label_tmp = label
        for i in range(len(label)):
            if label_tmp[-1] == 'I':
                label_tmp = label_tmp[:-1]
                ind_min += 1

        # Prepend X gates for control on 0
        for i in range(len(ctrl)):
            if ctrl[i] == 0:
                circ.x(i)

        # Prepend rotation gates for Pauli X and Y
        for i, c in enumerate(label[::-1]):
            if c == 'X':
                circ.h(i + n_anc)
            elif c == 'Y':
                circ.rx(np.pi / 2, i + n_anc)

        # Prepend CNOT gates for Pauli strings
        for i in range(ind_max + n_anc, ind_min + n_anc, -1):
            circ.cx(i, i - 1)

        # Prepend SWAP gates when ind_min != 0
        if ind_min != 0:
            circ.swap(n_anc, ind_min + n_anc)
        
        # Apply the controlled gate
        if len(ctrl) == 1: # single-controlled gate
            if coeff != 1:
                circ.p(angle, 0)
            if set(list(label)) == {'I'}:
                pass
            else:
                circ.cz(0, n_anc)
        elif len(ctrl) == 2: # double-controlled gate
            if coeff != 1:
                circ.cp(angle, 0, 1)
            if set(list(label)) == {'I'}:
                if self.ccx_angle != 0:
                    circ.cp(self.ccx_angle, 0, 1)
            else:
                circ.h(n_anc)
                # Apply the central CCX gate
                if self.ccx_inst_tups is not None:
                    for inst_tup in self.ccx_inst_tups:
                        circ.append(*inst_tup)
                else:
                    circ.ccx(0, n_anc, 1)
                circ.h(n_anc)
        else:
            raise NotImplementedError("Control on more than two qubits is not implemented")

        # Append SWAP gate when ind_min != 0
        if ind_min != 0:
            circ.swap(n_anc, ind_min + n_anc)

        # Append CNOT gates for Pauli strings
        for i in range(ind_min + n_anc, ind_max + n_anc):
            circ.cx(i + 1, i)

        # Append rotation gates for Pauli X and Y
        for i, c in enumerate(label[::-1]):
            if c == 'X':
                circ.h(i + n_anc)
            elif c == 'Y':
                circ.rx(-np.pi / 2, i + n_anc)

        # Append X gates for control on 0
        for i in range(len(ctrl)):
            if ctrl[i] == 0:
                circ.x(i)

    def _get_controlled_gate_inst_tups(
            self,
            sparse_pauli_op: SparsePauliOp,
            ctrl_states: Sequence[int] = [1]) -> Sequence[InstructionTuple]:
        """Applies a controlled-U gate to a quantum circuit.

        Args:
            sparse_pauli_op: The operator from which the controlled gate is constructed.
            ctrl_states: The qubit states on which the cU gate is controlled on.
        """
        assert len(sparse_pauli_op) == 1
        coeff = sparse_pauli_op.coeffs[0]
        label = sparse_pauli_op.table.to_labels()[0][::-1]
        _, angle = polar(coeff)

        # Find the indices and the pivot
        if len(ctrl_states) == 1:
            cnot_inds = [i + 1 for i in range(len(label)) if label[i] != 'I']
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            x_inds = [0] if ctrl_states == [0] else []
            h_inds = [i + 1 for i in range(len(label)) if label[i] == 'X']
            rx_inds = [i + 1 for i in range(len(label)) if label[i] == 'Y']
        else:
            cnot_inds = [self.sys[i] for i in range(len(label)) if label[i] != 'I']
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            x_inds = [self.anc[i] for i in range(len(ctrl_states)) if ctrl_states[i] == 0]
            h_inds = [self.sys[i] for i in range(len(label)) if label[i] == 'X']
            rx_inds = [self.sys[i] for i in range(len(label)) if label[i] == 'Y']

        # print('pivot =', pivot)
        # print('cnot_inds =', cnot_inds)

        # Apply the controlled gate
        inst_tups = []
        if len(ctrl_states) == 1:
            if coeff != 1:
                inst_tups += [(PhaseGate(angle), [0], [])]
            if set(label) != {'I'}:
                inst_tups += [(CZGate(), [0, pivot], [])]
        elif len(ctrl_states) == 2:
            if coeff != 1:
                inst_tups += [(CPhaseGate(angle), [self.anc[0], self.anc[1]], [])]
            if set(label) == {'I'}:
                if self.ccx_angle != 0:
                    inst_tups += [(CPhaseGate(self.ccx_angle), [self.anc[0], self.anc[1]], [])]
            else:
                inst_tups += [(CCZGate, [self.anc[0], self.anc[1], pivot], [])]

        # Wrap CNOT gates around for multi-qubit Pauli string
        # Note that these CNOTs are applied in the reverse way
        for ind in cnot_inds:
            cx_inst_tups = [(HGate(), [pivot], []),
                            (CZGate(), [pivot, ind], []),
                            (HGate(), [pivot], [])]
            inst_tups = cx_inst_tups + inst_tups
            inst_tups = inst_tups + cx_inst_tups
            pivot = ind

        # Wrap X gates around for control on 0
        for ind in x_inds:
            inst_tups = [(XGate(), [ind], [])] + inst_tups
            inst_tups = inst_tups + [(XGate(), [ind], [])]

        # Wrap H gates around for Pauli X
        for ind in h_inds:
            inst_tups = [(HGate(), [ind], [])] + inst_tups
            inst_tups = inst_tups + [(HGate(), [ind], [])]

        # Wrap Rx gates around for Pauli Y
        for ind in rx_inds:
            inst_tups = [(RXGate(np.pi/2), [ind], [])] + inst_tups
            inst_tups = inst_tups + [(RXGate(-np.pi/2), [ind], [])]

        return inst_tups


    @staticmethod
    def append_tomography_gates(circ: QuantumCircuit, 
                                qubits: Iterable[QubitLike],
                                label: Tuple[str],
                                use_u3: bool = True) -> QuantumCircuit:
        """Appends tomography gates to a circuit.

        Args:
            circ: The circuit to which tomography gates are to be appended.
            qubits: The qubits to be tomographed.
            label: The tomography states label.
            use_u3: Whether to use U3 rather than Clifford gates for basis change.
        
        Returns:
            A new circuit with tomography gates appended.
        """
        assert len(qubits) == len(label)
        tomo_circ = circ.copy()

        for q, s in zip(qubits, label):
            if s == 'x':
                if use_u3:
                    tomo_circ.u3(np.pi/2, 0, np.pi, q)
                else:
                    tomo_circ.h(q)
            elif s == 'y':
                if use_u3:
                    tomo_circ.u3(np.pi/2, 0, np.pi/2, q)
                else:
                    tomo_circ.sdg(q)
                    tomo_circ.h(q)
        return tomo_circ

    @staticmethod
    def append_measurement_gates(circ: QuantumCircuit) -> QuantumCircuit:
        """Appends measurement gates to a circuit.
        
        Args:
            The circuit to which measurement gates are to be appended.
            
        Returns:
            A new circuit with measurement gates appended.
        """
        n_qubits = len(circ.qregs[0])
        circ.barrier()
        circ.add_register(ClassicalRegister(n_qubits))
        circ.measure(range(n_qubits), range(n_qubits))
        return circ
    
class CircuitTranspiler:
    """A class for circuit transpilation."""

    def __init__(self, 
                 basis_gates: Sequence[str] = params.basis_gates,
                 swap_gates_pushed: bool = True,
                 swap_direcs_round1: List[List[str]] = params.swap_direcs_round1,
                 swap_direcs_round2: List[List[str]] = params.swap_direcs_round2
                ) -> None:
        """Initializes a CircuitTranspiler object.
        
        Args:
            basis_gates: A sequence of strings representing the basis gates used in transpilation.
            swap_gates_pushed: Whether SWAP gates are pushed.
            swap_direcs_round1: Strings indicating to which direction each SWAP gate is pushed 
                in the first round.
            swap_direcs_round2: Strings indicating to which direction each SWAP gate is pushed 
                in the second round.
        """
        self.basis_gates = basis_gates
        self.swap_gates_pushed = swap_gates_pushed
        self.swap_direcs_round1 = swap_direcs_round1
        self.swap_direcs_round2 = swap_direcs_round2

    def transpile(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpiles a circuit using Qiskit's built-in transpile function."""
        import qiskit
        circ_new = qiskit.transpile(circ, basis_gates=self.basis_gates)
        return circ_new

    def transpile_across_barriers(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpiles a circuit across barriers."""
        inst_tups_all = split_circuit_across_barriers(circ)

        def remove_first_swap_gate(circ_):
            """Removes the first SWAP gate in a circuit."""
            for i, inst_tup in enumerate(circ_.data):
                if inst_tup[0].name == 'swap':
                    del circ_.data[i]
                    break

        def swap_cp_and_u3(circ_):
            """Swaps the positions of CPhase and U3 gates."""
            tmp = circ_.data[6]
            circ_.data[6] = circ_.data[5]
            circ_.data[5] = circ_.data[4]
            circ_.data[4] = tmp

        # Transpile except for three-qubit gates
        qreg = circ.qregs[0]
        circ_new = QuantumCircuit(qreg)
        count = 0
        for i, inst_tups_single in enumerate(inst_tups_all):
            if len(inst_tups_single) > 1: # not 3-qubit gate
                # Create a single circuit between the barriers
                circ_single = create_circuit_from_inst_tups(inst_tups_single, qreg=qreg)
                circ_single = self.transpile(circ_single)

                # Swap positions of CPhase and U3 in the 5th sub-circuit
                if i == 4: swap_cp_and_u3(circ_single)
                
                # Push SWAP gates
                if self.swap_gates_pushed:
                    # First round pushes do not push through two-qubit gates
                    circ_single = self.push_swap_gates(
                        circ_single, 
                        direcs=self.swap_direcs_round1[count].copy(),
                        qreg=qreg)
                    circ_single = self.combine_swap_gates(circ_single)

                    # Second-round pushes push through two-qubit gates
                    circ_single = self.push_swap_gates(
                        circ_single, 
                        direcs=self.swap_direcs_round2[count].copy(),
                        qreg=qreg, 
                        push_through_2q=True)
                    # Final transpilation
                    circ_single = self.transpile(circ_single)

                    # Remove SWAP gates at the beginning
                    if i == 0: remove_first_swap_gate(circ_single)
                
                circ_new += circ_single
                count += 1

            else: # 3-qubit gate
                circ_new.barrier()
                circ_new.append(*inst_tups_single[0])
                circ_new.barrier()

        return circ_new

    @staticmethod
    def push_swap_gates(circ: QuantumCircuit, 
                        direcs: List[str] = [],
                        qreg: Optional[QuantumRegister] = None,
                        push_through_2q: bool = False) -> QuantumCircuit:
        """Pushes the swap gates across single- and two-qubit gates.
        
        Args:
            circ: The quantum circuit on which SWAP gates are pushed.
            direcs: The directions to which each SWAP gate is pushed.
            qreg: The quantum register of the circuit.
            push_through_2q: Whether to push through two-qubit gates.

        Returns:
            A new circuit on which SWAP gates are pushed.
        """
        assert set(direcs).issubset({'left', 'right', None})
        if direcs == []: return circ
        if qreg is None: qreg = circ.qregs[0]
        n_qubits = len(qreg)
        
        # Prepend and append barriers for easy processing
        inst_tups = circ.data.copy()
        inst_tups = [(Barrier(n_qubits), qreg, [])] + inst_tups
        inst_tups = inst_tups + [(Barrier(n_qubits), qreg, [])]
        n_inst_tups = len(inst_tups)

        # Store positions of SWAP gates that need to be pushed
        swap_gate_pos = []
        count = 0
        for i, inst_tup in enumerate(inst_tups):
            if inst_tup[0].name == 'swap' and direcs[count] is not None:
                swap_gate_pos.append(i)
                count += 1
        direcs = [d for d in direcs if d is not None]

        # Reorder SWAP gate positions due to pushing rightmost gates when pushing to the right
        for i, direc in enumerate(direcs):
            if direc == 'right':
                swap_gate_pos[i] *= -1
        sort_inds = np.argsort(swap_gate_pos)
        swap_gate_pos = [abs(swap_gate_pos[i]) for i in sort_inds]
        direcs = [direcs[i] for i in sort_inds]

        for i in swap_gate_pos:
            inst_tup = inst_tups[i]
            qargs = inst_tup[1]
            direc = direcs.pop(0)
            delete_pos = i + int(direc == 'left') # if direc == 'left', delete at i + 1

            # Push the SWAP gate to the left or the right. Insertion done before removal
            j = i + (-1) ** (direc == 'left')
            while j != 0 or j != n_inst_tups - 1:
                insert_pos = j + int(direc == 'left') # if direc == left, insert at right side
                inst_, qargs_, cargs_ = inst_tups[j]
                if inst_.name == 'barrier': # barrier
                    inst_tups.insert(insert_pos, inst_tup)
                    break
                else: # gate
                    if len(qargs_) == 1: # 1q gate
                        # Swap the indices and continue
                        if qargs_ == [qargs[0]]:
                            inst_tups[j] = (inst_, [qargs[1]], cargs_)
                        elif qargs_ == [qargs[1]]:
                            inst_tups[j] = (inst_, [qargs[0]], cargs_)
                    elif len(qargs_) == 2: # 2q gate but not SWAP gate
                        common_qargs = set(qargs).intersection(set(qargs_))
                        if len(common_qargs) > 0:
                            if push_through_2q and len(common_qargs) == 2:
                                # Swap the two-qubit gate and continue
                                inst_tups[j]  = (inst_, [qargs_[1], qargs_[0]], cargs_)
                            else:
                                # Insert here and exit the loop
                                inst_tups.insert(insert_pos, inst_tup)
                                break
                    else: # multi-qubit gate
                        # Insert here and exit the loop
                        inst_tups.insert(insert_pos, inst_tup)
                        break
                j += (-1) ** (direc == 'left') # j += 1 for right, j += -1 for left
            del inst_tups[delete_pos]

        # Remove the barriers and create new circuit
        inst_tups = inst_tups[1:-1]
        circ_new = create_circuit_from_inst_tups(inst_tups, qreg=qreg)
        return circ_new

    @staticmethod
    def combine_swap_gates(circ: QuantumCircuit) -> QuantumCircuit:
        """Combines adjacent SWAP gates."""
        inst_tups = circ.data.copy()
        n_inst_tups = len(inst_tups)
        inst_tup_pos = list(range(n_inst_tups))

        for i in range(n_inst_tups - 1):
            inst0, qargs0, _ = inst_tups[i]
            inst1, qargs1, _ = inst_tups[i + 1]
            if inst0.name == 'swap' and inst1.name == 'swap':
                common_qargs = set(qargs0).intersection(set(qargs1))
                if len(common_qargs) == 2:
                    inst_tup_pos.remove(i)
                    inst_tup_pos.remove(i + 1)

        inst_tups_new = [inst_tups[i] for i in inst_tup_pos]
        qreg = circ.qregs[0]
        circ_new = create_circuit_from_inst_tups(inst_tups_new, qreg=qreg)
        return circ_new

    def transpile_last_section(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpiles the last section of the circuit."""
        inst_tups = []
        while True:
            inst_tup = circ.data.pop()
            if inst_tup[0].name != 'barrier':
                inst_tups.insert(0, inst_tup)
            else:
                break

        qreg = circ.qregs[0]
        circ_last = create_circuit_from_inst_tups(inst_tups, qreg=qreg)
        circ_last = self.push_swap_gates(circ_last, direcs=['right'])
        # circ_last = transpile(circ_last, basis_gates=['u3', 'swap'])
        circ_last = self.transpile(circ_last)
        circ.barrier()
        circ += circ_last
        return circ
