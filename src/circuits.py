"""Functions for circuit construction"""

from typing import Tuple, Optional, Iterable, Union, Sequence, List
from functools import reduce
from cmath import polar

import numpy as np
from scipy.linalg import expm
from constants import HARTREE_TO_EV
from recompilation import apply_quimb_gates

import params

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Instruction, Barrier
from qiskit.extensions import UnitaryGate, CCXGate, SwapGate
from qiskit.quantum_info import SparsePauliOp

from utils import reverse_qubit_order, get_statevector, get_unitary

from recompilation import apply_quimb_gates


CircuitData = Iterable[Tuple[Instruction, List[int], Optional[List[int]]]]

class CircuitConstructor:
    """A class to construct circuits for calculating Green's Function."""

    def __init__(self,
                 ansatz: QuantumCircuit,
                 add_barriers: bool = True,
                 ccx_data: Optional[CircuitData] = None) -> None:
        """Creates a CircuitConstructor object.

        Args:
            ansatz: The ansatz quantum circuit containing the ground state.
            add_barriers: Whether to add barriers to the circuit.
            ccx_data: The circuit data for customized CCX gate.
        """
        self.ansatz = ansatz.copy()
        self.add_barriers = add_barriers
        if ccx_data is None:
            self.ccx_data = [(CCXGate(), [0, 1, 2], [])]
        else:
            self.ccx_data = ccx_data

            ccx_data_matrix = get_unitary(ccx_data)
            self.ccx_angle = polar(ccx_data_matrix[3, 7])[1]
            ccx_data_matrix[3, 7] /= np.exp(1j * self.ccx_angle)
            ccx_data_matrix[7, 3] /= np.exp(1j * self.ccx_angle)
            ccx_matrix = CCXGate().to_matrix()
            assert np.allclose(ccx_data_matrix, ccx_matrix)

    def build_diagonal_circuits(self,
                                a_op: List[SparsePauliOp]
                                ) -> QuantumCircuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            The creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the creation/annihilation operator appended.
        """
        # Copy the circuit with empty ancilla positions
        circ = copy_circuit_with_ancilla(self.ansatz, [0])

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

    def build_off_diagonal_circuits(self,
                                    a_op_m: List[SparsePauliOp],
                                    a_op_n: List[SparsePauliOp]
                                    ) -> QuantumCircuit:
        """Constructs the circuit to calculate off-diagonal transition amplitudes.

        Args:
            a_op_m: The first creation/annihilation operator of the circuit.
            a_op_n: The second creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the two creation/annihilation operators appended.
        """
        # Copy the circuit with empty ancilla positions
        circ = copy_circuit_with_ancilla(self.ansatz, [0, 1])

        # Apply the gates corresponding to the creation/annihilation terms
        if self.add_barriers: circ.barrier()
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

    def build_charge_diagonal(self, U_op: List[SparsePauliOp]) -> QuantumCircuit:
        """Constructs the circuit to calculate diagonal charge-charge transition elements."""
        circ = copy_circuit_with_ancilla(self.ansatz, [0, 1])
        
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

    def _apply_controlled_gate(self,
                              circ: QuantumCircuit,
                              op: SparsePauliOp,
                              ctrl: int = [1],
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
        assert set(ctrl).issubset({0, 1})
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
        
        # Apply single controlled gate
        if len(ctrl) == 1:
            if coeff != 1:
                circ.p(angle, 0)
            if set(list(label)) == {'I'}:
                pass
            else:
                circ.cz(0, n_anc)
        # Apply double controlled gate
        elif len(ctrl) == 2:
            if coeff != 1:
                circ.cp(angle, 0, 1)
            if set(list(label)) == {'I'}:
                if self.ccx_angle != 0:
                    circ.cp(self.ccx_angle, 0, 1)
            else:
                circ.h(n_anc)
                # Apply the central CCX gate
                if self.ccx_data is not None:
                    for inst_tup in self.ccx_data:
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

# TODO: Pass in n_gate_rounds and cache_options in a simpler way
def append_qpe_circuit(circ: QuantumCircuit,
                       hamiltonian_arr: np.ndarray,
                       ind_qpe: int,
                       recompiled: bool = False,
                       n_gate_rounds=None,
                       cache_options=None
                       ) -> QuantumCircuit:
    """Appends single-qubit QPE circuit to a given circuit.

    Args:
        circ: The quantum circuit on which the QPE circuit is to be appended.
        hamiltonian_arr: The Hamiltonian in array form.
        ind_qpe: Index of the QPE ancilla qubit.
        recompiled: Whether the controlled e^{iHt} gate is recompiled.

    Returns:
        A new quantum circuit on which the QPE circuit has been appended.
    """
    U_mat = expm(1j * hamiltonian_arr * HARTREE_TO_EV)
    n_sys = int(np.log2(U_mat.shape[0]))
    n_all = len(circ.qregs[0])
    n_anc = n_all - n_sys
    circ = copy_circuit_with_ancilla(circ, [ind_qpe])

    if recompiled:
        # Construct the controlled e^{iHt}
        cU_mat = np.kron(np.diag([1, 0]), np.eye(2 ** n_sys)) \
                + np.kron(np.diag([0, 1]), U_mat)
        cU_mat = np.kron(np.eye(2 ** n_anc), cU_mat)

        # Append single-qubit QPE circuit
        circ.barrier()
        circ.h(ind_qpe)
        statevector = reverse_qubit_order(get_statevector(circ))
        quimb_gates = recompile_with_statevector(
            statevector, cU_mat, n_gate_rounds=n_gate_rounds, cache_options=cache_options)
        circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
        circ.h(ind_qpe)
        circ.barrier()
    else:
        # Construct the controlled e^{iHt}
        cU_gate = UnitaryGate(U_mat).control(1)

        # Append single-qubit QPE circuit
        circ.barrier()
        circ.h(ind_qpe)
        circ.append(cU_gate, np.arange(n_sys + 1) + n_anc)
        circ.h(ind_qpe)
        circ.barrier()
    return circ

def copy_circuit_with_ancilla(circ: QuantumCircuit,
                              inds_anc: Sequence[int]) -> QuantumCircuit:
    """Copies a circuit with specific indices for ancillas.

    Args:
        circ: The quantum circuit to be copied.
        inds_anc: Indices of the ancilla qubits.

    Returns:
        The new quantum circuit with empty ancilla positions.
    """
    # Create a new circuit along with the quantum registers
    n_sys = circ.num_qubits
    n_anc = len(inds_anc)
    n_qubits = n_sys + n_anc
    inds_new = [i for i in range(n_qubits) if i not in inds_anc]
    qreg_new = QuantumRegister(n_qubits, name='q')
    circ_new = QuantumCircuit(qreg_new)

    # Copy instructions from the ansatz circuit to the new circuit
    for inst, qargs, cargs in circ.data:
        qargs = [inds_new[q._index] for q in qargs]
        circ_new.append(inst, qargs, cargs)
    return circ_new

def create_circuit_from_data(
        circ_data: CircuitData, 
        qreg: QuantumRegister = None,
        n_qubits: int = None
    ) -> QuantumCircuit:
    """Creates a circuit from circuit data.
    
    Args:
        circ_data: """
    n_qubits = 4
    if n_qubits is None:
        n_qubits = max([max([y.index for y in x[1]]) for x in circ_data]) + 1
    if qreg is None:
        qreg = QuantumRegister(n_qubits, name='q')
    circ = QuantumCircuit(qreg)
    for inst_tup in circ_data:
        try:
            circ.append(*inst_tup)
        except:
            inst, qargs, cargs = inst_tup
            qargs = [q._index for q in qargs]
            circ.append(inst, qargs, cargs)
    return circ

def transpile_across_barrier(circ: QuantumCircuit,
                             basis_gates: List[str] = None,
                             push: bool = False, ind = None
                             ) -> QuantumCircuit:
    """Transpiles a circuit across barriers."""
    if basis_gates is None:
        basis_gates = params.basis_gates

    qreg = circ.qregs[0]
    circ_data = circ.data.copy()
    circ_data_split = []
    circ_data_single = []

    # Split when encoutering a barrier
    for i, inst_tup in enumerate(circ_data):
        if inst_tup[0].name == 'barrier':
            circ_data_split.append(circ_data_single)
            circ_data_single = []
        elif i == len(circ_data) - 1:
            circ_data_single.append(inst_tup)
            circ_data_split.append(circ_data_single)
        else:
            circ_data_single.append(inst_tup)

    # Transpile except for three-qubit gate
    circ_new = QuantumCircuit(qreg)
    count = 0
    for i, circ_data_single in enumerate(circ_data_split):
        if len(circ_data_single) > 1:
            circ_single = create_circuit_from_data(circ_data_single, qreg=qreg)
            circ_single = transpile(circ_single, basis_gates=basis_gates)
            if i == 4: 
                # Swap positions of CPhase and U3
                # print(circ_single)
                # for j, inst_tup in enumerate(circ_single.data):
                #     print(j, inst_tup[0].name)
                tmp = circ_single.data[3]
                circ_single.data[3] = circ_single.data[4]
                circ_single.data[4] = tmp
            if push:
                # First round pushes do not push through two-qubit gates
                circ_single = push_swap_gates(
                    circ_single, direcs=params.swap_direcs_round1[ind][count].copy(), qreg=qreg)
                circ_single = combine_swap_gates(circ_single)

                # Second-round pushes push through two-qubit gates
                circ_single = push_swap_gates(
                    circ_single, direcs=params.swap_direcs_round2[ind][count].copy(), qreg=qreg, 
                    push_through_2q=True)

                # Final transpilation
                circ_single = transpile(circ_single, basis_gates=basis_gates)
                if i == 0 and ind == (0, 1): 
                   # The SWAP gates at the beginning and the end can be kept track of classically
                  del circ_single.data[2]
            circ_new += circ_single
            count += 1
        else:
            circ_new.barrier()
            circ_new.append(*circ_data_single[0])
            circ_new.barrier()
    return circ_new


def push_swap_gates(circ: QuantumCircuit, 
                    direcs: List[str] = [],
                    qreg: QuantumRegister = None,
                    push_through_2q: bool = False) -> QuantumCircuit:
    """Pushes the swap gates across single- and two-qubit gates.
    
    Args:
        circ: The quantum circuit on which SWAP gates are pushed.
        direcs: The directions on which each swap gate is pushed.
        qreg: The quantum register of the circuit.

    Returns:
        A new circuit on which SWAP gates are pushed.
    """

    assert set(direcs).issubset({'left', 'right', None})
    if direcs == []:
        return circ
    if qreg is None:
        qreg = circ.qregs[0]
    n_qubits = len(qreg)
    
    # Copy circ.data to two objects, circ_data_ref and circ_data.
    # circ_data_ref will not be modified while circ_data will be.
    # Also add barriers to first and last positions for easy processing.
    barr = Barrier(n_qubits)
    circ_data_ref = circ.data.copy()
    circ_data_ref.insert(0, (barr, qreg, []))
    circ_data_ref.append((barr, qreg, []))
    circ_data = circ_data_ref.copy()

    # Record SWAP gate positions
    swap_gate_pos = []
    swap_gate_ind = 0
    for i, inst_tup in enumerate(circ_data):
        if inst_tup[0].name == 'swap' and direcs[swap_gate_ind] is not None:
            swap_gate_pos.append(i)
            swap_gate_ind += 1
    # print('swap gate pos', swap_gate_pos)
    direcs = [d for d in direcs if d is not None]

    # Process direcs with swap_gate_pos
    direcs_single = []
    swap_gate_pos_direc = None
    folded_swap_gate_pos_direc = []
    for i, direc in enumerate(direcs):
        if len(direcs_single) == 0 or direc != direcs_single[-1]:
            # Append the old list and start building a new list
            direcs_single.append(direc)
            if swap_gate_pos_direc is not None:
                # Append the old list when it is not the first iteration
                folded_swap_gate_pos_direc.append(swap_gate_pos_direc)
            swap_gate_pos_direc = [swap_gate_pos[i]]
        else:
            swap_gate_pos_direc.append(swap_gate_pos[i])

        # If last element, append the list anyway
        if i == len(direcs) - 1:
            folded_swap_gate_pos_direc.append(swap_gate_pos_direc)

    # Reverse enumeration order for right pushing swap gates
    for i, direc in enumerate(direcs_single):
        if direc == 'right':
            folded_swap_gate_pos_direc[i] = reversed(folded_swap_gate_pos_direc[i])

    # Flatten the folded_swap_gate_pos_direc list
    flattened_swap_gate_pos_direc = [x for y in folded_swap_gate_pos_direc for x in y]
    # print('flattened swap gate pos direc =', flattened_swap_gate_pos_direc)

    for i in flattened_swap_gate_pos_direc:
        # print('i =', i)
        inst_tup = circ_data[i]
        qargs = inst_tup[1]

        try:
            direc = direcs.pop(0)
        except:
            direc = 'right'
        
        # Set up the data enumeration direction
        if direc == 'right':
            enumeration = zip(range(i + 1, len(circ_data)), circ_data[i + 1:])
        else:
            enumeration = zip(range(i - 1, -1, -1), reversed(circ_data[:i]))

        # Start sweeping to the left or the right.
        # The int(direc == 'left) term was due to whether to insert the new SWAP gate 
        # in front of or behind the barrier
        for j, inst_tup_ in enumeration:
            # print('j =', j)
            inst_, qargs_, cargs_ = inst_tup_
            if inst_.name == 'barrier':
                # Barrier. Insert here and exit the loop
                # print("Barrier")
                # print('inserting at position', j + int(direc == 'left'))
                circ_data.insert(j + int(direc == 'left'), inst_tup)
                break
            else:
                # Gate
                if len(qargs_) == 1:
                    # Single-qubit gate. Swap the indices and move on
                    # print("Single-qubit gate")
                    if qargs_ == [qargs[0]]:
                        circ_data[j] = (inst_, [qargs[1]], cargs_)
                    elif qargs_ == [qargs[1]]:
                        circ_data[j] = (inst_, [qargs[0]], cargs_)
                elif len(qargs_) == 2:
                    # Two-qubit gate but not SWAP gate
                    # print("Two-qubit gate")
                    common_qargs = set(qargs).intersection(set(qargs_))
                    if push_through_2q:
                        '''
                        if inst_.name != 'swap':
                            if len(common_qargs) == 1:
                                # Overlap on one qubit. Insert here and exit the loop
                                print("Overlap on one qubit")
                                common_qarg = list(common_qargs)[0]
                                qargs_copy = qargs.copy()
                                qargs__copy = qargs_.copy()
                                qargs_copy.remove(common_qarg)
                                qargs__copy.remove(common_qarg)
                                print(qargs_copy)
                                print(qargs__copy)
                                circ_data[j] = (inst_, [qargs_copy[0], qargs__copy[0]], [])
                                # circ_data.insert(j + int(direc == 'right'), inst_tup)
                            elif len(common_qargs) == 2:
                                # Overlap on both qubits. Swap the two-qubit gate and move on
                                print("Overlap on both qubits")
                                circ_data[j]  = (inst_, [qargs_[1], qargs_[0]], cargs_)
                        '''
                        if len(common_qargs) == 2:
                            # Overlap on both qubits. Swap the two-qubit gate and move on
                            circ_data[j]  = (inst_, [qargs_[1], qargs_[0]], cargs_)
                        else:
                            # Overlap on one qubit. Insert here and exit the loop
                            # print('inserting at position', j + int(direc == 'left'))
                            circ_data.insert(j + int(direc == 'left'), inst_tup)
                            break
                    else:
                        if len(common_qargs) > 0:
                            # print('inserting at position', j + int(direc == 'left'))
                            circ_data.insert(j + int(direc == 'left'), inst_tup)
                            break
                        
                else:
                    # n-qubit (n > 2) gate. Insert here and exit the loop
                    # print("n-qubit (n > 2) gate")
                    # print('inserting at position', j + int(direc == 'left'))
                    circ_data.insert(j + int(direc == 'left'), inst_tup)
                    break
            # print(create_circuit_from_data(circ_data, qreg=qreg))
        # print('deleting position', i + int(direc == 'left'))
        del circ_data[i + int(direc == 'left')]

    circ_data = circ_data[1:-1] # Remove the barriers
    circ_new = create_circuit_from_data(circ_data, qreg=qreg)
    return circ_new


def combine_swap_gates(circ: QuantumCircuit) -> QuantumCircuit:
    """Combines adjacent SWAP gates."""
    qreg = circ.qregs[0]
    circ_data = circ.data.copy()

    old_gate_pos = []
    new_gate_pos = []
    new_gates = []
    pos_shift = 0
    for i, inst_tup in enumerate(circ_data[:-1]):
        if inst_tup[0].name == 'swap' and circ_data[i + 1][0].name == 'swap':
            inst, qargs, cargs = inst_tup
            inst_, qargs_, cargs_ = circ_data[i + 1]
            common_qargs = set(qargs).intersection(set(qargs_))
            # if len(common_qargs) == 1:
            #     old_gate_pos.append(i + pos_shift)
            #     old_gate_pos.append(i + pos_shift)
            #     new_gate_pos.append(i + pos_shift)

            #     common_qarg = list(common_qargs)[0]
            #     qargs.remove(common_qarg)
            #     qargs_.remove(common_qarg)
            #     inst_tup_new = (SwapGate(), [qargs[0], qargs_[0]], [])
            #     new_gates.append(inst_tup_new)

            #     pos_shift -= 1
            if len(common_qargs) == 2:
                old_gate_pos.append(i + pos_shift)
                old_gate_pos.append(i + pos_shift)
                pos_shift -= 2

    for i in old_gate_pos:
        del circ_data[i]

    for i, inst_tup in zip(new_gate_pos, new_gates):
        circ_data.insert(i, inst_tup)

    circ_new = create_circuit_from_data(circ_data, qreg=qreg)
    return circ_new


def remove_swap_gates(circ: QuantumCircuit) -> QuantumCircuit:
    """Removes SWAP gates in a circuit."""
    qreg = circ.qregs[0]
    circ_data = []
    for inst_tup in circ.data:
        if inst_tup[0].name != 'swap':
            circ_data.append(inst_tup)
    circ_new = create_circuit_from_data(circ_data, qreg=qreg)
    return circ_new
