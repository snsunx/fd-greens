"""Circuit construction module."""

import os
from typing import Mapping, Tuple, Optional, Iterable, Union, Sequence, List
from cmath import polar
from itertools import combinations

import numpy as np
from permutation import Permutation

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Instruction, Qubit, Clbit, Barrier, Measure
from qiskit.extensions import (CCXGate, XGate, HGate, RXGate, RZGate, CPhaseGate, 
                               CZGate, PhaseGate, SwapGate, UGate, UnitaryGate, iSwapGate)
from qiskit.quantum_info import SparsePauliOp

from params import C0C0iXGate, CCZGate
from utils import (circuit_to_qasm_str, get_unitary, get_registers_in_inst_tups, create_circuit_from_inst_tups, 
                   remove_instructions, circuit_equal)

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]
InstructionTuples = List[InstructionTuple]

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
        # TODO: anc is not necessary. Can possibly deprecate this variable.
        self.anc = anc
        self.sys = [i for i in range(4) if i not in anc] # Total # of qubits = 4

        ccx_inst_tups_matrix = get_unitary(ccx_inst_tups)
        ccx_matrix = CCXGate().to_matrix()
        assert np.allclose(ccx_inst_tups_matrix, ccx_matrix)

        # TODO: Forgot why need this part. Probably can remove
        self.ccx_angle = 0
        # self.ccx_angle = polar(ccx_inst_tups_matrix[3, 7])[1]
        # ccx_inst_tups_matrix[3, 7] /= np.exp(1j * self.ccx_angle)
        # ccx_inst_tups_matrix[7, 3] /= np.exp(1j * self.ccx_angle)

    def build_eh_diagonal(self, a_op: SparsePauliOp) -> QuantumCircuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            The creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the creation/annihilation operator added.
        """
        assert len(a_op) == 2
        n_qubits = 3

        # Copy the ansatz circuit into the system qubit indices.
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [q._index + 1 for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)
        
        # Main part of the circuit to add the creation/annihilation operator.
        inst_tups += [(HGate(), [0], [])]
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op[0], ctrl_states=[0])
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op[1], ctrl_states=[1])
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += [(HGate(), [0], [])]

        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    build_diagonal = build_eh_diagonal

    def build_eh_off_diagonal(self,
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
        n_qubits = 4

        # Copy the ansatz circuit into the system qubit indices.
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [q._index + 2 for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)

        # Add the first creation/annihilation operator.
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[0], ctrl_states=[0, 0])
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[1], ctrl_states=[1, 0])
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]

        # Add the phase gate in the middle.
        inst_tups += [(RZGate(np.pi/4), [self.anc[1]], [])]
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]

        # Add the second creation/annihilation operator.
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[0], ctrl_states=[0, 1])
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[1], ctrl_states=[1, 1])
        if self.add_barriers: inst_tups += [(Barrier(n_qubits), range(n_qubits), [])]
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        
        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    build_off_diagonal = build_eh_off_diagonal
        
    def _get_controlled_gate_inst_tups(
            self,
            sparse_pauli_op: SparsePauliOp,
            ctrl_states: Sequence[int] = [1]) -> Sequence[InstructionTuple]:
        """Obtains the instruction tuple corresponding to a controlled-U gate.

        Args:
            sparse_pauli_op: The Pauli operator from which the controlled gate is constructed.
            ctrl_states: The qubit states on which the cU gate is controlled on.
        """
        assert len(sparse_pauli_op) == 1 # Only a single Pauli string
        assert set(ctrl_states).issubset({0, 1})
        n_anc = len(ctrl_states)
        assert n_anc in [1, 2] # Only single or double controlled gates
        coeff = sparse_pauli_op.coeffs[0]
        label = sparse_pauli_op.table.to_labels()[0][::-1]
        n_sys = len(label)
        inst_tups = []

        # Find the indices to apply X gates (for control on 0), H gates (for Pauli X)
        # and X(pi/2) gates (for Pauli Y) as well as the pivot (target qubit of the 
        # multi-controlled gate). The pivot selection is hardcoded: for single-controlled 
        # gates, the pivot is the smallest index; for double-controlled gate, the pivot is
        # the largest index.
        if set(label) != {'I'}:
            cnot_inds = [i + n_anc for i in range(n_sys) if label[i] != 'I']
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            x_inds = [i for i in range(n_anc) if ctrl_states[i] == 0]
            h_inds = [i + n_anc for i in range(n_sys) if label[i] == 'X']
            rx_inds = [i + n_anc for i in range(n_sys) if label[i] == 'Y']
        else: # Identity gate, only need to check the phase below.
            cnot_inds = []
            x_inds = []
            h_inds = []
            rx_inds = []

        # Apply the phase gate and controlled-Z gate.
        if len(ctrl_states) == 1:
            if coeff != 1:
                angle = polar(coeff)[1]
                inst_tups += [(PhaseGate(angle), [0], [])]
            if set(label) != {'I'}:
                inst_tups += [(CZGate(), [0, pivot], [])]
        elif len(ctrl_states) == 2:
            if coeff != 1:
                angle = polar(coeff)[1]
                inst_tups += [(CPhaseGate(angle), [self.anc[0], self.anc[1]], [])]
            if set(label) == {'I'}:
                if self.ccx_angle != 0:
                    inst_tups += [(CPhaseGate(self.ccx_angle), [self.anc[0], self.anc[1]], [])]
            else:
                inst_tups += [(CCZGate, [self.anc[0], self.anc[1], pivot], [])]

        # Wrap CNOT gates around for multi-qubit Pauli string. Note that these CNOTs are applied
        # in the reverse direction because the controlled gate is Z.
        for ind in cnot_inds:
            cx_inst_tups = [(HGate(), [pivot], []),
                            (CZGate(), [pivot, ind], []),
                            (HGate(), [pivot], [])]
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
            inst_tups = [(RXGate(np.pi/2), [ind], [])] + inst_tups
            inst_tups = inst_tups + [(RZGate(np.pi), [ind], []), 
                                     (RXGate(np.pi/2), [ind], []),
                                     (RZGate(np.pi), [ind], [])]

        return inst_tups

def append_tomography_gates(circ: QuantumCircuit, 
                            qubits: Iterable[QubitLike],
                            label: Tuple[str]) -> QuantumCircuit:
    """Appends tomography gates to a circuit.

    Args:
        circ: The circuit to which tomography gates are to be appended.
        qubits: The qubits to be tomographed.
        label: The tomography states label.
    
    Returns:
        A new circuit with tomography gates appended.
    """
    assert len(qubits) == len(label)
    inst_tups = circ.data.copy()
    inst_tups_swap = []
    perms = []

    # Split off the last few SWAP gates.
    while True:
        inst, qargs, cargs = inst_tups.pop()
        if inst.name == 'swap':
            inst_tups_swap.insert(0, (inst, qargs, cargs))
            qinds = [q._index for q in qargs]
            perms.append(Permutation.cycle(*[i + 1 for i in qinds]))
        else:
            inst_tups.append((inst, qargs, cargs))
            break

    # Append rotation gates when tomographing on X or Y.
    for q, s in zip(qubits, label):
        q_new = q + 1
        for perm in perms: q_new = perm(q_new)
        q_new -= 1
        if s == 'x':
            inst_tups += [(RZGate(np.pi/2), [q_new], []), 
                          (RXGate(np.pi/2), [q_new], []),
                          (RZGate(np.pi/2), [q_new], [])]
        elif s == 'y':
            inst_tups += [(RXGate(np.pi/2), [q_new], []),
                          (RZGate(np.pi/2), [q_new], [])]

    inst_tups += inst_tups_swap
    tomo_circ = create_circuit_from_inst_tups(inst_tups)
    return tomo_circ

# TODO: Deprecate this function.
def append_tomography_gates1(circ: QuantumCircuit, 
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

def append_measurement_gates(circ: QuantumCircuit) -> QuantumCircuit:
    """Appends measurement gates to a circuit.
    
    Args:
        The circuit to which measurement gates are to be appended.
        
    Returns:
        A new circuit with measurement gates appended.
    """
    qreg = circ.qregs[0]
    n_qubits = len(qreg)
    circ.add_register(ClassicalRegister(n_qubits))
    inst_tups = circ.data.copy()
    # if n_qubits == 4:
    #     perms = [Permutation.cycle(2, 3)] # XXX
    # else:
    perms = []

    # Split off the last few SWAP gates.
    while True:
        inst, qargs, cargs = inst_tups.pop()
        if inst.name == 'swap':
            qinds = [q._index for q in qargs]
            perms.append(Permutation.cycle(*[i + 1 for i in qinds]))
        else:
            inst_tups.append((inst, qargs, cargs))
            break
    
    # Append the measurement gates permuted by the SWAP gates.
    inst_tups += [(Barrier(n_qubits), qreg, [])]
    for c in range(n_qubits):
        q = c + 1
        for perm in perms: q = perm(q)
        q -= 1
        
        inst_tups += [(Measure(), [q], [c])]
    # circ.measure(range(n_qubits), range(n_qubits))

    circ_new = create_circuit_from_inst_tups(inst_tups)
    return circ_new

# TODO: Deprecate this function.
def append_measurement_gates1(circ: QuantumCircuit) -> QuantumCircuit:
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

def transpile_into_berkeley_gates(circ: QuantumCircuit,
                                  circ_label: str,
                                  save_figs: bool = True) -> QuantumCircuit:
    """Transpiles the circuit into the native gates on the Berkeley device.

    The transpilation procedure depends on the circuit label. The basis gates
    are assumed to be X(pi/2), virtual Z, CS, CZ, and C-iX-C.
    
    Args:
        circ: The quantum circuit to be transpiled.
        circ_label: The circuit label. '0', '1', or '01'.
        save_figs: Whether to save the circuit figures.
    
    Returns:
        The circuit after transpilation.
    """
    if save_figs:
        fig = circ.draw('mpl')
        fig.savefig(f'figs/circ{circ_label}_stage0.png', bbox_inches='tight')

    circ_new = remove_instructions(circ, ['barrier'])
        
    if circ_label == '0u':
        circ_new = permute_qubits(circ_new, [1, 2])
        pass
    elif circ_label == '0d':
        circ_new = permute_qubits(circ_new, [1, 2], end=17)
    elif circ_label == '1u':
        pass
    elif circ_label == '1d':
        pass
    elif circ_label == '01u':
        circ_new = permute_qubits(circ_new, [2, 3], start=-5)
    elif circ_label == '01d':
        circ_new = permute_qubits(circ_new, [2, 3], end=20)
        circ_new = convert_ccz_to_cixc(circ_new)
        circ_new = combine_1q_gates(circ_new)
        if save_figs:
            fig = circ_new.draw('mpl')
            fig.savefig(f'figs/circ{circ_label}_stage1.png', bbox_inches='tight')

        circ_new = transpile1(circ_new, {(0, 1): ['swap', 'czxcz']})[1]
        if save_figs:
            fig = circ_new.draw('mpl')
            fig.savefig(f'figs/circ{circ_label}_stage2.png', bbox_inches='tight')

        # circ_new = permute_qubits(circ_new, [0, 1], start=30, end=36)
        # circ_new = permute_qubits(circ_new, [0, 1], start=79, end=83)
        # circ_new = combine_2q_gates(circ_new)
        # if save_figs:
        #     fig = circ_new.draw('mpl')
        #     fig.savefig(f'figs/circ{circ_label}_stage2.png', bbox_inches='tight')
        # circ_new = special_transpilation(circ_new)
    
    # Convert 2q and 3q gates to native gates.
    circ_new = combine_1q_gates(circ_new)
    circ_new = combine_2q_gates(circ_new, [[0, 1], [2, 3]])

    # Convert 1q gates to native gates.
    circ_new = convert_swap_to_cz(circ_new, [(2, 3)])
    circ_new = combine_2q_gates(circ_new, [[2, 3]])

    circ_new = convert_1q_to_xpi2(circ_new)
    circ_new = combine_1q_gates(circ_new)

    if circ_label == '01d':
        circ_new = transpile1(circ_new, 
                                {(0,): ['xzpi2x', 'combz', 'xzpix', 'combz', '3xpi2', 'combz'], 
                                (1,): ['xzpi2x', 'combz', 'xzpix', 'combz', '3xpi2', 'combz'], 
                                (2,): ['xzpi2x', 'combz', 'xzpix', 'combz', '3xpi2', 'combz', 'xzpi2x', 'combz'], 
                                (3,): ['xzpi2x', 'combz', 'xzpix', 'combz', '3xpi2', 'combz']})[1]
    if save_figs:
        fig = circ_new.draw('mpl')
        fig.savefig(f'figs/circ{circ_label}_transpiled.png', bbox_inches='tight')
        
    return circ_new

def permute_qubits(circ: QuantumCircuit, 
                   swap_inds: Sequence[int], 
                   start: Optional[int] = None, 
                   end: Optional[int] = None) -> QuantumCircuit:
    """Permutes qubits in a circuit and adds SWAP gates.
    
    Args:
        circ: The quantum circuit to be transpiled.
        swap_inds: The indices to permute.
        start: The starting index of transpilation.
        end: The end index for transpilation.

    Returns:
        The circuit after transpilation by swapping indices.
    """
    if start is None: start = 0
    if end is None: end = len(circ)
    if start < 0: start += len(circ)
    if end < 0: end += len(circ)
    inst_tups = circ.data.copy()

    def conjugate(gate_inds, swap_inds):
        perm = Permutation.cycle(*[i + 1 for i in swap_inds])
        gate_inds = [i + 1 for i in gate_inds]
        gate_inds_new = [perm(i) for i in gate_inds]
        gate_inds_new = sorted([i - 1 for i in gate_inds_new])
        return gate_inds_new
    
    inst_tups_new = []
    for i, (inst, qargs, cargs) in enumerate(inst_tups):
        if i >= start and i < end:
            qargs = sorted([q._index for q in qargs])
            qargs = conjugate(qargs, swap_inds)
        inst_tups_new.append((inst, qargs, cargs))
    
    # Insert SWAP gates at the start and end indices. The SWAP gate 
    # at the start index is only inserted when it is not 0. 
    inst_tups_new.insert(end, (SwapGate(), swap_inds, []))
    if start != 0:
        inst_tups_new.insert(start, (SwapGate(), swap_inds, []))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new

def convert_ccz_to_cixc(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts CCZ gates to CXC gates with dressing H and X gates."""
    inst_tups = circ.data.copy()
    inst_tups_new = []

    count = 0
    for inst, qargs, cargs in inst_tups:
        if inst.name == 'c-unitary':
            # The double controlled unitary is the original CCZ gate used in building the circuits.
            # To make the decomposition, simply wrap X gates around q0 and q2 and wrap H gates
            # around q1 in the CXC gate double controlled on |0>.
            # inst_tups_new += [(XGate(), [qargs[0]], []), 
            #                   (HGate(), [qargs[1]], []),
            #                   (XGate(), [qargs[2]], []),
            #                   (CCXGate(ctrl_state='00'), [qargs[0], qargs[2], qargs[1]], []),
            #                   (XGate(), [qargs[0]], []), 
            #                   (HGate(), [qargs[1]], []), 
            #                   (XGate(), [qargs[2]], [])]

            cizc_inst_tups = [(XGate(), [qargs[0]], []), 
                              (HGate(), [qargs[1]], []),
                              (XGate(), [qargs[2]], []),
                              (C0C0iXGate, [qargs[0], qargs[2], qargs[1]], []),
                              (XGate(), [qargs[0]], []), 
                              (HGate(), [qargs[1]], []), 
                              (XGate(), [qargs[2]], [])]

            csdag_inst_tups_after = [(RXGate(np.pi/2), [qargs[0]], []),
                               (RXGate(np.pi/2), [qargs[1]], []),
                               (CZGate(), [qargs[0], qargs[1]], []),
                               (RXGate(np.pi/2), [qargs[0]], []),
                               (RXGate(np.pi/2), [qargs[1]], []),
                               (CZGate(), [qargs[0], qargs[1]], []),
                               (RXGate(np.pi/2), [qargs[0]], []),
                               (RXGate(np.pi/2), [qargs[1]], []),
                               (CPhaseGate(-np.pi/2), [qargs[1], qargs[2]], []),
                               (CZGate(), [qargs[0], qargs[1]], []),
                               (SwapGate(), [qargs[0], qargs[1]], [])]

            csdag_inst_tups_before = [(SwapGate(), [qargs[0], qargs[1]], []),
                                      (CZGate(), [qargs[0], qargs[1]], []),
                                      (CPhaseGate(-np.pi/2), [qargs[1], qargs[2]], []),
                                      (RXGate(np.pi/2), [qargs[0]], []),
                                      (RXGate(np.pi/2), [qargs[1]], []),
                                      (CZGate(), [qargs[0], qargs[1]], []),
                                      (RXGate(np.pi/2), [qargs[0]], []),
                                      (RXGate(np.pi/2), [qargs[1]], []),
                                      (CZGate(), [qargs[0], qargs[1]], []),
                                      (RXGate(np.pi/2), [qargs[0]], []),
                                      (RXGate(np.pi/2), [qargs[1]], [])]

            if count % 2 == 0:
                inst_tups_new += cizc_inst_tups
                inst_tups_new += csdag_inst_tups_after
            else:
                inst_tups_new += csdag_inst_tups_before
                inst_tups_new += cizc_inst_tups

            count += 1

        else:
            inst_tups_new.append((inst, qargs, cargs))

    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new

def convert_swap_to_cz(circ: QuantumCircuit, q_pairs: Sequence[Tuple[int, int]] = None) -> QuantumCircuit:
    """Converts SWAP gates to CZ and X(pi/2) gates except for the ones at the end."""
    inst_tups = circ.data.copy()
    inst_tups_new = []

    # Iterate from the end of the circuit, do not start converting SWAP gates to CZ gates
    # unless after encountering a non-SWAP gate. This is because SWAP gates at the end can
    # be kept track of classically.
    convert_swap = False
    for inst, qargs, cargs in inst_tups[::-1]:
        if inst.name == 'swap': # SWAP gate
            q_pair = tuple([x._index for x in qargs])
            q_pair_in = True
            if q_pairs is not None: q_pair_in = q_pair in q_pairs
            if convert_swap and q_pair_in: # SWAP gate not at the end, convert
                inst_tups_new = [(RXGate(np.pi/2), [qargs[0]], []), 
                                 (RXGate(np.pi/2), [qargs[1]], []), 
                                 (CZGate(), [qargs[0], qargs[1]], [])] * 3 + inst_tups_new
            else: # SWAP gate at the end, do not convert
                inst_tups_new.insert(0, (inst, qargs, cargs))
        else: # Non-SWAP gate, set convert_swap to True
            inst_tups_new.insert(0, (inst, qargs, cargs))
            convert_swap = True

    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new

def convert_1q_to_xpi2(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts single-qubit gates H, X and Ry(theta) to X(pi/2) and virtual Z gates."""
    inst_tups = circ.data.copy()
    inst_tups_new = []
    for inst, qargs, cargs in inst_tups:
        if inst.name == 'h':
            inst_tups_new += [(RZGate(np.pi/2), qargs, []), 
                              (RXGate(np.pi/2), qargs, []), 
                              (RZGate(np.pi/2), qargs, [])]
        elif inst.name == 'x':
            inst_tups_new += [(RXGate(np.pi/2), qargs, []), 
                              (RXGate(np.pi/2), qargs, [])]
        elif inst.name == 'ry':
            theta = inst.params[0]
            inst_tups_new += [(RZGate(-np.pi), qargs, []), 
                              (RXGate(np.pi/2), qargs, []), 
                              (RZGate((np.pi-theta) % (2*np.pi)), qargs, []),
                              (RXGate(np.pi/2), qargs, [])]
        elif inst.name == 'u3':
            theta, phi, lam = inst.params
            inst_tups_new += [(RZGate(lam-np.pi), qargs, []),
                              (RXGate(np.pi/2), qargs, []),
                              (RZGate(np.pi-theta), qargs, []),
                              (RXGate(np.pi/2), qargs, []),
                              (RZGate(phi), qargs, [])]
        else:
            inst_tups_new.append((inst, qargs, cargs))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new

def combine_1q_gates(circ: QuantumCircuit, 
                     qubits: Optional[Iterable[int]] = None
                     ) -> QuantumCircuit:
    """Combines single-qubit gates H, X and Rx to identities on given qubits."""
    inst_tups = circ.data.copy()
    if qubits is None:
        qreg, _ = get_registers_in_inst_tups(inst_tups)
        qubits = range(len(qreg))

    # for i, (inst, qargs, cargs) in enumerate(inst_tups[:10]):
    #     print(i, inst.name, qargs, cargs)
        
    del_inds = []
    for q in qubits:
        # Initialize i_old and inst_old. i_old is the previous index of the 1q gate we record, 
        # and inst_old is a sentinel that does not appear in the circuit.
        i_old = None
        inst_old = UGate(0, 0, 0) # Sentinel
        for i, (inst, qargs, _) in enumerate(inst_tups):
            if q in [x._index for x in qargs]:
                if len(qargs) == 1: # 1q gate
                    # if verbose and inst.name == 'rx' and qargs[0]._index == 1: 
                    #     print(i, 'inst params', inst.params[0], len(qargs))
                    if inst.name == inst_old.name:
                        # if inst.name == 'rx': print(q, inst.params[0], inst_old.params[0])
                        # Encountering a 1q gate the same as the previous gate. The two gates are
                        # deleted when they are H gates, X gates, or Rx(\theta) and Rx(-\theta).
                        if inst.name in ['h', 'x'] or (inst.name == 'rx' and
                            abs(inst.params[0] + inst_old.params[0]) < 1e-8): 
                            del_inds += [i_old, i]
                            i_old = None
                            inst_old = UGate(0, 0, 0)
                    else: 
                        # Encountering a 1q gate not the same as the previous gate.
                        # Update i_old and inst_old and continue the search.
                        i_old = i
                        inst_old = inst.copy()
                else: 
                    # Encountering a 2q gate. Start over the search by resetting i_old and inst_old.
                    i_old = None
                    inst_old = UGate(0, 0, 0)
    
    # Include only the gates that are not in del_inds and create the new circuit.
    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds]
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new

def special_transpilation(circ: QuantumCircuit) -> QuantumCircuit:
    """A special transpilation function for the 01d circuit. Should be deprecated."""
    count_x = 0
    count_cz = 0

    process_counts_x = [1, 3]
    del_counts_cz = [2, 3, 8, 9]

    insert_inds_z = []
    del_inds_cz = []

    inst_tups = circ.data.copy()
    for i, (inst, qargs, cargs) in enumerate(inst_tups):
        if inst.name == 'x' and qargs[0]._index == 1:
            if count_x in process_counts_x:
                insert_inds_z.append(i)
            count_x += 1
        elif inst.name == 'cz' and [q._index for q in qargs] == [0, 1]:
            if count_cz in del_counts_cz:
                del_inds_cz.append(i)
            count_cz += 1

    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds_cz]
    for i in insert_inds_z:
        inst_tups_new.insert(i, (RZGate(np.pi), [0], []))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    fig = circ_new.draw('mpl')
    fig.savefig(f'figs/circ01d_stage3.png', bbox_inches='tight')
    assert circuit_equal(circ, circ_new)
    return circ_new

def combine_2q_gates(circ: QuantumCircuit, 
                     qubit_pairs: Optional[Iterable[Tuple[int, int]]] = None
                     ) -> QuantumCircuit:    
    """Combines two CS(pi) i.e. CZ gates into identity on given qubits."""
    inst_tups = circ.data.copy()
    if qubit_pairs is None:
        qreg, _ = get_registers_in_inst_tups(inst_tups)
        qubits = range(len(qreg))
        qubit_pairs = list(combinations(qubits, 2))

    del_inds = []
    count = 0
    for q_pair in qubit_pairs:
        i_old = None
        inst_old = UGate(0, 0, 0)
        for i, (inst, qargs, _) in enumerate(inst_tups):
            qarg_inds = [x._index for x in qargs]
            if tuple(q_pair) == tuple(qarg_inds):
                if (inst.name == inst_old.name == 'cp' and
                    inst.params[0] == inst_old.params[0] == np.pi) or \
                    inst.name == inst_old.name == 'cz' or \
                    inst.name == inst_old.name == 'swap':
                    del_inds += [i_old, i]
                    i_old = None
                    inst_old = UGate(0, 0, 0)
                else:
                    i_old = i
                    inst_old = inst.copy()
            else:
                # Encountering a gate that is not a CS gate. Start over the search by 
                # resetting i_old and inst_old to the initial values.
                i_old = None
                inst_old = UGate(0, 0, 0)

    # Include only the gates that are not in del_inds. Insert CZ gates 
    # at the insert_inds and create the new circuit.
    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds]
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(circ, circ_new)
    return circ_new


def transpile1(circ: QuantumCircuit,
               qubits_trans: Mapping[Sequence[int], Sequence[str]]
               ) -> QuantumCircuit:
    # print('circ\n', circ)
    inst_tups = circ.data.copy()
    for qubits, trans_strs in qubits_trans.items():
        # Split the inst_tups of the main circuit into a subcircuit and a main circuit.
        # Also record the barrier locations which will be used in reconstructing the circuit.
        inst_tups_sub, inst_tups_main, barr_loc = split_subcircuit(inst_tups, qubits)
        # print(create_circuit_from_inst_tups(inst_tups_sub))
        # print(inst_tups_main)
        # inst_tups_main = [inst_tup for inst_tup in inst_tups_main if inst_tup[0] is not None]
        # print(create_circuit_from_inst_tups([inst_tup for inst_tup in inst_tups_main if inst_tup[0] is not None]))

        for trans_str in trans_strs:
            trans_func = transpilation_dict[trans_str]
            inst_tups_sub = trans_func(inst_tups_sub)
        inst_tups = merge_subcircuit(inst_tups_sub, inst_tups_main, barr_loc, qubits)
        # print(create_circuit_from_inst_tups(inst_tups.copy()))
    
    # print('circ\n', circ)
    circ_new = create_circuit_from_inst_tups(inst_tups)
    # print('circ_new\n', circ_new)
    assert circuit_equal(circ, circ_new)
    return circ, circ_new

def split_subcircuit(inst_tups: Sequence[InstructionTuple], qubits_subcirc: Sequence[int]
                    ) -> Tuple[List[InstructionTuple], List[InstructionTuple], List[int]]:
    """Split a subcircuit from a quantum circuit."""
    n_qubits = len(qubits_subcirc)
    map_qubits = lambda qubits_gate: [qubits_subcirc.index(q) for q in qubits_gate]

    # inst_tups = circ.data.copy()
    barr_loc = []
    inst_tups_sub = []
    inst_tups_main = []

    for i, (inst, qargs, cargs) in enumerate(inst_tups):
        if isinstance(qargs[0], int):
            qubits_gate = qargs
        else:
            qubits_gate = [q._index for q in qargs]

        # if qargs_ind == qubits:
        if set(qubits_gate).issubset(set(qubits_subcirc)): 
            # Gate qubits are a subset of subcircuit qubits. Append the instruction tuple to the
            # subcircuit. Append (None, None, None) to the main circuit, which will be handled
            # in the merge function.
            inst_tups_sub.append((inst, map_qubits(qubits_gate), cargs))
            inst_tups_main.append((None, None, None))
        elif set(qubits_gate).intersection(set(qubits_subcirc)):
            # Gate qubits intersect subcircuit qubits. Insert a barrier in the circuit and 
            # record the barrier location. Append the instruction tuple to the main circuit.
            barr_loc.append(i)
            inst_tups_sub.append((Barrier(n_qubits), range(n_qubits), []))
            inst_tups_main.append((inst, qargs, cargs))
        else:
            # Gate qubits do not overlap with subcircuit qubits. Only append the instruction tuple
            # to the main circuit.
            inst_tups_main.append((inst, qargs, cargs))
    
    return inst_tups_sub, inst_tups_main, barr_loc

def merge_subcircuit(inst_tups_sub: Sequence[InstructionTuple],
                     inst_tups_main: Sequence[InstructionTuple],
                     barr_loc: Sequence[int],
                     qubits: Sequence[int]) -> QuantumCircuit:
    print('barr_loc =', barr_loc)
    map_qubits = lambda x: [qubits[i] for i in x]
    inst_tups = []

    barr_inds = []
    inst_tups_sub_new = []
    barr_ind = -1
    for inst, qargs, cargs in inst_tups_sub:
        if inst.name != 'barrier':
            inst_tups_sub_new.append((inst, qargs, cargs))
            barr_inds.append(barr_ind)
        else:
            barr_ind = barr_loc.pop(0)

    print('barr_inds =', barr_inds)

    for i, (inst, qargs, cargs) in enumerate(inst_tups_main):
        if inst is not None:
            # If the instruction is not one of the subcircuit instructions, 
            # just append it to inst_tups.
            inst_tups.append((inst, qargs, cargs))
        else:
            # Look for instruction in inst_tups_sub.
            # print('barr_inds =', barr_inds)
            # barr_ind = barr_inds[0]
            if len(barr_inds) > 0 and i > barr_inds[0]:
                inst_, qargs_, cargs_ = inst_tups_sub_new.pop(0)
                inst_tups.append((inst_, map_qubits(qargs_), cargs_))
                barr_inds = barr_inds[1:]

    # circ = create_circuit_from_inst_tups(inst_tups)
    return inst_tups

def convert_xzpi2x(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts X(pi/2)Z(pi/2)X(pi/2) to Z(pi/2)X(pi/2)Z(pi/2) recursively on a single-qubit circuit."""
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    converted = False
    for inst, qargs, cargs in inst_tups:
        # Always append the instruction tuple to inst_tups_new. 
        # If transpilation is carried out, inst_tups_new is modified at the end.
        inst_tups_new.append((inst, qargs, cargs))

        if inst == RXGate(np.pi/2):
            # If the gate is X(pi/2), append to inst_tups_running if it contains 0 or 2 elements,
            # otherwise reset inst_tups_running to [X(pi/2)].
            if len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 1:
                inst_tups_running = [(inst, qargs, cargs)]
        elif inst.name == 'rz' and abs(inst.params[0]) == np.pi/2:
            # If the gate is Z(pi/2), append to inst_tups_running if it contains 1 element,
            # otherwise reset inst_tups_running to [].
            if len(inst_tups_running) == 1:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running = []
        else:
            # The gate is neither X(pi/2) nor Z(pi/2). Reset inst_tups_running to [].
            inst_tups_running = []

        # print('len(inst_tups_running) =', len(inst_tups_running))
        if len(inst_tups_running) == 3:
            # inst_tups_running is of the form [X(pi/2), Z(pi/2), X(pi/2)]. 
            # Remove the last 3 elements of inst_tups_new and append [Z(pi/2), X(pi/2), Z(pi/2)].
            inst_tups_new = inst_tups_new[:-3]
            z_angle = inst_tups_running[1][0].params[0]
            print(z_angle)
            inst_tups_new += [(RZGate(z_angle), [0], []),
                              (RXGate(np.pi/2), [0], []),
                              (RZGate(z_angle), [0], [])]
            
            # Set converted to True so that the function will be called again recursively.
            # Reset inst_tups_running to [] so that the chain of gates will be built up again.
            converted = True
            inst_tups_running = []

    
    print('converted =', converted)

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new
    
    # If converted is True, call the function again recursively. Otherwise return the circuit.
    if converted:
        circ_new = convert_xzpi2x(circ_new)
    return circ_new

def convert_xzpix(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts X(pi/2)Z(pi)X(pi/2) to Z(pi) recursively on a single-qubit circuit."""
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    converted = False
    for inst, qargs, cargs in inst_tups:
        # Always append the instruction tuple to inst_tups_new. 
        # If transpilation is carried out, inst_tups_new is modified at the end.
        inst_tups_new.append((inst, qargs, cargs))

        if inst == RXGate(np.pi/2):
            # If the gate is X(pi/2), append to inst_tups_running if it contains 0 or 2 elements,
            # otherwise reset inst_tups_running to [X(pi/2)].
            if len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 1:
                inst_tups_running = [(inst, qargs, cargs)]
        elif inst == RZGate(np.pi):
            # If the gate is Z(pi), append to inst_tups_running if it contains 1 element,
            # otherwise reset inst_tups_running to [].
            if len(inst_tups_running) == 1:
                inst_tups_running.append((inst, qargs, cargs))
            elif len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running = []
        else:
            # The gate is neither X(pi/2) nor Z(pi/2). Reset inst_tups_running to [].
            inst_tups_running = []

        # print('len(inst_tups_running) =', len(inst_tups_running))
        if len(inst_tups_running) == 3:
            # inst_tups_running is of the form [X(pi/2), Z(pi/2), X(pi/2)]. 
            # Remove the last 3 elements of inst_tups_new and append [Z(pi/2), X(pi/2), Z(pi/2)].
            inst_tups_new = inst_tups_new[:-3]
            # inst_tups_new += [(RZGate(np.pi/2), [0], []),
            #                   (RXGate(np.pi/2), [0], []),
            #                   (RZGate(np.pi/2), [0], [])]
            inst_tups_new.append((RZGate(np.pi), [0], []))

            # Set converted to True so that the function will be called again recursively.
            # Reset inst_tups_running to [] so that the chain of gates will be built up again.
            converted = True
            inst_tups_running = []

    
    print('converted =', converted)

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new
    
    # If converted is True, call the function again recursively. Otherwise return the circuit.
    if converted:
        circ_new = convert_xzpix(circ_new)
    return circ_new

def convert_3xpi2(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts X(pi/2)X(pi/2)X(pi/2) to Z(pi)X(pi/2)Z(pi) recursively on a single-qubit circuit."""
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    converted = False
    for inst, qargs, cargs in inst_tups:
        # Always append the instruction tuple to inst_tups_new. 
        # If transpilation is carried out, inst_tups_new is modified at the end.
        inst_tups_new.append((inst, qargs, cargs))

        if inst == RXGate(np.pi/2):
            # Append to inst_tups_running if the gate is X(pi/2).
            inst_tups_running.append((inst, qargs, cargs))
        else:
            # The gate is not X(pi/2). Reset inst_tups_running to [].
            inst_tups_running = []

        # print('len(inst_tups_running) =', len(inst_tups_running))
        if len(inst_tups_running) == 3:
            # inst_tups_running is of the form [X(pi/2), Z(pi/2), X(pi/2)]. 
            # Remove the last 3 elements of inst_tups_new and append [Z(pi/2), X(pi/2), Z(pi/2)].
            inst_tups_new = inst_tups_new[:-3]
            inst_tups_new += [(RZGate(np.pi), [0], []),
                              (RXGate(np.pi/2), [0], []),
                              (RZGate(np.pi), [0], [])]

            # Set converted to True so that the function will be called again recursively.
            # Reset inst_tups_running to [] so that the chain of gates will be built up again.
            converted = True
            inst_tups_running = []

    
    print('converted =', converted)

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new
    
    # If converted is True, call the function again recursively. Otherwise return the circuit.
    if converted:
        circ_new = convert_xzpix(circ_new)
    return circ_new

def combine_z_gates(circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
                    ) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    """Combines Z gates recursively on a single-qubit circuit."""
    # assert len(circ.qregs[0]) == 1
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    z_angles = []

    # combined = False
    inst_tups += [(Barrier(1), [0], [])] # Append setinel
    for inst, qargs, cargs in inst_tups:
        if inst.name == 'rz':
            z_angles.append(inst.params[0])
        else:
            if len(z_angles) > 0:
                angle = sum(z_angles) % (2*np.pi)
                if angle > np.pi: angle -= 2*np.pi # [-pi, pi)
                if abs(angle) > 1e-8:
                    inst_tups_new.append((RZGate(angle), [0], []))
                z_angles = []
            inst_tups_new.append((inst, qargs, cargs))
    inst_tups_new = inst_tups_new[:-1] # Remove setinel

    # Create the new circuit and check if the two circuits are equivalent.
    # circ_new = create_circuit_from_inst_tups(inst_tups_new)
    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)

    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new
    return circ_new

def wrap_swap_gates(circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
                    ) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:

    # assert len(circ.qregs[0]) == 2
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else:
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    converted = False
    status = 0
    for inst, qargs, cargs in inst_tups:
        inst_tups_new.append((inst, qargs, cargs))

        if inst.name == 'swap':
            inst_tups_running.append((inst, qargs, cargs))
            qargs_2q = qargs.copy()
            status += 1 # 0 to 1 or 1 to 2
        else:
            if status == 1:
                if len(qargs) == 1:
                    inst_tups_running.append((inst, list(set(qargs_2q).difference(set(qargs))), cargs))
                if len(qargs) == 2:
                    if inst.name == 'barrier':
                        status = 0
                        inst_tups_running = []
                    else:
                        inst_tups_running.append((inst, list(reversed(qargs)), cargs))

        # print('status =', status)
        if status == 2:
            # print(len(inst_tups_running))
            # print(inst_tups_running)
            inst_tups_new = inst_tups_new[:-len(inst_tups_running)]
            inst_tups_new += inst_tups_running[1:-1]
            converted = True
            inst_tups_running = []
            status = 0

    # print(create_circuit_from_inst_tups(inst_tups))
    # print(create_circuit_from_inst_tups(inst_tups_new))

    assert circuit_equal(inst_tups, inst_tups_new, init_state_0=False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new

    if converted:
        circ_new = wrap_swap_gates(circ_new)
    return circ_new

def convert_czxcz(circ: Union[QuantumCircuit, Sequence[InstructionTuple]]
                 ) -> Union[QuantumCircuit, Sequence[InstructionTuple]]:
    if isinstance(circ, QuantumCircuit):
        inst_tups = circ.data.copy()
    else: # Already instruction tuples
        inst_tups = circ.copy()
    inst_tups_new = []
    inst_tups_running = []

    converted = False
    for inst, qargs, cargs in inst_tups:
        inst_tups_new.append((inst, qargs, cargs))
        if inst.name == 'cz':
            qargs_2q = qargs.copy()
            if len(inst_tups_running) == 0 or len(inst_tups_running) == 2:
                inst_tups_running.append((inst, qargs, cargs))
            else:
                assert len(inst_tups_running) == 1
                inst_tups_running = [(inst, qargs, cargs)]
        
        elif inst.name == 'x':
            qargs_1q = qargs.copy()
            if len(inst_tups_running) == 1:
                inst_tups_running.append((inst, qargs, cargs))
            else:
                assert len(inst_tups_running) == 0 or len(inst_tups_running) == 2
                inst_tups_running = []

        if len(inst_tups_running) == 3:
            inst_tups_new = inst_tups_new[:-3]
            inst_tups_new += [(RZGate(np.pi), list(set(qargs_2q).difference(set(qargs_1q))), []),
                              (XGate(), qargs_1q, [])]

            converted = True
            inst_tups_running = []

    assert circuit_equal(inst_tups, inst_tups_new, False)
    if isinstance(circ, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new)
    else:
        circ_new = inst_tups_new

    return circ_new

transpilation_dict = {
    'xzpi2x': convert_xzpi2x,
    'combz': combine_z_gates,
    'xzpix': convert_xzpix,
    '3xpi2': convert_3xpi2,
    'swap': wrap_swap_gates,
    'czxcz': convert_czxcz}
