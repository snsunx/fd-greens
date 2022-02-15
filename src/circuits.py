"""Circuit construction module."""

from typing import Tuple, Optional, Iterable, Union, Sequence, List
from cmath import polar

import numpy as np
from permutation import Permutation

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Instruction, Qubit, Clbit, Barrier, Measure
from qiskit.extensions import CCXGate, XGate, HGate, RXGate, RZGate, CPhaseGate, CZGate, PhaseGate, SwapGate, UGate
from qiskit.quantum_info import SparsePauliOp

from params import CCZGate
from utils import (get_unitary, get_registers_in_inst_tups, create_circuit_from_inst_tups, 
                   remove_instructions)

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

        # Copy the ansatz circuit into the system qubit indices.
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [q._index + 1 for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)
        
        # Main part of the circuit to add the creation/annihilation operator.
        inst_tups += [(HGate(), [0], [])]
        if self.add_barriers: inst_tups += [(Barrier(3), range(3), [])] # Total # of qubits = 3
        inst_tups += self._get_controlled_gate_inst_tups(a_op[0], ctrl_states=[0])
        if self.add_barriers: inst_tups += [(Barrier(3), range(3), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op[1], ctrl_states=[1])
        if self.add_barriers: inst_tups += [(Barrier(3), range(3), [])]
        inst_tups += [(HGate(), [0], [])]

        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

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

        # Copy the ansatz circuit into the system qubit indices.
        inst_tups = self.ansatz.data.copy()
        for i, (inst, qargs, cargs) in enumerate(inst_tups):
            qargs = [self.sys[q._index] for q in qargs]
            inst_tups[i] = (inst, qargs, cargs)

        # Add the first creation/annihilation operator.
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[0], ctrl_states=[0, 0])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_m[1], ctrl_states=[1, 0])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]

        # Add the phase gate in the middle.
        inst_tups += [(RZGate(np.pi/4), [self.anc[1]], [])]
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]

        # Add the second creation/annihilation operator.
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[0], ctrl_states=[0, 1])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += self._get_controlled_gate_inst_tups(a_op_n[1], ctrl_states=[1, 1])
        if self.add_barriers: inst_tups += [(Barrier(4), range(4), [])]
        inst_tups += [(HGate(), [self.anc[0]], []), (HGate(), [self.anc[1]], [])]
        
        circ = create_circuit_from_inst_tups(inst_tups)
        return circ

    # TODO: Finish this function.
    def build_charge_diagonal(self, U_op: SparsePauliOp) -> QuantumCircuit:
        """Constructs the circuit to calculate diagonal charge-charge transition amplitudes."""
        '''
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
        '''

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

        # Find the indices to apply X gates (for control on 0), H gates (for Pauli X) 
        # and X(pi/2) gates (for Pauli Y) as well as the pivot (target qubit of the 
        # multi-controlled gate). The pivot selection is hardcoded: for single-controlled
        # gates, the pivot is the smallest index; for double-controlled gate, the pivot is
        # the largest index. 
        if len(ctrl_states) == 1: # Single-controlled gate, assume 0 is the control qubit
            cnot_inds = [i + 1 for i in range(len(label)) if label[i] != 'I']
            pivot = min(cnot_inds)
            cnot_inds.remove(pivot)
            x_inds = [0] if ctrl_states == [0] else []
            h_inds = [i + 1 for i in range(len(label)) if label[i] == 'X']
            rx_inds = [i + 1 for i in range(len(label)) if label[i] == 'Y']
        else: # Double-controlled gate, get ancilla and system qubits from attributes
            cnot_inds = [self.sys[i] for i in range(len(label)) if label[i] != 'I']
            pivot = max(cnot_inds)
            cnot_inds.remove(pivot)
            x_inds = [self.anc[i] for i in range(len(ctrl_states)) if ctrl_states[i] == 0]
            h_inds = [self.sys[i] for i in range(len(label)) if label[i] == 'X']
            rx_inds = [self.sys[i] for i in range(len(label)) if label[i] == 'Y']

        # Apply the controlled-Z gate.
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
            inst_tups = inst_tups + [(RXGate(-np.pi/2), [ind], [])]

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

    # Append rotation gates when tomographing on X or Y
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
    if n_qubits == 4:
        perms = [Permutation.cycle(2, 3)] # XXX
    else:
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
    The transpilation procedure depends on the circuit label.
    
    The basis gates are assumed to be X(pi/2), virtual Z, CS, CZ, and CXC (C-iX-C).
    In the '0' and '1' circuits, the ancilla is 0 and system qubits are 1 and 2.
    In the '01' circuit, the ancillae are 0 and 2 while the system qubits are 1 and 3.
    
    Args:
        circ: The quantum circuit to be transpiled.
        circ_label: The circuit label. '0', '1', or '01'.
        save_figs: Whether to save the circuit figures.
    
    Returns:
        The circuit after transpilation.
    """
    if circ_label == '0':
        circ_new = permute_qubits(circ, [1, 2], end=12)
    elif circ_label == '1':
        circ_new = circ
    elif circ_label == '01':
        if save_figs:
            fig = circ.draw('mpl')
            fig.savefig('figs/circ_untranspiled.png', bbox_inches='tight')
        # uni = get_unitary(circ)

        circ_new = permute_qubits(circ, [0, 3])
        circ_new = permute_qubits(circ_new, [0, 1], start=26)
        if save_figs:
            fig = circ_new.draw('mpl')
            fig.savefig('figs/circ_permuted.png', bbox_inches='tight')
        # uni1 = get_unitary(circ_new)

        circ_new = convert_ccz_to_cxc(circ_new)
        circ_new = convert_swap_to_cz(circ_new)
        circ_new = remove_instructions(circ_new, ['barrier'])
        circ_new = combine_1q_gates(circ_new)
        circ_new = combine_1q_gates(circ_new)
        circ_new = convert_1q_to_xpi2(circ_new)
        circ_new = combine_1q_gates(circ_new)
        
        # Some temporary checking statement
        # uni2 = get_unitary(circ_new)
        # vec1 = uni1[:, 0]
        # vec1n = vec1 / (vec1[0] / abs(vec1[0]))
        # vec2 = uni2[:, 0]
        # vec2n = vec2 / (vec2[0] / abs(vec2[0]))
        # print(np.allclose(vec1, vec2))
        # print(np.allclose(vec1n, vec2n))
        if save_figs:
            fig = circ_new.draw('mpl')
            fig.savefig('figs/circ_additional.png', bbox_inches='tight')
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
    inst_tups_new.insert(end, (SwapGate(), swap_inds, []))
    if start != 0:
        inst_tups_new.insert(start, (SwapGate(), swap_inds, []))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    return circ_new

def convert_ccz_to_cxc(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts CCZ gates to CXC gates with dressing H and X gates."""
    inst_tups = circ.data.copy()
    inst_tups_new = []
    for inst, qargs, cargs in inst_tups:
        if inst.name == 'c-unitary':
            inst_tups_new += [(XGate(), [qargs[0]], []), 
                              (HGate(), [qargs[1]], []),
                              (XGate(), [qargs[2]], []),
                              (CCXGate(ctrl_state='00'), [qargs[0], qargs[2], qargs[1]], []),
                              (XGate(), [qargs[0]], []), 
                              (HGate(), [qargs[1]], []), 
                              (XGate(), [qargs[2]], [])]
        else:
            inst_tups_new.append((inst, qargs, cargs))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    return circ_new

def convert_swap_to_cz(circ: QuantumCircuit) -> QuantumCircuit:
    """Converts SWAP gates to CZ and X(pi/2) gates except for the ones at the end."""
    inst_tups = circ.data.copy()
    inst_tups_new = []

    # Iterate from the end of the circuit, do not start converting SWAP gates to CZ gates
    # unless after encountering a non-SWAP gate.
    convert_swap = False
    for inst, qargs, cargs in inst_tups[::-1]:
        if inst.name == 'swap': # SWAP gate
            if convert_swap: # SWAP gate not at the end, convert
                inst_tups_new = [(RXGate(np.pi/2), [qargs[0]], []), 
                                 (RXGate(np.pi/2), [qargs[1]], []), 
                                 (CZGate(), [qargs[0], qargs[1]], [])] * 3 + inst_tups_new
            else: # SWAP gate at the end, do not convert
                inst_tups_new.insert(0, (inst, qargs, cargs))
        else: # Non-SWAP gate, set convert_swap to True
            inst_tups_new.insert(0, (inst, qargs, cargs))
            convert_swap = True

    circ_new = create_circuit_from_inst_tups(inst_tups_new)
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
        else:
            inst_tups_new.append((inst, qargs, cargs))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    return circ_new

def combine_1q_gates(circ: QuantumCircuit, qubits: Optional[Iterable[int]] = None
        ) -> QuantumCircuit:
    """Combines single-qubit gates on certain qubits."""
    inst_tups = circ.data.copy()
    if qubits is None:
        qreg, _ = get_registers_in_inst_tups(inst_tups)
        qubits = range(len(qreg))

    i_old = None
    inst_old = UGate(0, 0, 0) # Sentinel
    del_inds = []
    for q in qubits:
        for i, (inst, qargs, _) in enumerate(inst_tups):
            if q in [x._index for x in qargs]:
                if len(qargs) == 1: # 1q gate
                    # if verbose and inst.name == 'rx' and qargs[0]._index == 1: 
                    #     print(i, 'inst params', inst.params[0], len(qargs))
                    if inst.name == inst_old.name:
                        # if inst.name == 'rx': print(q, inst.params[0], inst_old.params[0])
                        if inst.name in ['h', 'x'] or (inst.name == 'rx' and
                            abs(inst.params[0] + inst_old.params[0]) < 1e-8): 
                        # update del_inds and start over
                            del_inds += [i, i_old]
                            i_old = None
                            inst_old = UGate(0, 0, 0)
                    else: # update old vars and continue
                        i_old = i
                        inst_old = inst.copy()
                else: # 2q gate, start over
                    i_old = None
                    inst_old = UGate(0, 0, 0)

    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds]
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    return circ_new