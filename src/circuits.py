"""Functions for circuit construction"""

from typing import Tuple, Optional, Iterable, Union, Sequence, List
from functools import reduce
from cmath import polar

import numpy as np
from scipy.linalg import expm
from constants import HARTREE_TO_EV

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction
from qiskit.extensions import UnitaryGate, CCXGate

from openfermion.ops import PolynomialTensor
from openfermion.transforms import get_fermion_operator, jordan_wigner
from qiskit.quantum_info import SparsePauliOp

from qiskit.extensions import CPhaseGate, HGate, SwapGate
from qiskit.quantum_info import OneQubitEulerDecomposer

from utils import data_to_circuit, reverse_qubit_order, get_statevector, get_unitary

from recompilation import apply_quimb_gates

import params


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
        self.ccx_data = ccx_data

        ccx_data_matrix = get_unitary(ccx_data)
        self.ccx_angle = polar(ccx_data_matrix[3, 7])[1]
        ccx_data_matrix[3, 7] /= np.exp(1j * self.ccx_angle)
        ccx_data_matrix[7, 3] /= np.exp(1j * self.ccx_angle)
        ccx_matrix = CCXGate().to_matrix()
        assert np.allclose(ccx_data_matrix, ccx_matrix)

    def build_diagonal_circuits(self,
                                a_op: SparsePauliOp
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
        The new quantum circuit with the empty ancilla positions.
    """
    # Create a new circuit along with the quantum registers
    n_sys = circ.num_qubits
    n_anc = len(inds_anc)
    n_qubits = n_sys + n_anc
    inds_new = [i for i in range(n_qubits) if i not in inds_anc]
    qreg_new = QuantumRegister(n_qubits)
    circ_new = QuantumCircuit(qreg_new)

    # Copy instructions from the ansatz circuit to the new circuit
    for inst, qargs, cargs in circ.data:
        qargs = [inds_new[q._index] for q in qargs]
        circ_new.append(inst, qargs, cargs)
    return circ_new

def create_circuit_from_data(circ_data, qreg=None, n_qubits=None):
    """Creates a circuit from circuit data."""
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

def transpile_across_barrier(circ, basis_gates=None, push=False):
    """Transpiles a circuit across barriers."""

    if basis_gates is None:
        basis_gates = params.basis_gates

    qreg = circ.qregs[0]
    circ_data = circ.data.copy()
    circ_data_split = []
    circ_data_single = []

    # Split when encoutering a barrier
    len_circ_data = len(circ_data)
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
    for circ_data_single in circ_data_split:
        if len(circ_data_single) > 1:
            circ_single = create_circuit_from_data(circ_data_single, qreg=qreg)
            circ_single = transpile(circ_single, basis_gates=basis_gates)
            print(circ_single)
            if push:
                for _ in range(3):
                    circ_single = push_swap_gates(circ_single, direcs=params.swap_direcs[count].copy(), qreg=qreg)
            print(circ_single)       
            circ_new += circ_single
            count += 1
        else:
            # circ_new.barrier()
            circ_new.append(*circ_data_single[0])
            # circ_new.barrier()
    return circ_new


def push_swap_gates(circ, direcs=[], qreg=None):
    """Pushes the swap gates across single- and two-qubit gates."""
    from qiskit.circuit import Barrier
    if qreg is None:
        qreg = circ.qregs[0]
    n_qubits = len(qreg)
    
    barr = Barrier(n_qubits)
    circ_data_ref = circ.data.copy()
    circ_data_ref.insert(0, (barr, qreg, []))
    circ_data_ref.append((barr, qreg, []))
    circ_data = circ_data_ref.copy()

    for i, inst_tup in enumerate(circ_data_ref):
        if inst_tup[0].name == 'swap':
            print('i', i)
            inst, qargs, cargs = inst_tup

            try:
                direc = direcs.pop(0)
            except:
                direc = 'right'
            
            if direc == 'right':
                enumeration = zip(range(i + 1, len(circ_data_ref)), circ_data_ref[i + 1:])
            else:
                enumeration = zip(range(i - 1, -1, -1), reversed(circ_data_ref[:i]))

            for j, inst_tup_ in enumeration:
                print('j', j)
                inst_, qargs_, cargs_ = inst_tup_
                print(inst_.name)
                if inst_.name == 'barrier':
                    # Barrier. Insert here and exit the loop
                    print("Barrier")
                    circ_data.insert(j + int(direc == 'left'), inst_tup)
                    break
                else:
                    # Gate
                    if len(qargs_) == 1:
                        # Single-qubit gate. Swap the indices and move on
                        print("Single-qubit gate")
                        if qargs_ == [qargs[0]]:
                            circ_data[j] = (inst_, [qargs[1]], cargs_)
                        elif qargs_ == [qargs[1]]:
                            circ_data[j] = (inst_, [qargs[0]], cargs_)
                    elif len(qargs_) == 2:
                        # Two-qubit gate
                        print("Two-qubit gate")
                        common_qargs = set(qargs).intersection(set(qargs_))
                        if len(common_qargs) == 1:
                            # Overlap on one qubit. Insert here and exit the loop
                            print("Overlap on one qubit")
                            circ_data.insert(j + int(direc == 'left'), inst_tup)
                            break
                        elif len(common_qargs) == 2:
                            # Overlap on both qubits. Swap the two-qubit gate and move on
                            print("Overlap on both qubits")
                            circ_data[j]  = (inst_, [qargs_[1], qargs_[0]], cargs_)
                    else:
                        # n-qubit (n > 2) gate. Insert here and exit the loop
                        print("n-qubit (n > 2) gate")
                        circ_data.insert(j + int(direc == 'left'), inst_tup)
                        break
                print(create_circuit_from_data(circ_data, qreg=qreg))
            del circ_data[i + int(direc == 'left')]

    circ_data = circ_data[1:-1] # Remove the barriers
    circ_new = create_circuit_from_data(circ_data, qreg=qreg)
    return circ_new

'''
class SingleQubitGateChain:
    def __init__(self, *, qubit, gates) -> None:
        self.qubit = qubit
        self.gates = gates
        self.n_gates = len(gates)

    @property
    def combined_gate(self):
        """Returns the U3 gate from combining a chain of single-qubit gates."""
        matrix = reduce(np.dot, [gate.to_matrix() for gate in self.gates[::-1]])
        decomposer = OneQubitEulerDecomposer()
        gate = decomposer(matrix).data[0][0]
        return gate

    @classmethod
    def combine_single_qubit_gates(cls, circ: QuantumCircuit) -> QuantumCircuit:
        """Combines each single-qubit gate chain to a single U3 gate on a circuit.

        Args:
            The circuit on which all chains of single-qubit gates are to be combined.

        Returns:
            A new circuit after all single-qubit gates have been combined.
        """
        gate_1q_chains = []
        gates_2q = []

        insert_inds = []

        i = 0
        while len(circ.data) != 0:
            print('i =', i)
            inst, qargs, cargs = circ.data[i]
            subtract_1 = False
            print(circ)
            if len(qargs) == 1:
                insert_inds.append(i)
                delete_inds = [i]
                gates = [inst]
                qubit = qargs[0]
                for j, (inst, qargs, cargs) in enumerate(circ.data[i+1:]):
                    if len(qargs) == 1 and qargs[0] == qubit:
                        # Append to single-qubit gate chain
                        gates.append(inst)
                        delete_inds.append(j + i + 1)
                    elif len(qargs) > 1 and qubit in qargs:
                        # Stop appending
                        i += 1
                        break
                    elif len(qargs) == 1 and qargs[0] != qubit:
                        subtract_1 = True
                    else:
                        # Ignore
                        continue

                # Delete the gates
                count = 0
                for k in delete_inds:
                    del circ.data[k - count]
                    count += 1


                # Append single-qubit gate chain
                gate_chain = cls(qubit=qubit, gates=gates)
                gate_chains.append(gate_chain)


        # Adjust insertion indices
        insert_inds = [ind + k for k, ind in enumerate(insert_inds)]
        for i, gate_chain in enumerate(gate_chains):
            circ.data.insert(insert_inds[i], (gate_chain.combined_gate, [gate_chain.qubit], []))
        return circ
'''