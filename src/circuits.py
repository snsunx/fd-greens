from typing import Tuple, Optional, Iterable
from cmath import polar

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from openfermion.ops import PolynomialTensor
from openfermion.transforms import get_fermion_operator, jordan_wigner
from tools import get_pauli_tuple, pauli_label_to_tuple
from qiskit.quantum_info import SparsePauliOp

# Change Term to PauliTuple
PauliTuple = Tuple[Tuple[str, int]]

def build_diagonal_circuits(ansatz: QuantumCircuit,
                            a_op: SparsePauliOp,
                            with_qpe: bool = True,
                            add_barriers: bool = True,
                            measure: bool = False) -> QuantumCircuit:
    """Constructs the circuit to calculate a diagonal transition amplitude.
    
    Args:
        ansatz: The ansatz quantum circuit corresponding to the ground state.
        tup_xy: The creation/annihilation operator of the circuit in tuple form.
        with_qpe: Whether an additional qubit is added for QPE.
        add_barriers: Whether barriers are added to the circuit.
        measure: Whether the ancilla qubits are measured.

    Returns:
        A QuantumCircuit with the diagonal Pauli string appended.
    """
    # Create a new circuit along with the quantum registers
    n_qubits = ansatz.num_qubits
    n_anc = 2 if with_qpe else 1
    qreg = QuantumRegister(n_qubits + n_anc)
    creg = ClassicalRegister(n_anc)
    circ = QuantumCircuit(qreg, creg)

    # Copy instructions from the ansatz circuit to the new circuit
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + n_anc for qubit in qargs]
        circ.append(inst, qargs, cargs)

    # Apply the gates corresponding to the creation/annihilation terms
    if add_barriers: circ.barrier()
    circ.h(0)
    if add_barriers: circ.barrier()
    apply_multicontrolled_gate(circ, a_op[0], ctrl=[0], offset=n_anc)
    if add_barriers: circ.barrier()
    apply_multicontrolled_gate(circ, a_op[1], ctrl=[1], offset=n_anc)
    if add_barriers: circ.barrier()
    circ.h(0)
    if add_barriers: circ.barrier()

    if measure:
        circ.measure(0, 0)

    print(circ)
    
    return circ

def build_off_diagonal_circuits(ansatz: QuantumCircuit,
                                a_op_m: SparsePauliOp,
                                a_op_n: SparsePauliOp,
                                with_qpe: bool = True,
                                add_barriers: bool = True,
                                measure: bool = False) -> QuantumCircuit:
    """Constructs the circuit to calculate off-diagonal transition amplitudes.
    
    Args:
        ansatz: The ansatz quantum circuit corresponding to the ground state.
        tup_xy_left: The left creation/annihilation operator of the circuit 
            in tuple form.
        tup_xy_right: The right creation/annihilation operator of the circuit
            in tuple form.
        with_qpe: Whether an additional qubit is added for QPE.
        add_barriers: Whether barriers are added to the circuit.
        measure: Whether the ancilla qubits are measured.

    Returns:
        A QuantumCircuit with the off-diagonal Pauli string appended.
    """
    # Create a new circuit along with the quantum registers
    n_qubits = ansatz.num_qubits
    n_anc = 3 if with_qpe else 2
    qreg = QuantumRegister(n_qubits + n_anc)
    creg = ClassicalRegister(n_anc)
    circ = QuantumCircuit(qreg, creg)

    # Copy instructions from the ansatz circuit to the new circuit
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + n_anc for qubit in qargs]
        circ.append(inst, qargs, cargs)

    # Apply the gates corresponding to the creation/annihilation terms
    if add_barriers: circ.barrier()
    circ.h([0, 1])
    if add_barriers: circ.barrier()
    apply_multicontrolled_gate(circ, a_op_m[0], ctrl=(0, 0), offset=n_anc)
    if add_barriers: circ.barrier()
    apply_multicontrolled_gate(circ, a_op_m[1], ctrl=(1, 0), offset=n_anc)
    if add_barriers: circ.barrier()
    circ.rz(np.pi / 4, 1)
    if add_barriers: circ.barrier()
    apply_multicontrolled_gate(circ, a_op_n[0], ctrl=(0, 1), offset=n_anc)
    if add_barriers: circ.barrier()
    apply_multicontrolled_gate(circ, a_op_n[1], ctrl=(1, 1), offset=n_anc)
    if add_barriers: circ.barrier()
    circ.h([0, 1])

    if measure:
        circ.measure([0, 1], [0, 1])

    return circ

def apply_multicontrolled_gate(
        circ: QuantumCircuit, 
        op: SparsePauliOp,
        ctrl: int = [1], 
        offset: int = 1
    ) -> None:
    """Applies a controlled-U gate to a quantum circuit.
    
    Args:
        circ: The quantum circuit to which the controlled U gate is appended.
        term: A tuple specifying the Pauli string corresponding to 
            the creation/annihilation operator, e.g. Z0Z1X2 is specified as 
            (('Z', 0), ('Z', 1), ('X', 2)).
        ctrl: An integer indicating the qubit state on which the controlled-U
            gate is controlled on. Must be 0 or 1.
        offset: An integer indicating the number of qubits skipped when 
            applying the controlled-U gate.
    """
    assert set(ctrl).issubset({0, 1})
    assert len(op.coeffs) == 1
    coeff = op.coeffs[0]
    label = op.table.to_labels()[0]
    if coeff == 1 and set(list(label)) == {'I'}:
        return
    amp, angle = polar(coeff)
    assert amp == 1

    ind_max = len(label) - 1
    label_tmp = label
    for i in range(len(label)):
        if label_tmp[0] == 'I':
            label_tmp = label_tmp[1:]
            ind_max -= 1

    # Prepend X gates for control on 0
    for i in range(len(ctrl)):
        if ctrl[i] == 0:
            circ.x(i)

    # Prepend rotation gates for Pauli X and Y
    for i, c in enumerate(label[::-1]):
        if c == 'X':
            circ.h(i + offset)
        elif c == 'Y':
            circ.rx(np.pi / 2, i + offset)
    
    # Implement multicontrolled all-Z gate
    for i in range(ind_max + offset, offset, -1):
        circ.cx(i, i - 1)
    if len(ctrl) == 1:
        if coeff != 1:
            circ.p(angle, 0)
        circ.cz(0, offset)
    elif len(ctrl) == 2:
        if coeff != 1:
            circ.cp(angle, 0, 1)
        circ.h(offset)
        circ.ccx(0, 1, offset)
        circ.h(offset)
    else:
        raise ValueError("Control on more than two qubits is not implemented")
    for i in range(offset, ind_max + offset):
        circ.cx(i + 1, i)

    # Append rotation gates for Pauli X and Y
    for i, c in enumerate(label[::-1]):
        if c == 'X':
            circ.h(i + offset)
        elif c == 'Y':
            circ.rx(-np.pi / 2, i + offset)
    
    # Append X gates for control on 0
    for i in range(len(ctrl)):
        if ctrl[i] == 0:
            circ.x(i)