import numpy as np
from qiskit import *
from openfermion.ops import PolynomialTensor, QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from qiskit.extensions import UnitaryGate

def get_diagonal_circuits(ansatz, ind, measure=False):
    """Returns the quantum circuits to calculate diagonal elements of the 
    Green's functions."""
    # Create a new circuit with the ancilla as qubit 0
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 1)
    creg_anc = ClassicalRegister(1)
    creg_sys = ClassicalRegister(n_qubits) 
    circ = QuantumCircuit(qreg, creg_anc, creg_sys)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 1 for qubit in qargs]
        circ.append(inst, qargs, cargs)
    
    arr = np.zeros((n_qubits,))
    arr[ind] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms = list(qubit_op.terms)

    circ.barrier()
    circ.h(0)
    circ.barrier()
    apply_cU(circ, terms[0], ctrl=0)
    circ.barrier()
    apply_cU(circ, terms[1], ctrl=1)
    circ.barrier()
    circ.h(0)
    if measure:
        circ.measure(qreg[0], creg_anc)
    # TODO: Implement phase estimation on the system qubits

    return circ

build_diagonal_circuits = get_diagonal_circuits

def build_diagonal_circuits_with_qpe(ansatz, ind, measure=True):
    """Builds the quantum circuits to calculate diagonal elements of the 
    Green's function."""
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 2)
    creg = ClassicalRegister(2)
    circ = QuantumCircuit(qreg, creg)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 2 for qubit in qargs]
        circ.append(inst, qargs, cargs)

    arr = np.zeros((n_qubits,))
    arr[ind] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms = list(qubit_op.terms)

    circ.barrier()
    circ.h(0)
    circ.barrier()
    apply_cU(circ, terms[0], ctrl=0, offset=2)
    circ.barrier()
    apply_cU(circ, terms[1], ctrl=1, offset=2)
    circ.barrier()
    circ.h(0)

    circ.measure(0, 0)
    
    return circ


# XXX: Deprecated.
def get_diagonal_circuits1(ansatz, ind, measure=True):
    """Returns the quantum circuits to calculate diagonal elements of the 
    Green's functions."""
    # Create a new circuit with the ancilla as qubit 0
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 1)
    creg_anc = ClassicalRegister(1)
    creg_sys = ClassicalRegister(n_qubits) 
    circ = QuantumCircuit(qreg, creg_anc, creg_sys)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 1 for qubit in qargs]
        circ.append(inst, qargs, cargs)
    
    shape_expanded = (2,) * (ind + 1) * 2
    transpose_inds = np.hstack((np.flip(np.arange(ind + 1)), 
                                np.flip(np.arange(ind + 1) + ind + 1)))
    shape_compact = (2 ** (ind + 1), 2 ** (ind + 1))

    arr = np.zeros((n_qubits,))
    arr[ind] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    # TODO: The following is one way to construct the U0 and U1 gates
    for key, val in qubit_op.terms.items():
        if np.isreal(val):
            U0_op = 2 * val * QubitOperator(key)
            U0_mat = np.asarray(get_sparse_operator(U0_op).todense())
            U0_mat = U0_mat.reshape(*shape_expanded)
            U0_mat = U0_mat.transpose(*transpose_inds)
            U0_mat = U0_mat.reshape(*shape_compact)
            U0_mat = np.kron(np.eye(2 ** (n_qubits - ind - 1)), U0_mat)
            U0_gate = UnitaryGate(U0_mat)
            cU0_gate = U0_gate.control(ctrl_state=0)
        else:
            U1_op = 2 * val * QubitOperator(key)
            U1_mat = np.asarray(get_sparse_operator(U1_op).todense())
            U1_mat = U1_mat.reshape(*shape_expanded)
            U1_mat = U1_mat.transpose(*transpose_inds)
            U1_mat = U1_mat.reshape(*shape_compact)
            U1_mat = np.kron(np.eye(2 ** (n_qubits - ind - 1)), U1_mat)
            U1_gate = UnitaryGate(U1_mat)
            cU1_gate = U1_gate.control(ctrl_state=1)

    circ.h(0)
    circ.append(cU0_gate, qreg)
    circ.append(cU1_gate, qreg)
    circ.h(0)
    if measure:
        circ.measure(qreg[0], creg_anc)
    # TODO: Implement phase estimation on the system qubits
    
    return circ

def get_off_diagonal_circuits(ansatz, ind_left, ind_right, measure=True):
    """Returns the quantum circuits to calculate off-diagonal 
    elements of the Green's function."""
    # Create a new circuit with the ancillas as qubits 0 and 1
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 2)
    creg_anc = ClassicalRegister(2)
    creg_sys = ClassicalRegister(n_qubits)
    circ = QuantumCircuit(qreg, creg_anc, creg_sys)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 2 for qubit in qargs]
        circ.append(inst, qargs, cargs)

    arr = np.zeros((n_qubits,))
    arr[ind_left] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms_left = list(qubit_op.terms)

    arr = np.zeros((n_qubits,))
    arr[ind_right] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms_right = list(qubit_op.terms)

    # Build the circuit
    circ.barrier()
    circ.h(qreg[:2])
    circ.barrier()
    apply_ccU(circ, terms_left[0], ctrl=(0, 0))
    circ.barrier()
    apply_ccU(circ, terms_left[1], ctrl=(1, 0))
    circ.barrier()
    circ.rz(np.pi / 4, qreg[1])
    circ.barrier()
    apply_ccU(circ, terms_right[0], ctrl=(0, 1))
    circ.barrier()
    apply_ccU(circ, terms_right[1], ctrl=(1, 1))
    circ.barrier()
    circ.h(qreg[:2])
    if measure:
        circ.measure(qreg[:2], creg_anc)
    # TODO: Implement phase estimation on the system qubits

    return circ 

build_off_diagonal_circuits = get_off_diagonal_circuits

# XXX: Deprecated
def get_off_diagonal_circuits1(ansatz, ind_left, ind_right, measure=True):
    """Returns the quantum circuits to calculate off-diagonal 
    elements of the Green's function."""
    # Create a new circuit with the ancillas as qubits 0 and 1
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 2)
    creg_anc = ClassicalRegister(2)
    creg_sys = ClassicalRegister(n_qubits)
    circ = QuantumCircuit(qreg, creg_anc, creg_sys)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 2 for qubit in qargs]
        circ.append(inst, qargs, cargs)


    shape_expanded = (2,) * (ind_left + 1) * 2
    transpose_inds = np.hstack((np.flip(np.arange(ind_left + 1)), 
                                np.flip(np.arange(ind_left + 1) + ind_left + 1)))
    shape_compact = (2 ** (ind_left + 1), 2 ** (ind_left + 1))
    arr = np.zeros((n_qubits,))
    arr[ind_left] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    for key, val in qubit_op.terms.items():
        if np.isreal(val):
            U0_op = 2 * val * QubitOperator(key)
            U0_mat = np.asarray(get_sparse_operator(U0_op).todense())
            U0_mat = U0_mat.reshape(*shape_expanded)
            U0_mat = U0_mat.transpose(*transpose_inds)
            U0_mat = U0_mat.reshape(*shape_compact)
            U0_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_left)), U0_mat)
            U0_gate = UnitaryGate(U0_mat)
            cU0_gate_left = U0_gate.control(num_ctrl_qubits=2, ctrl_state='00')
        else:
            U1_op = 2 * val * QubitOperator(key)
            U1_mat = np.asarray(get_sparse_operator(U1_op).todense())
            U1_mat = U1_mat.reshape(*shape_expanded)
            U1_mat = U1_mat.transpose(*transpose_inds)
            U1_mat = U1_mat.reshape(*shape_compact)
            U1_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_left)), U1_mat)
            U1_gate = UnitaryGate(U1_mat)
            cU1_gate_left = U1_gate.control(num_ctrl_qubits=2, ctrl_state='01')

    shape_expanded = (2,) * (ind_right + 1) * 2
    transpose_inds = np.hstack((np.flip(np.arange(ind_right + 1)), 
                                np.flip(np.arange(ind_right + 1) + ind_right + 1)))
    shape_compact = (2 ** (ind_right + 1), 2 ** (ind_right + 1))
    arr = np.zeros((n_qubits,))
    arr[ind_right] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    for key, val in qubit_op.terms.items():
        if np.isreal(val):
            U0_op = 2 * val * QubitOperator(key)
            U0_mat = np.asarray(get_sparse_operator(U0_op).todense())
            U0_mat = U0_mat.reshape(*shape_expanded)
            U0_mat = U0_mat.transpose(*transpose_inds)
            U0_mat = U0_mat.reshape(*shape_compact)
            U0_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_right)), U0_mat)
            U0_gate = UnitaryGate(U0_mat)
            cU0_gate_right = U0_gate.control(num_ctrl_qubits=2, ctrl_state='10')
        else:
            U1_op = 2 * val * QubitOperator(key)
            U1_mat = np.asarray(get_sparse_operator(U1_op).todense())
            U1_mat = U1_mat.reshape(*shape_expanded)
            U1_mat = U1_mat.transpose(*transpose_inds)
            U1_mat = U1_mat.reshape(*shape_compact)
            U1_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_right)), U1_mat)
            U1_gate = UnitaryGate(U1_mat)
            cU1_gate_right = U1_gate.control(num_ctrl_qubits=2, ctrl_state='11')

    # Build the circuit
    circ.h(qreg[:2])
    circ.append(cU0_gate_left, qreg)
    circ.append(cU1_gate_left, qreg)
    circ.rz(np.pi / 4, qreg[1])
    circ.append(cU0_gate_right, qreg)
    circ.append(cU1_gate_right, qreg)
    circ.h(qreg[:2])
    if measure:
        circ.measure(qreg[:2], creg_anc)
    # TODO: Implement phase estimation on the system qubits

    return circ 

def apply_cU(circ, term, ctrl=1, offset=1):
    """Applies U0 or U1 in the diagonal-element circuits."""
    ind_max = max([t[0] for t in term])

    # Prepend gates for control on 0
    if ctrl == 0:
        circ.x(0)
    # Prepend gates for Pauli
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.s(0)
        circ.rx(np.pi / 2, ind_max + offset)
    # Implement multi-qubit gate
    for i in range(ind_max + offset, offset, -1):
        circ.cx(i, i - 1)
    circ.cz(0, offset)
    for i in range(offset, ind_max + offset):
        circ.cx(i + 1, i)
    # Append gates for Pauli
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.rx(-np.pi / 2, ind_max + offset)
    # Append gates for control on 0
    if ctrl == 0:
        circ.x(0)

def apply_ccU(circ, term, ctrl=(1, 1)):
    """Applies U0 or U1 in the off-diagonal-element circuits."""
    ind_max = max([t[0] for t in term])
    
    # Prepend gates for control on 0
    if ctrl[0] == 0:
        circ.x(0)
    if ctrl[1] == 0:
        circ.x(1)

    #if ctrl[0] == 1:
    #    circ.s(0)
    # Prepend gates for Pauli
    if term[-1][1] == 'X':
        circ.h(ind_max + 2)
    elif term[-1][1] == 'Y':
        circ.cp(np.pi / 2, 0, 1)
        circ.rx(np.pi / 2, ind_max + 2)
    # Impement multi-qubit gate
    for i in range(ind_max + 2, 2, -1):
        circ.cx(i, i - 1)
    circ.h(2)
    circ.ccx(0, 1, 2)
    circ.h(2)
    for i in range(2, ind_max + 2):
        circ.cx(i + 1, i)
    # Append gates for rotation
    if term[-1][1] == 'X':
        circ.h(ind_max + 2)
    elif term[-1][1] == 'Y':
        circ.rx(-np.pi / 2, ind_max + 2)
    
    # Append gates for control on 0
    if ctrl[0] == 0:
        circ.x(0)
    if ctrl[1] == 0:
        circ.x(1)

