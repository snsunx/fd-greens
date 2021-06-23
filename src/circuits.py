import numpy as np
from qiskit import *
from openfermion.ops import PolynomialTensor, QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from qiskit.extensions import UnitaryGate

def build_diagonal_circuits(ansatz, ind, measure=False):
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
    circ.barrier()

    if measure:
        circ.measure(qreg[0], creg_anc)
    # TODO: Implement phase estimation on the system qubits

    return circ

def build_diagonal_circuits_with_qpe(ansatz, ind, add_barriers=True, measure=False):
    """Constructs the quantum circuit to calculate diagonal elements of the 
    Green's function with an ancilla qubit for QPE."""
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

    if add_barriers:
        circ.barrier()
    circ.h(0)
    if add_barriers:
        circ.barrier()
    apply_cU(circ, terms[0], ctrl=0, offset=2)
    if add_barriers:
        circ.barrier()
    apply_cU(circ, terms[1], ctrl=1, offset=2)
    if add_barriers:
        circ.barrier()
    circ.h(0)
    if add_barriers:
        circ.barrier()

    if measure:
        circ.measure(0, 0)
    
    return circ

def build_off_diagonal_circuits(ansatz, ind_left, ind_right, add_barriers=True, measure=True):
    """Returns the quantum circuits to calculate off-diagonal 
    elements of the Green's function."""
    # Create a new circuit with the ancillas as qubits 0 and 1
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 2)
    creg = ClassicalRegister(2)
    # creg_sys = ClassicalRegister(n_qubits)
    circ = QuantumCircuit(qreg, creg)
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
    circ.h([0, 1])
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
    circ.h([0, 1])
    if add_barriers:
        circ.barrier()

    if measure:
        circ.measure([0, 1], [0, 1])

    return circ 

def build_off_diagonal_circuits_with_qpe(ansatz, ind_left, ind_right, 
                                         add_barriers=True, measure=False):
    """Constructs the quantum circuit to calculate off-diagonal elements of the 
    Green's function with an ancilla qubit for QPE."""
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 3)
    creg = ClassicalRegister(3)
    circ = QuantumCircuit(qreg, creg)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 3 for qubit in qargs]
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

    if add_barriers:
        circ.barrier()
    circ.h([0, 1])
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, terms_left[0], ctrl=(0, 0), offset=3)
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, terms_left[1], ctrl=(1, 0), offset=3)
    if add_barriers:
        circ.barrier()
    circ.rz(np.pi / 4, 1)
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, terms_right[0], ctrl=(0, 1), offset=3)
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, terms_right[1], ctrl=(1, 1), offset=3)
    if add_barriers:
        circ.barrier()
    circ.h([0, 1])

    if measure:
        circ.measure([0, 1], [0, 1])

    return circ



def apply_cU(circ, term, ctrl=1, offset=1):
    """Applies a controlled-U gate."""
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

def apply_ccU(circ, term, ctrl=(1, 1), offset=2):
    """Applies a controlled-controlled-U gate."""
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
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.cp(np.pi / 2, 0, 1)
        circ.rx(np.pi / 2, ind_max + offset)
    # Implement multi-qubit gate
    for i in range(ind_max + offset, offset, -1):
        circ.cx(i, i - 1)
    circ.h(offset)
    circ.ccx(0, 1, offset)
    circ.h(offset)
    for i in range(offset, ind_max + offset):
        circ.cx(i + 1, i)
    # Append gates for rotation
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.rx(-np.pi / 2, ind_max + offset)
    
    # Append gates for control on 0
    if ctrl[0] == 0:
        circ.x(0)
    if ctrl[1] == 0:
        circ.x(1)

