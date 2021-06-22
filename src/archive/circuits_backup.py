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