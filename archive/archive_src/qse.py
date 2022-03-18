# TODO: Complete this script.

def run_qse(vqe_result, paulis):
    if isinstance(paulis[0], str):
        paulis = [Pauli(label) for label in paulis]
    print(vqe_result.__dict__.keys())

def build_qse_matrices(hamiltonian, state, paulis):
    states = [pauli @ state for pauli in paulis]
    states = np.array(states)
    print(states.conj().T.shape)
    print(hamiltonian.shape)
    H = states @ hamiltonian @ states.conj().T
    S = states @ states.conj().T
    return H, S

def roothaan_eig(Hmat, Smat):
    s, U = np.linalg.eigh(Smat)
    idx = np.where(abs(s)>1e-8)[0]
    s = s[idx]
    U = U[:,idx]
    Xmat = np.dot(U, np.diag(1/np.sqrt(s)))
    Hmat_ = np.dot(Xmat.T.conj(), np.dot(Hmat, Xmat))
    w, v = np.linalg.eigh(Hmat_)
    v = np.dot(Xmat, v)
    return w, v


def quantum_subspace_expansion(ansatz,
                               hamiltonian_op: PauliSumOp,
                               qse_ops: List[PauliSumOp],
                               q_instance: Optional[QuantumInstance] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Quantum subspace expansion.
    
    Args:
        hamiltonian_op: The Hamiltonian operator.
        qse_ops: The quantum subspace expansion operators.
        q_instance: The QuantumInstance used to execute the circuits.
        
    Returns:
        eigvals: The energy eigenvalues in the subspace.
        eigvecs: The states in the subspace.
    """
    # if q_instance is None or q_instance.backend.name() == 'statevector_simulator':
    # q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100000)

    dim = len(qse_ops)

    qse_mat = np.zeros((dim, dim))
    overlap_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            op = qse_ops[i].adjoint().compose(hamiltonian_op.compose(qse_ops[j]))
            op = transform_4q_hamiltonian(op.reduce(), init_state=[1, 1])
            op = op.reduce()
            qse_mat[i, j] = measure_operator(ansatz, op, q_instance)

            op = qse_ops[i].adjoint().compose(qse_ops[j])
            op = transform_4q_hamiltonian(op.reduce(), init_state=[1, 1])
            op = op.reduce()
            overlap_mat[i, j] = measure_operator(ansatz, op, q_instance)

    print('qse_mat\n', qse_mat * HARTREE_TO_EV)
    print('overlap_mat\n', overlap_mat * HARTREE_TO_EV)
    
    eigvals, eigvecs = roothaan_eig(qse_mat, overlap_mat)
    # print(eigvals * HARTREE_TO_EV)
    return eigvals, eigvecs

def quantum_subspace_expansion_exact(
        ansatz,
        hamiltonian_op: PauliSumOp,
        qse_ops: List[PauliSumOp],
        q_instance = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact quantum subspace expansion for benchmarking."""

    psi = get_statevector(ansatz)
    
    dim = len(qse_ops)
    qse_mat = np.zeros((dim, dim))
    overlap_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            op = qse_ops[i].adjoint().compose(hamiltonian_op.compose(qse_ops[j]))
            op = transform_4q_hamiltonian(op.reduce(), init_state=[1, 1])
            mat = op.to_matrix()
            qse_mat[i, j] = psi.conj() @ mat @ psi

            op = qse_ops[i].adjoint().compose(qse_ops[j])
            op = transform_4q_hamiltonian(op.reduce(), init_state=[1, 1])
            mat = op.to_matrix()
            overlap_mat[i, j] = psi.conj() @ mat @ psi
        
    print('qse_mat\n', qse_mat * HARTREE_TO_EV)
    print('overlap_mat\n', overlap_mat * HARTREE_TO_EV)
    
    eigvals, eigvecs = roothaan_eig(qse_mat, overlap_mat)
    print(eigvals * HARTREE_TO_EV)
    return eigvals, eigvecs

def roothaan_eig(Hmat: np.ndarray, 
                 Smat: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Solves the Roothaan-type eigenvalue equation HC = SCE.
    
    Args:
        Hmat: The Hamiltonian matrix.
        Smat: The overlap matrix.
        
    Returns:
        w: The eigenvalues.
        v: The eigenvectors.
    """
    s, U = np.linalg.eigh(Smat)
    idx = np.where(abs(s) > 1e-8)[0]
    s = s[idx]
    U = U[:,idx]
    Xmat = np.dot(U, np.diag(1 / np.sqrt(s)))
    Hmat_ = np.dot(Xmat.T.conj(), np.dot(Hmat, Xmat))
    w, v = np.linalg.eigh(Hmat_)
    v = np.dot(Xmat, v)
    return w, v




