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


