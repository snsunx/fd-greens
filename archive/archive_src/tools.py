from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp
from itertools import combinations
import numpy as np

def get_number_state_indices(n_orb, n_elec, anc=''):
    inds = []
    for tup in combinations(range(n_orb), n_elec):
        bin_list = ['1' if (n_orb - 1 - i) in tup else '0' for i in range(n_orb)]
        bin_str = anc + ''.join(bin_list)
        inds.append(int(bin_str, 2))
    inds.sort()
    return inds

def number_state_eigensolver(hamiltonian_mat, n_elec):
    # TODO: If hamiltonian is not a sparse array, convert it to sparse array
    n_orb = int(np.log2(hamiltonian_mat.shape[0]))
    inds = get_number_state_indices(n_orb, n_elec)
    hamiltonian_subspace = hamiltonian_mat[inds][:, inds].toarray()
    eigvals, eigvecs = np.linalg.eigh(hamiltonian_subspace)
    return eigvals, eigvecs

# XXX: This function is deprecated.
def openfermion_to_qiskit_operator(opf_qubit_op):
    """Converts an openfermion QubitOperator to a qiskit PauliSumOp."""
    table = []
    coeffs = []
    n_qubits = 0
    for key in opf_qubit_op.terms:
        if key == ():
            continue
        num = max([t[0] for t in key])
        if num > n_qubits:
            n_qubits = num
    n_qubits += 1

    for key, val in opf_qubit_op.terms.items():
        coeffs.append(val)
        label = ['I'] * n_qubits
        for i, s in key:
            label[i] = s
        label = ''.join(label)
        pauli = Pauli(label[::-1]) # because Qiskit qubit order is reversed
        mask = list(pauli.x) + list(pauli.z)
        table.append(mask)
    primitive = SparsePauliOp(table, coeffs)
    qk_qubit_op = PauliSumOp(primitive)
    return qk_qubit_op
