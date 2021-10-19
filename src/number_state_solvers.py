from typing import Union, Tuple, List, Iterable, Optional, Sequence
from itertools import combinations

import numpy as np
from scipy.sparse.data import _data_matrix

from qiskit import *
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliSumOp

from openfermion.linalg import get_sparse_operator
from openfermion.ops.operators.qubit_operator import QubitOperator

from hamiltonians import MolecularHamiltonian
from z2_symmetries import transform_4q_hamiltonian

from constants import HARTREE_TO_EV
from utils import get_statevector


def get_number_state_indices(n_orb: int,
                             n_elec: int,
                             anc: Iterable[str] = '',
                             return_type: str = 'decimal',
                             reverse: bool = True) -> List[int]:
    """Obtains the indices corresponding to a certain number of electrons.

    Args:
        n_orb: An integer indicating the number of orbitals.
        n_elec: An integer indicating the number of electrons.
        anc: An iterable of '0' and '1' indicating the state of the
            ancilla qubit(s).
        return_type: Type of the indices returned. Must be 'decimal'
            or 'binary'. Default to 'decimal'.
        reverse: Whether the qubit indices are reversed because of
            Qiskit qubit order. Default to True.
    """
    assert return_type in ['binary', 'decimal']
    inds = []
    for tup in combinations(range(n_orb), n_elec):
        bin_list = ['1' if (n_orb - 1 - i) in tup else '0'
                    for i in range(n_orb)]
        # TODO: Technically the anc[::-1] can be taken care of outside this function.
        # Should implement binary indices in both list form and string form
        if reverse:
            bin_str = ''.join(bin_list) + anc[::-1]
        else:
            bin_str = anc + ''.join(bin_list)
        inds.append(bin_str)
    if reverse:
        inds = sorted(inds, reverse=True)
    if return_type == 'decimal':
        inds = [int(s, 2) for s in inds]
    return inds

def number_state_eigensolver(
        hamiltonian: Union[MolecularHamiltonian, QubitOperator, np.ndarray],
        n_elec: Optional[int] = None,
        inds: Optional[Sequence[str]] = None,
        reverse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: Update docstring for n_elec, inds and reverse
    """Exact eigensolver for the Hamiltonian in the subspace of a certain number of electrons.

    Args:
        hamiltonian: The Hamiltonian of the molecule.
        n_elec: An integer indicating the number of electrons.
    `
    Returns:
        eigvals: The eigenenergies in the number state subspace.
        eigvecs: The eigenstates in the number state subspace.
    """
    if isinstance(hamiltonian, MolecularHamiltonian):
        hamiltonian_arr = hamiltonian.to_array(array_type='sparse')
    elif isinstance(hamiltonian, QubitOperator):
        hamiltonian_arr = get_sparse_operator(hamiltonian)
    elif isinstance(hamiltonian, _data_matrix) or isinstance(hamiltonian, np.ndarray):
        hamiltonian_arr = hamiltonian
    else:
        raise TypeError("Hamiltonian must be one of MolecularHamiltonian,"
                        "QubitOperator, sparse array or ndarray")

    if inds is None:
        n_orb = int(np.log2(hamiltonian_arr.shape[0]))
        inds = get_number_state_indices(n_orb, n_elec, reverse=reverse)
    hamiltonian_subspace = hamiltonian_arr[inds][:, inds]
    if isinstance(hamiltonian_subspace, _data_matrix):
        hamiltonian_subspace = hamiltonian_subspace.toarray()

    eigvals, eigvecs = np.linalg.eigh(hamiltonian_subspace)

    # TODO: Note that the minus sign below depends on the `reverse` variable.
    # Might need to take care of this
    sort_arr = [(eigvals[i], -np.argmax(np.abs(eigvecs[:, i])))
                for i in range(len(eigvals))]
    sort_arr = [x[0] + 1e-4 * x[1] for x in sort_arr] # XXX: Ad-hoc
    # print('sort_arr =', sort_arr)
    # print('sorted(sort_arr) =', sorted(sort_arr))
    inds_new = sorted(range(len(sort_arr)), key=sort_arr.__getitem__)
    # print(inds_new)
    eigvals = eigvals[inds_new]
    eigvecs = eigvecs[:, inds_new]
    return eigvals, eigvecs

def quantum_subspace_expansion(ansatz,
                               hamiltonian_op: PauliSumOp,
                               qse_ops: List[PauliSumOp],
                               q_instance: Optional[QuantumInstance] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Quantum subspace expansion."""
    # if q_instance is None or q_instance.backend.name() == 'statevector_simulator':
    q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100000)

    # print('q_instance =', q_instance)
    dim = len(qse_ops)

    qse_mat = np.zeros((dim, dim))
    overlap_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            op = qse_ops[i].adjoint().compose(hamiltonian_op.compose(qse_ops[j]))
            op = transform_4q_hamiltonian(op.reduce(), init_state=[1, 1])
            op = op.reduce()
            # print('i, j', i, j, '\n', op.reduce())
            qse_mat[i, j] = measure_operator(ansatz, op, q_instance)

            op = qse_ops[i].adjoint().compose(qse_ops[j])
            op = transform_4q_hamiltonian(op.reduce(), init_state=[1, 1])
            op = op.reduce()
            # print('i, j', i, j, '\n', op.reduce())
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

def measure_operator(circ: QuantumCircuit,
                     op: PauliSumOp,
                     q_instance: QuantumInstance,
                     anc_state: Sequence[int] = [],
                     qreg: Optional[QuantumRegister] = None) -> float:
    """Measures an operator on a circuit.

    Args:
        circ: The quantum circuit to be measured.
        op: The operator to be measured.
        q_instance: The QuantumInstance on which to execute the circuit.
        qreg: The quantum register on which the operator is measured.
    
    Returns:
        The value of the operator on the circuit.
    """
    if qreg is None:
        qreg = circ.qregs[0]
    n_qubits = len(qreg)
    n_anc = len(anc_state)
    n_sys = n_qubits - n_anc

    circ_list = []
    for term in op:
        label = term.primitive.table.to_labels()[0]
        
        # Create circuit for measuring this Pauli string
        circ_meas = circ.copy()
        creg = ClassicalRegister(n_qubits)
        circ_meas.add_register(creg)

        for i in range(n_anc):
            circ_meas.measure(i, i)
        
        for i in range(n_sys):
            if label[n_sys - 1 - i] == 'X':
                circ_meas.h(qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == 'Y':
                circ_meas.rx(np.pi / 2, qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == 'Z':
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])

        circ_list.append(circ_meas)
    
    result = q_instance.execute(circ_list)
    counts_list = result.get_counts()

    value = 0
    print('measure_operator')
    for i, counts in enumerate(counts_list):
        print('counts =', counts)
        coeff = op.primitive.coeffs[i]
        counts_new = counts.copy()

        for key, val in counts.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if key_anc != anc_state:
                del counts_new[key]
        
        shots = sum(counts_new.values())
        for key, val in counts_new.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if sum(key_sys) % 2 == 0:
                value += coeff * val / shots
            else:
                value -= coeff * val / shots
        print(shots)
    return value

def roothaan_eig(Hmat, Smat):
    """Solves the Roothaan-type eigenvalue equation."""
    s, U = np.linalg.eigh(Smat)
    idx = np.where(abs(s) > 1e-8)[0]
    s = s[idx]
    U = U[:,idx]
    Xmat = np.dot(U, np.diag(1 / np.sqrt(s)))
    Hmat_ = np.dot(Xmat.T.conj(), np.dot(Hmat, Xmat))
    w, v = np.linalg.eigh(Hmat_)
    v = np.dot(Xmat, v)
    return w, v