from hamiltonians import MolecularHamiltonian
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp
from itertools import combinations
import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion.ops.operators.qubit_operator import QubitOperator
from scipy.sparse.data import _data_matrix

# from greens_function import GreensFunction

def reverse_qubit_order(arr):
    dim = arr.shape[0]
    num = int(np.log2(dim))
    shape_expanded = (2,) * num
    inds_transpose = np.flip(np.arange(num))
    arr = arr.reshape(*shape_expanded)
    arr = arr.transpose(*inds_transpose)
    arr = arr.reshape(dim)
    return arr

def get_number_state_indices(n_orb, n_elec, anc='', return_type='decimal'):
    assert return_type in ['binary', 'decimal']
    inds = []
    for tup in combinations(range(n_orb), n_elec):
        bin_list = ['1' if (n_orb - 1 - i) in tup else '0' for i in range(n_orb)]
        bin_str = anc + ''.join(bin_list)
        inds.append(bin_str)
    inds.sort()
    if return_type == 'binary':
        return inds
    else:
        inds = [int(s, 2) for s in inds]
        return inds

def number_state_eigensolver(hamiltonian, n_elec):
    """Exact eigensolver for the Hamiltonian in the subspace of 
    a certain number of electrons."""
    if isinstance(hamiltonian, MolecularHamiltonian):
        hamiltonian_arr = hamiltonian.to_array(array_type='sparse')
    elif isinstance(hamiltonian, QubitOperator):
        hamiltonian_arr = get_sparse_operator(hamiltonian)
    elif (isinstance(hamiltonian, _data_matrix) or 
          isinstance(hamiltonian, np.ndarray)):
        hamiltonian_arr = hamiltonian
    else:
        raise TypeError("""hamiltonian must be one of MolecularHamiltonian, 
                        QubitOperator, sparse array or ndarray""")

    n_orb = int(np.log2(hamiltonian_arr.shape[0]))
    inds = get_number_state_indices(n_orb, n_elec)
    hamiltonian_subspace = hamiltonian_arr[inds][:, inds]
    if isinstance(hamiltonian_subspace, _data_matrix):
        hamiltonian_subspace = hamiltonian_subspace.toarray()
    
    eigvals, eigvecs = np.linalg.eigh(hamiltonian_subspace)
    return eigvals, eigvecs




