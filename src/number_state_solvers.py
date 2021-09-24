from typing import Union, Tuple, List, Iterable, Optional, Sequence
from itertools import combinations

import numpy as np
from scipy.sparse.data import _data_matrix

from qiskit import *
from openfermion.linalg import get_sparse_operator
from openfermion.ops.operators.qubit_operator import QubitOperator

from hamiltonians import MolecularHamiltonian


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
    """Exact eigensolver for the Hamiltonian in the subspace of 
    a certain number of electrons.
    
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
    elif (isinstance(hamiltonian, _data_matrix) or 
          isinstance(hamiltonian, np.ndarray)):
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