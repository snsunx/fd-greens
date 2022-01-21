"""Number states solver module."""

from typing import Union, Tuple, List, Iterable, Optional, Sequence
from itertools import combinations

import h5py
import numpy as np
from scipy.sparse.data import _data_matrix

from qiskit import *
from qiskit.utils import QuantumInstance

from openfermion.linalg import get_sparse_operator
from openfermion.ops.operators.qubit_operator import QubitOperator

from hamiltonians import MolecularHamiltonian
from operators import transform_4q_pauli
import params
from ground_state_solvers import vqe_minimize
from ansatze import AnsatzFunction
from utils import state_tomography
from qubit_indices import transform_4q_indices

def get_number_state_indices(n_orb: int,
                             n_elec: int,
                             anc: Iterable[str] = '',
                             return_type: str = 'decimal',
                             reverse: bool = True) -> List[int]:
    """Obtains the indices corresponding to a certain number of electrons.

    Args:
        n_orb: An integer indicating the number of orbitals.
        n_elec: An integer indicating the number of electrons.
        anc: An iterable of '0' and '1' indicating the state of the ancilla qubit(s).
        return_type: Type of the indices returned. Must be 'decimal' or 'binary'. Default to 'decimal'.
        reverse: Whether the qubit indices are reversed because of Qiskit qubit order. Default to True.
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
    """Exact Hamiltonian eigensolver in the subspace of a certain number of electrons.

    Args:
        hamiltonian: The Hamiltonian of the molecule.
        n_elec: An integer indicating the number of electrons.
    
    Returns:
        eigvals: The energies in the number state subspace.
        eigvecs: The states in the number state subspace.
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

eigensolve = number_state_eigensolver

class EHStatesSolver:
    """A class to calculate and store information of N+/-1 electron states."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 ansatz_func_e: Optional[AnsatzFunction] = None, 
                 ansatz_func_h: Optional[AnsatzFunction] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 spin: str = 'edhu',
                 h5fname: str = 'lih', 
                 dsetname: str = 'eh'):
        """Initializes a EHStatesSolver object.
        
        Args:
            h_op: The Hamiltonian operator.
            ansatz_func_e: The ansatz function for N+1 electron states.
            ansatz_func_h: The ansatz function for N-1 electron states.
            q_instance: The QuantumInstance object for N+/-1 electron state calculation.
        """
        assert spin in ['euhd', 'edhu']

        self.h = h
        if spin == 'euhd': # e up h down
            self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=[0, 1])
        elif spin == 'edhu': # e down h up
            self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=[1, 0])
        self.h_mat = self.h_op.to_matrix()

        if spin == 'euhd':
            self.inds_e = transform_4q_indices(params.eu_inds)
            self.inds_h = transform_4q_indices(params.hd_inds)
        elif spin == 'edhu':
            self.inds_e = transform_4q_indices(params.ed_inds)
            self.inds_h = transform_4q_indices(params.hu_inds)

        self.ansatz_func_e = ansatz_func_e
        self.ansatz_func_h = ansatz_func_h
        self.q_instance = q_instance

        self.states_e = None
        self.states_h = None

        self.h5fname = h5fname + '.hdf5'
        self.dsetname = dsetname

    def run_exact(self):
        """Calculates the exact N+/-1 electron states of the Hamiltonian."""
        self.energies_e, self.states_e = eigensolve(self.h_mat, inds=self.inds_e.int_form)
        self.energies_h, self.states_h = eigensolve(self.h_mat, inds=self.inds_h.int_form)
        self.states_e = self.states_e.T
        self.states_h = self.states_h.T
        print(f"N+1 electron energies are {self.energies_e} eV")
        print(f"N-1 electron energies are {self.energies_h} eV")

    def run_vqe(self):
        """Calculates the N+/-1 electron states of the Hamiltonian using VQE."""
        energy_min, circ_min = vqe_minimize(self.h_op, ansatz_func=self.ansatz_func_e,
                                            init_params=(0.,), q_instance=self.q_instance)
        m_energy_max, circ_max = vqe_minimize(-self.h_op, ansatz_func=self.ansatz_func_e,
                                              init_params=(0.,), q_instance=self.q_instance)
        self.energies_e = np.array([energy_min, -m_energy_max])
        states_e = [state_tomography(circ_min, q_instance=self.q_instance), 
                    state_tomography(circ_max, q_instance=self.q_instance)]
        self.states_e = [rho[self.inds_e.int_form][:, self.inds_e.int_form] for rho in states_e]

        energy_min, circ_min = vqe_minimize(self.h_op, ansatz_func=self.ansatz_func_h, 
                                            init_params=(0.,), q_instance=self.q_instance)
        m_energy_max, circ_max = vqe_minimize(-self.h_op, ansatz_func=self.ansatz_func_h,
                                              init_params=(0.,), q_instance=self.q_instance)
        self.energies_h = np.array([energy_min, -m_energy_max])
        states_h = [state_tomography(circ_min, q_instance=self.q_instance), 
                    state_tomography(circ_max, q_instance=self.q_instance)]
        self.states_h = [rho[self.inds_h.int_form][:, self.inds_h.int_form] for rho in states_h]

        print(f"N+1 electron energies are {self.energies_e} eV")
        print(f"N-1 electron energies are {self.energies_h} eV")

    def save_data(self) -> None:
        h5file = h5py.File(self.h5fname, 'r+')
        dset = h5file[self.dsetname]

        dset.attrs['energies_e'] = self.energies_e
        dset.attrs['energies_h'] = self.energies_h
        dset.attrs['states_e'] = self.states_e
        dset.attrs['states_h'] = self.states_h

        h5file.close()

    def run(self, method='vqe'):
        """Runs the N+/-1 electron states calculation."""
        if method == 'exact':
            self.run_exact()
        elif method == 'vqe':
            self.run_vqe()
        self.save_data()

class ExcitedStatesSolver:
    """A class to calculate and store information of excited states."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 ansatz_func_e: Optional[AnsatzFunction] = None, 
                 ansatz_func_h: Optional[AnsatzFunction] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 apply_tomography: bool = False):
        """Initializes a EHStatesSolver object.
        
        Args:
            h: The MolecularHamiltonian object.
            ansatz_func_e: The ansatz function for N+1 electron states.
            ansatz_func_h: The ansatz function for N-1 electron states.
            q_instance: The QuantumInstance object for N+/-1 electron state calculation.
            apply_tomography: Whether tomography of the states is applied.
        """
        self.h = h
        self.h_op = self.h.qiskit_op
        self.h_op_s = transform_4q_pauli(self.h_op, init_state=[1, 1])
        self.h_op_t = transform_4q_pauli(self.h_op, init_state=[0, 0])
        self.h_mat_s = self.h_op_s.to_matrix()
        self.h_mat_t = self.h_op_t.to_matrix()

        self.inds_s = transform_4q_indices(params.singlet_inds)
        self.inds_t = transform_4q_indices(params.triplet_inds)

    def _run_exact(self):
        self.energies_s, self.states_s = eigensolve(self.h_mat_s, inds=self.inds_s.int_form)
        self.energies_t, self.states_t = eigensolve(self.h_mat_t, inds=self.inds_t.int_form)
        self.states_s = self.states_s.T
        self.states_t = self.states_t.T
        print(f"Singlet excited-state energies are {self.energies_s} eV")
        print(f"Triplet excited state energies are {self.energies_t} eV")

    def run(self, method='exact'):
        """Runs the excited states calculation."""
        if method == 'exact':
            self._run_exact()
        elif method == 'vqe':
            pass