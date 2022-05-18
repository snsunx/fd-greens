"""
=================================================================
N-Electron States Solver (:mod:`fd_greens.excited_states_solver`)
=================================================================
"""

from typing import Optional

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .parameters import method_indices_pairs

# from .params import singlet_inds, triplet_inds
# from .ansatze import AnsatzFunction, build_ansatz_e, build_ansatz_h


class ExcitedStatesSolver:
    """A class to calculate and store information of excited states."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        h5fname: str = "lih",
    ):
        """Initializes a EHStatesSolver object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            apply_tomography: Whether tomography of the states is applied.
            h5fname: The HDF5 file name.
        """
        self.hamiltonian = hamiltonian
        self.hamiltonian.transform(method_indices_pairs, tapered_state=[1, 1])

        # self.h_op = self.h.qiskit_op
        # self.h_op_s = transform_4q_pauli(self.h_op, init_state=[1, 1])
        # self.h_op_t = transform_4q_pauli(self.h_op, init_state=[0, 0])
        # self.h_mat_s = self.h_op_s.to_matrix()
        # self.h_mat_t = self.h_op_t.to_matrix()

        self.inds_s = transform_4q_indices(singlet_inds)
        self.inds_t = transform_4q_indices(triplet_inds)

        self.h5fname = h5fname + ".h5"

    def _run_exact(self):
        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        self.energies_s, self.states_s = eigensolve(
            self.h_mat_s, inds=self.inds_s.int_form
        )
        self.energies_t, self.states_t = eigensolve(
            self.h_mat_t, inds=self.inds_t.int_form
        )
        self.states_s = self.states_s.T
        self.states_t = self.states_t.T
        print(f"Singlet excited-state energies are {self.energies_s} eV")
        print(f"Triplet excited state energies are {self.energies_t} eV")

    def _save_data(self):
        """Saves N-electron excited-state energies and states to hdf5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        for dset_name in ['es/energies_s', 'es/energies_t', 'es/states_s', 'es_states_t']:
            if dset_name in h5file:
                del h5file[dset_name]

        h5file['es/energies_s'] = self.energies['s']
        h5file['es/energies_t'] = self.energies['t']
        h5file['es/states_s'] = self.states['s']
        h5file['es/states_t'] = self.states['t']

        h5file.close()

    def run(self) -> None:
        """Runs the excited state calculation."""
        self._run_exact()
        self._save_data()
