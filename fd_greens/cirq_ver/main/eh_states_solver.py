"""
=====================================================================
(N±1)-Electron States Solver (:mod:`fd_greens.main.eh_states_solver`)
=====================================================================
"""
from typing import Optional

import h5py
import numpy as np

from .parameters import method_indices_pairs
from .molecular_hamiltonian import MolecularHamiltonian
from .z2symmetries import transform_4q_pauli
from .qubit_indices import get_qubit_indices_dict


class EHStatesSolver:
    """(N±1)-electron states solver."""

    def __init__(
        self, hamiltonian: MolecularHamiltonian, spin: str = "d", h5fname: str = "lih"
    ) -> None:
        """Initializes an ``EHStatesSolver`` object.
        
        Args:
            h: The molecular Hamiltonian.
            spin: A character indicating whether up (``'u'``) or down spins (``'d'``) are included.
            h5fname: The HDF5 file name.
        """
        assert spin in ["u", "d"]

        init_state = [1, 0] if spin == "d" else [0, 1]
        self.h_op = transform_4q_pauli(hamiltonian.qiskit_op, init_state=init_state)
        self.qubit_indices_dict = get_qubit_indices_dict(4, spin, method_indices_pairs, system_only=True)
        self.h5fname = h5fname + ".h5"

        self.energies = dict()
        self.state_vectors = dict()

    def _run_exact(self) -> None:
        """Calculates the exact (N±1)-electron energies and states of the Hamiltonian."""

        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        h_arr = self.h_op.to_matrix()
        self.energies['e'], self.state_vectors['e'] = eigensolve(h_arr, inds=self.qubit_indices_dict['e'].int)
        self.energies['h'], self.state_vectors['h'] = eigensolve(h_arr, inds=self.qubit_indices_dict['h'].int)
        print(f"(N+1)-electron energies are {self.energies['e']} eV")
        print(f"(N-1)-electron energies are {self.energies['h']} eV")

    def _save_data(self) -> None:
        """Saves (N±1)-electron energies and state vectors to HDF5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        for dset_name in ['es/energies_e', 'es/energies_h', 'es/states_e', 'es/states_h']:
            if dset_name in h5file:
                del h5file[dset_name]
        
        h5file['es/energies_e'] = self.energies['e']
        h5file['es/energies_h'] = self.energies['h']
        h5file['es/states_e'] = self.state_vectors['e']
        h5file['es/states_h'] = self.state_vectors['h']

        h5file.close()

    def run(self) -> None:
        """Runs the (N±1)-electron states calculation."""
        self._run_exact()
        self._save_data()
