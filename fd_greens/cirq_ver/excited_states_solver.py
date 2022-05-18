"""
=================================================================
N-Electron States Solver (:mod:`fd_greens.excited_states_solver`)
=================================================================
"""

import h5py
import numpy as np

from fd_greens.cirq_ver.qubit_indices import QubitIndices

from .molecular_hamiltonian import MolecularHamiltonian
from .parameters import method_indices_pairs


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
            h5fname: The HDF5 file name.
        """
        self.hamiltonian = hamiltonian
        self.hamiltonian.transform(method_indices_pairs, tapered_state=[1, 1])
        self.h5fname = h5fname + '.h5'

        n_qubits = 2 * len(self.hamiltonian.active_indices)
        self.qubit_indices_dict = QubitIndices.get_excited_qubit_indices_dict(
            n_qubits, method_indices_pairs, True)

        self.energies = dict()
        self.state_vectors = dict()

        self.h5fname = h5fname + ".h5"

    def _run_exact(self):
        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        self.energies['s'], self.state_vectors['s'] = eigensolve(
            self.hamiltonian.matrix, self.qubit_indices_dict['s'].int)
        self.energies['t'], self.state_vectors['t'] = eigensolve(
            self.hamiltonian.matrix, self.qubit_indices_dict['t'].int)
        print(f"Singlet excited-state energies are {self.energies['s']} eV")
        print(f"Triplet excited state energies are {self.energies['t']} eV")

    def _save_data(self):
        """Saves N-electron excited-state energies and states to hdf5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        for dset_name in ['es/energies_s', 'es/energies_t', 'es/states_s', 'es_states_t']:
            if dset_name in h5file:
                del h5file[dset_name]

        h5file['es/energies_s'] = self.energies['s']
        h5file['es/energies_t'] = self.energies['t']
        h5file['es/states_s'] = self.state_vectors['s']
        h5file['es/states_t'] = self.state_vectors['t']

        h5file.close()

    def run(self) -> None:
        """Runs the excited state calculation."""
        self._run_exact()
        self._save_data()
