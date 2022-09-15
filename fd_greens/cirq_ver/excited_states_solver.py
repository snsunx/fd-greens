"""
=================================================================
N-Electron States Solver (:mod:`fd_greens.excited_states_solver`)
=================================================================
"""

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .parameters import Z2TransformInstructions
from .qubit_indices import QubitIndices
from .helpers import save_to_hdf5

class ExcitedStatesSolver:
    """N-electron excited states solver."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        h5fname: str = "lih",
    ) -> None:
        """Initializes a ``EHStatesSolver`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            h5fname: The HDF5 file name.
        """
        self.hamiltonian = hamiltonian.copy()
        instructions = Z2TransformInstructions.get_instructions(' ')
        self.hamiltonian.transform(instructions, tapered_state=[1, 1])
        self.h5fname = h5fname

        n_qubits = 2 * len(self.hamiltonian.active_indices)
        self.qubit_indices_dict = QubitIndices.get_excited_qubit_indices_dict(
            n_qubits, instructions, system_only=True)

        self.energies = None
        self.state_vectors = None

    def _run_exact(self):
        """Runs exact calculation of excited states."""
        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            # print("e\n", e)
            # print("v\n", v)
            return e[1:], v[:, 1:] # 1: for excluding ground state

        self.energies, self.state_vectors = eigensolve(self.hamiltonian.matrix, self.qubit_indices_dict['n'].int)
        print(f"Singlet excited-state energies are {self.energies} eV")

    def _save_data(self):
        """Saves N-electron excited-state energies and states to HDF5 file."""
        with h5py.File(self.h5fname + '.h5', 'r+') as h5file:
            save_to_hdf5(h5file, "es/energies_n", self.energies)
            save_to_hdf5(h5file, "es/states_n", self.state_vectors)

    def run(self) -> None:
        """Runs the excited state calculation."""
        self._run_exact()
        self._save_data()
