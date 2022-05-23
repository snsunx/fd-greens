"""
================================================================
(N±1)-Electron States Solver (:mod:`fd_greens.eh_states_solver`)
================================================================
"""

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import REVERSE_QUBIT_ORDER, get_method_indices_pairs


class EHStatesSolver:
    """(N±1)-electron states solver."""

    def __init__(self, hamiltonian: MolecularHamiltonian, spin: str = 'd', fname: str = 'lih') -> None:
        """Initializes an ``EHStatesSolver`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            spin: Spin of the second-quantized operators. Either ``'u'`` or ``'d'``. 
            fname: The HDF5 file name.
        """
        assert spin in ['u', 'd']

        self.hamiltonian = hamiltonian.copy()
        
        method_indices_pairs = get_method_indices_pairs(spin)
        tapered_state = [0, 1] if spin == 'u' else [1, 0]
        if REVERSE_QUBIT_ORDER:
            tapered_state.reverse()

        self.hamiltonian.transform(method_indices_pairs, tapered_state=tapered_state)
        
        self.h5fname = fname + ".h5"

        n_qubits = 2 * len(self.hamiltonian.active_indices)
        self.qubit_indices_dict = QubitIndices.get_eh_qubit_indices_dict(
            n_qubits, spin, method_indices_pairs, system_only=True)

        self.energies = dict()
        self.state_vectors = dict()

    def _run_exact(self) -> None:
        """Runs exact calculation of (N±1)-electron states."""
        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        self.energies['e'], self.state_vectors['e'] = eigensolve(
            self.hamiltonian.matrix, inds=self.qubit_indices_dict['e'].int)
        self.energies['h'], self.state_vectors['h'] = eigensolve(
            self.hamiltonian.matrix, inds=self.qubit_indices_dict['h'].int)
        
        print(f"(N+1)-electron energies are {self.energies['e']} eV")
        print(f"(N-1)-electron energies are {self.energies['h']} eV")

    def _save_data(self) -> None:
        """Saves (N±1)-electron energies and state vectors to HDF5 file."""
        h5file = h5py.File(self.h5fname, "r+")
        
        h5file['es/energies_e'] = self.energies['e']
        h5file['es/energies_h'] = self.energies['h']
        h5file['es/states_e'] = self.state_vectors['e']
        h5file['es/states_h'] = self.state_vectors['h']

        h5file.close()

    def run(self) -> None:
        """Runs the (N±1)-electron states calculation."""
        print("Start (N±1)-electron states solver.")
        self._run_exact()
        self._save_data()
        print("(N±1)-electron states solver finished.")
