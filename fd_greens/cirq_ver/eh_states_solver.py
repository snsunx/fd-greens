"""
================================================================
(N±1)-Electron States Solver (:mod:`fd_greens.eh_states_solver`)
================================================================
"""

import h5py
import numpy as np

from .molecular_hamiltonian import MolecularHamiltonian
from .qubit_indices import QubitIndices
from .parameters import Z2TransformInstructions
from .helpers import save_to_hdf5


class EHStatesSolver:
    """(N±1)-electron states solver."""

    def __init__(self, hamiltonian: MolecularHamiltonian, h5fname: str = 'lih') -> None:
        """Initializes an ``EHStatesSolver`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian.
            spin: Spin of the second-quantized operators. Either ``'u'`` or ``'d'``.
            h5fname: The HDF5 file name.
        """
        # self.hamiltonian = hamiltonian.copy()
        # method_indices_pairs = MethodIndicesPairs.get_pairs(spin)
        # tapered_state = [0, 1] if spin == 'u' else [1, 0]

        self.hamiltonian = dict()
        self.qubit_indices = dict()
        for spin in ["u", "d"]:
            # Save the transformed Hamiltonian to self.hamiltonian.
            hamiltonian_ = hamiltonian.copy()
            tapered_state = [0, 1] if spin == 'u' else [1, 0]
            instructions = Z2TransformInstructions.get_instructions(spin)
            hamiltonian_.transform(instructions, tapered_state=tapered_state)
            self.hamiltonian[spin] = hamiltonian_

            # Save the qubit indices to self.qubit_indices.
            n_qubits = 2 * len(hamiltonian_.active_indices)
            qubit_indices = QubitIndices.get_eh_qubit_indices_dict(
                n_qubits, spin, instructions, system_only=True)
            self.qubit_indices[spin] = qubit_indices
            
        # self.hamiltonian.transform(method_indices_pairs, tapered_state=tapered_state)
        self.h5fname = h5fname

        # n_qubits = 2 * len(self.hamiltonian.active_indices)
        # self.qubit_indices_dict = QubitIndices.get_eh_qubit_indices_dict(
        #     n_qubits, spin, method_indices_pairs, system_only=True)

    def _run_exact(self, spin: str) -> None:
        """Runs exact calculation of (N±1)-electron states."""
        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        self.energies['e'], self.state_vectors['e'] = eigensolve(
            self.hamiltonian[spin].matrix, inds=self.qubit_indices[spin]['e'].int)
        self.energies['h'], self.state_vectors['h'] = eigensolve(
            self.hamiltonian[spin].matrix, inds=self.qubit_indices[spin]['h'].int)
        
        print(f"# (N+1)-electron energies are {self.energies['e']} eV")
        print(f"# (N-1)-electron energies are {self.energies['h']} eV")

    def _save_data(self, spin: str) -> None:
        """Saves (N±1)-electron energies and state vectors to HDF5 file."""        
        with h5py.File(self.h5fname + ".h5", 'r+') as h5file:
            save_to_hdf5(h5file, f"es{spin}/energies_e", self.energies['e'])
            save_to_hdf5(h5file, f"es{spin}/energies_h", self.energies['h'])
            save_to_hdf5(h5file, f"es{spin}/states_e", self.state_vectors['e'])
            save_to_hdf5(h5file, f"es{spin}/states_h", self.state_vectors['h'])

    def run(self) -> None:
        """Runs the (N±1)-electron states calculation."""
        print("> Start (N±1)-electron states solver.")
        for spin in ["u", "d"]:
            self.energies = dict()
            self.state_vectors = dict()
            self._run_exact(spin)
            self._save_data(spin)
        print("> (N±1)-electron states solver finished.")
