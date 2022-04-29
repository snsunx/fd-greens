"""
======================================================================
N-Electron States Solver (:mod:`fd_greens.main.excited_states_solver`)
======================================================================
"""

from typing import Optional

import h5py
import numpy as np

from qiskit import *
from qiskit.utils import QuantumInstance

from .molecular_hamiltonian import MolecularHamiltonian
from .z2symmetries import transform_4q_pauli, transform_4q_indices

# from .params import singlet_inds, triplet_inds
from .ansatze import AnsatzFunction, build_ansatz_e, build_ansatz_h
from ..utils import write_hdf5, vqe_minimize


class ExcitedStatesSolver:
    """A class to calculate and store information of excited states."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        ansatz_func_e: Optional[AnsatzFunction] = None,
        ansatz_func_h: Optional[AnsatzFunction] = None,
        q_instance: Optional[QuantumInstance] = None,
        apply_tomography: bool = False,
        h5fname: str = "lih",
    ):
        """Initializes a EHStatesSolver object.
        
        Args:
            h: The molecular Hamiltonian.
            ansatz_func_e: The ansatz function for N+1 electron states.
            ansatz_func_h: The ansatz function for N-1 electron states.
            q_instance: The QuantumInstance object for N+/-1 electron state calculation.
            apply_tomography: Whether tomography of the states is applied.
            h5fname: The HDF5 file name.
        """
        self.h = h
        self.h_op = self.h.qiskit_op
        self.h_op_s = transform_4q_pauli(self.h_op, init_state=[1, 1])
        self.h_op_t = transform_4q_pauli(self.h_op, init_state=[0, 0])
        self.h_mat_s = self.h_op_s.to_matrix()
        self.h_mat_t = self.h_op_t.to_matrix()

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

    def save_data(self):
        """Saves N-electron excited-state energies and states to hdf5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        write_hdf5(h5file, "es", "energies_s", self.energies_s)
        write_hdf5(h5file, "es", "energies_t", self.energies_t)
        write_hdf5(h5file, "es", "states_s", self.states_s)
        write_hdf5(h5file, "es", "states_t", self.states_t)

        h5file.close()

    def run(self, method: str = "exact") -> None:
        """Runs the excited state calculation.
        
        Args:
            method: The method to calculate the excited states. Either exact ('exact') or VQE ('vqe').
        """
        if method == "exact":
            self._run_exact()
        elif method == "vqe":
            raise NotImplementedError(
                "VQE calculation of excited states is not implemented"
            )
        self.save_data()
