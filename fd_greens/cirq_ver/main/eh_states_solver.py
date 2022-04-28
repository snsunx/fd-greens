"""
=====================================================================
(N±1)-Electron States Solver (:mod:`fd_greens.main.eh_states_solver`)
=====================================================================
"""
from typing import Optional

import h5py
import numpy as np

from qiskit import Aer
from qiskit.utils import QuantumInstance

from .molecular_hamiltonian import MolecularHamiltonian
from .z2symmetries import transform_4q_pauli, transform_4q_indices
from .qubit_indices import e_inds, h_inds
from ..utils import write_hdf5


class EHStatesSolver:
    """A class to calculate and store information of (N±1)-electron states."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        q_instance: Optional[QuantumInstance] = None,
        spin: str = "d",
        h5fname: str = "lih",
    ) -> None:
        """Initializes an ``EHStatesSolver`` object.
        
        Args:
            h: The molecular Hamiltonian.
            q_instance: The quantum instance for (N±1)-electron state calculation.
            spin: A character indicating whether up spins (``'u'``) or down spins (``'d'``) are included.
            h5fname: The HDF5 file name.
        """
        assert spin in ["u", "d"]

        self.h = h
        init_state = [1, 0] if spin == "d" else [0, 1]
        self.spin = spin
        self.hspin = "d" if self.spin == "u" else "u"
        self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=init_state)
        self.inds = {
            "e": transform_4q_indices(e_inds[self.spin]),
            "h": transform_4q_indices(h_inds[self.hspin]),
        }

        if q_instance is None:
            self.q_instance = QuantumInstance(Aer.get_backend("statevector_simulator"))
        else:
            self.q_instance = q_instance
        self.h5fname = h5fname + ".h5"

        self.energies_e = None
        self.energies_h = None
        self.states_e = None
        self.states_h = None

    def _run_exact(self) -> None:
        """Calculates the exact (N±1)-electron energies and states of the Hamiltonian."""

        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        h_arr = self.h_op.to_matrix()
        self.energies_e, self.states_e = eigensolve(h_arr, inds=self.inds["e"]._int)
        self.energies_h, self.states_h = eigensolve(h_arr, inds=self.inds["h"]._int)
        # self.states_e = self.states_e
        # self.states_h = self.states_h
        print(f"(N+1)-electron energies are {self.energies_e} eV")
        print(f"(N-1)-electron energies are {self.energies_h} eV")

    def _save_data(self) -> None:
        """Saves (N±1)-electron energies and states to HDF5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        write_hdf5(h5file, "es", "energies_e", self.energies_e)
        write_hdf5(h5file, "es", "energies_h", self.energies_h)
        write_hdf5(h5file, "es", "states_e", self.states_e)
        write_hdf5(h5file, "es", "states_h", self.states_h)

        h5file.close()

    def run(self) -> None:
        """Runs the (N±1)-electron states calculation."""
        self._run_exact()
        self._save_data()
