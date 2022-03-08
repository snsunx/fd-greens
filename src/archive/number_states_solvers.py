"""Number states solver module."""

from typing import Optional

import h5py
import numpy as np

from qiskit import *
from qiskit.utils import QuantumInstance

from hamiltonians import MolecularHamiltonian
from operators import transform_4q_pauli
import params
from ground_state_solvers import vqe_minimize
from ansatze import AnsatzFunction, build_ansatz_e, build_ansatz_h
from utils import state_tomography, get_quantum_instance, write_hdf5
from qubit_indices import transform_4q_indices

class EHStatesSolver:
    """A class to calculate and store information of (N+/-1)-electron states."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 ansatz_func_e: AnsatzFunction = build_ansatz_e, 
                 ansatz_func_h: AnsatzFunction = build_ansatz_h,
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 spin: str = 'd',
                 method: str = 'exact',
                 h5fname: str = 'lih') -> None:
        """Initializes an EHStatesSolver object.
        
        Args:
            h: The molecular Hamiltonian.
            ansatz_func_e: The ansatz function for (N+1)-electron states.
            ansatz_func_h: The ansatz function for (N-1)-electron states.
            q_instance: The quantum instance for (N+/-1)-electron state calculation.
            spin: A string indicating which spin states are included.
            h5fname: The HDF5 file name.
        """
        assert spin in ['u', 'd']
        self.h = h
        init_state = [1, 0] if spin == 'd' else [0, 1]
        self.spin = spin
        self.hspin = 'd' if self.spin == 'u' else 'u'
        self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=init_state)
        self.inds = {'e': transform_4q_indices(params.e_inds[self.spin]),
                     'h': transform_4q_indices(params.h_inds[self.hspin])}

        # print('spin =', self.spin)
        # print('init_state =', init_state)
        # print('inds =', self.inds['e'], self.inds['h'])

        self.ansatz_func_e = ansatz_func_e
        self.ansatz_func_h = ansatz_func_h
        self.q_instance = q_instance
        self.method = method
        self.h5fname = h5fname + '.h5'

        self.energies_e = None
        self.energies_h = None
        self.states_e = None
        self.states_h = None

    def run_exact(self) -> None:
        """Calculates the exact (N+/-1)-electron energies and states of the Hamiltonian."""
        def eigensolve(arr, inds):
            # print('arr =', arr)
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v
        h_arr = self.h_op.to_matrix()
        self.energies_e, self.states_e = eigensolve(h_arr, inds=self.inds['e']._int)
        self.energies_h, self.states_h = eigensolve(h_arr, inds=self.inds['h']._int)
        # self.states_e = self.states_e
        # self.states_h = self.states_h
        print(f"(N+1)-electron energies are {self.energies_e} eV")
        print(f"(N-1)-electron energies are {self.energies_h} eV")

    def run_vqe(self) -> None:
        """Calculates the (N+/-1)-electron energies and states of the Hamiltonian using VQE."""
        energy_min, circ_min = vqe_minimize(self.h_op, self.ansatz_func_e, (0.,), self.q_instance)
        minus_energy_max, circ_max = vqe_minimize(-self.h_op, self.ansatz_func_e, (0.,), self.q_instance)
        state_min = state_tomography(circ_min, q_instance=self.q_instance)
        state_max = state_tomography(circ_max, q_instance=self.q_instance)
        self.energies_e = np.array([energy_min, -minus_energy_max])
        self.states_e = [self.inds['e'](state_min), self.inds['e'](state_max)]
        # inds_e = self.inds['e'].int_form        
        # self.states_e = [state_min[[inds_e][:, inds_e]], state_max[[inds_e][:, inds_e]]]

        energy_min, circ_min = vqe_minimize(self.h_op, self.ansatz_func_h, (0.,), self.q_instance)
        minus_energy_max, circ_max = vqe_minimize(-self.h_op, self.ansatz_func_h, (0.,), self.q_instance)
        state_min = state_tomography(circ_min, q_instance=self.q_instance)
        state_max = state_tomography(circ_max, q_instance=self.q_instance)
        self.energies_h = np.array([energy_min, -minus_energy_max])
        self.states_h = [self.inds['h'](state_min), self.inds['h'](state_max)]
        # inds_h = self.inds['h'].int_form
        # self.states_h = [state_min[[inds_h][:, inds_h]], state_max[[inds_h][:, inds_h]]]

        print(f"(N+1)-electron energies are {self.energies_e} eV")
        print(f"(N-1)-electron energies are {self.energies_h} eV")

    def save_data(self) -> None:
        """Saves (N+/-1)-electron energies and states to HDF5 file."""
        h5file = h5py.File(self.h5fname, 'r+')

        write_hdf5(h5file, 'es', 'energies_e', self.energies_e)
        write_hdf5(h5file, 'es', 'energies_h', self.energies_h)
        write_hdf5(h5file, 'es', 'states_e', self.states_e)
        write_hdf5(h5file, 'es', 'states_h', self.states_h)

        h5file.close()

    def run(self, method: Optional[str] = None) -> None:
        """Runs the (N+/-1)-electron states calculation."""
        if method is not None:
            self.method = method
        if self.method == 'exact': 
            self.run_exact()
        elif self.method == 'vqe': 
            self.run_vqe()
        self.save_data()

class ExcitedStatesSolver:
    """A class to calculate and store information of excited states."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 ansatz_func_e: Optional[AnsatzFunction] = None, 
                 ansatz_func_h: Optional[AnsatzFunction] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 apply_tomography: bool = False,
                 h5fname: str = 'lih'):
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

        self.inds_s = transform_4q_indices(params.singlet_inds)
        self.inds_t = transform_4q_indices(params.triplet_inds)

        self.h5fname = h5fname + '.h5'

    def _run_exact(self):

        def eigensolve(arr, inds):
            arr = arr[inds][:, inds]
            e, v = np.linalg.eigh(arr)
            return e, v

        self.energies_s, self.states_s = eigensolve(self.h_mat_s, inds=self.inds_s.int_form)
        self.energies_t, self.states_t = eigensolve(self.h_mat_t, inds=self.inds_t.int_form)
        self.states_s = self.states_s.T
        self.states_t = self.states_t.T
        print(f"Singlet excited-state energies are {self.energies_s} eV")
        print(f"Triplet excited state energies are {self.energies_t} eV")

    def save_data(self):
        """Saves N-electron excited-state energies and states to hdf5 file."""
        h5file = h5py.File(self.h5fname, 'r+')

        write_hdf5(h5file, 'es', 'energies_s', self.energies_s)
        write_hdf5(h5file, 'es', 'energies_t', self.energies_t)
        write_hdf5(h5file, 'es', 'states_s', self.states_s)
        write_hdf5(h5file, 'es', 'states_t', self.states_t)

        h5file.close()

    def run(self, method='exact'):
        """Runs the excited states calculation."""
        if method == 'exact':
            self._run_exact()
        elif method == 'vqe':
            raise NotImplementedError('VQE calculation of excited states is not implemented')
        self.save_data()