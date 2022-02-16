"""Amplitudes solver module."""

from typing import Iterable, Optional, Sequence
from functools import partial
from collections import defaultdict

import h5py
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance

import params
from hamiltonians import MolecularHamiltonian
from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from operators import SecondQuantizedOperators, ChargeOperators, transform_4q_pauli
from qubit_indices import QubitIndices, transform_4q_indices
from circuits import (CircuitConstructor, InstructionTuple, append_tomography_gates,
                      append_measurement_gates, transpile_into_berkeley_gates)
from utils import get_overlap, get_quantum_instance, counts_dict_to_arr, write_hdf5


class ExcitedAmplitudesSolver:
    """A class to calculate transition amplitudes between the ground state and the excited states."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 method: str = 'exact',
                 h5fname: str = 'lih',
                 suffix: str = '') -> None:
        """Initializes an ExcitedAmplitudesSolver object.

        Args:
            h: The molecular Hamiltonian.
            q_instance: The quantum instance for executing the circuits.
            method: The method for extracting the transition amplitudes.
            h5fname: The HDF5 file name.
            anc: Location of the ancilla qubits.
        """
        assert method in ['exact', 'tomo']

        # Basic variables
        self.h = h
        self.q_instance = q_instance
        self.method = method
        self.suffix = suffix

        # Load data and initialize quantities
        self.h5fname = h5fname + '.h5'
        self._initialize()


    def _initialize(self) -> None:
        """Initializes physical quantity attributes."""
        h5file = h5py.File(self.h5fname, 'r+')

        # Occupied and active orbital indices
        self.occ_inds = self.h.occ_inds
        self.act_inds = self.h.act_inds

        self.qinds = transform_4q_indices(params.singlet_inds)
        self.qinds_tot = self.qinds.insert_ancilla([0, 1])

        # Number of spatial orbitals and (N+/-1)-electron states
        self.n_elec = self.h.molecule.n_electrons
        self.ansatz = QuantumCircuit.from_qasm_str(h5file['gs/ansatz'][()].decode())
        self.energies = h5file['es/energies_s']
        self.states = h5file['es/states_s'][:]
        self.n_states = len(self.energies)
        self.n_orb = 2
        #self.n_occ = self.n_elec // 2 - len(self.occ_inds)
        #self.n_vir = self.n_orb - self.n_occ

        # Transition amplitude arrays
        self.L = np.zeros((self.n_orb, self.n_orb, self.n_states), dtype=complex)
    
        # Create Pauli dictionaries for operators after symmetry transformation.
        charge_ops = ChargeOperators(self.n_elec)
        charge_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = charge_ops.get_pauli_dict()

        # The circuit constructor.
        self.constructor = CircuitConstructor(self.ansatz)

    def build_diagonal(self) -> None:
        """Calculates diagonal transition amplitudes."""
        print("----- Calculating diagonal transition amplitudes -----")
        h5file = h5py.File(self.h5fname, 'r+')
        
        for m in range(self.n_orb): # 0, 1
            U_op = self.pauli_dict[(m, 'd')]
            circ = self.constructor.build_charge_diagonal(U_op)
            circ = transpile_into_berkeley_gates(circ, 'something')
            write_hdf5(h5file, f'circ{m}', 'base', circ.qasm())

            if self.method == 'tomo':
                pass

        h5file.close()

    def run_diagonal(self) -> None:
        """Executes the diagonal circuits circ0 and circ1."""
        h5file = h5py.File(self.h5fname, 'r+')
        for m in range(self.n_orb): # 0, 1
            if self.method == 'exact':
                dset = h5file[f'circ{m}/base']
                circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                result = self.q_instance.execute(circ)
                psi = result.get_statevector()
                dset.attrs[f'psi{self.suffix}'] = psi
            elif self.method == 'tomo':
                pass

        h5file.close()

    def process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitudes circuits."""
        h5file = h5py.File(self.h5fname, 'r+')
        for m in range(self.n_orb): # 0, 1
            if self.method == 'exact':
                psi = h5file[f'circ{m}/base'].attrs[f'psi{self.suffix}']
                psi = self.qinds_tot(psi)
                self.L[m, m] = np.abs(self.states.conj().T @ psi)
                print(f'L[{m}, {m}] = {self.L[m, m]}')
            elif self.method == 'tomo':
                pass
        h5file.close()

    def run(self, method=None) -> None:
        """Runs the functions to compute transition amplitudes."""
        if method is not None:
            self.method = method
        self.build_diagonal()
        self.run_diagonal()
        self.process_diagonal()