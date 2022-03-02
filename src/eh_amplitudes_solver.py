"""Amplitudes solver module."""

from typing import Optional, Sequence
from functools import partial
from collections import defaultdict

import itertools
import h5py
import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
import params
from hamiltonians import MolecularHamiltonian
from operators import SecondQuantizedOperators, transform_4q_pauli
from qubit_indices import transform_4q_indices
from circuits import (CircuitConstructor, append_tomography_gates, append_tomography_gates1,
                      append_measurement_gates, append_measurement_gates1, transpile_into_berkeley_gates,
                      replace_with_cixc)
from utils import get_overlap, get_quantum_instance, counts_dict_to_arr, write_hdf5, circuit_to_qasm_str

np.set_printoptions(precision=6)

class EHAmplitudesSolver:
    """A class to calculate transition amplitudes."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 method: str = 'exact',
                 h5fname: str = 'lih',
                 anc: Sequence[int] = [0, 1],
                 spin: str = 'd',
                 suffix: str = '',
                 verbose: bool = True,
                 overwrite: bool = False) -> None:
        """Initializes an EHAmplitudesSolver object.

        Args:
            h: The molecular Hamiltonian.
            q_instance: The quantum instance for executing the circuits.
            method: The method for extracting the transition amplitudes.
            h5fname: The HDF5 file name.
            anc: Location of the ancilla qubits.
            spin: The spin of the creation and annihilation operators.
            suffix: The suffix for a specific experimental run.
            verbose: Whether to print out information about the calculation.
            overwrite: Whether circuits are overwritten.
        """
        assert method in ['exact', 'tomo']
        assert spin in ['u', 'd']

        # Basic variables.
        self.h = h
        self.q_instance = q_instance
        self.method = method
        self.anc = anc
        self.sys = [i for i in range(4) if i not in anc]
        self.spin = spin
        self.hspin = 'd' if self.spin == 'u' else 'u'
        self.suffix = suffix
        self.verbose = verbose

        # Hardcoded problem parameters.
        self.n_anc_diag = 1
        self.n_anc_off_diag = 2
        self.n_sys = 2

        # Load data and initialize quantities.
        self.h5fname = h5fname + '.h5'
        self._initialize()

    def _initialize(self) -> None:
        """Loads ground state and (N+/-1)-electron states data from hdf5 file."""
        # Attributes from ground and (N+/-1)-electron state solver.
        h5file = h5py.File(self.h5fname, 'r+')
        self.ansatz = QuantumCircuit.from_qasm_str(h5file['gs/ansatz'][()].decode())
        self.states = {'e': h5file['es/states_e'][:], 'h': h5file['es/states_h'][:]}
        h5file.close()

        # Number of spatial orbitals and (N+/-1)-electron states.
        self.n_states = {'e': self.states['e'].shape[1], 'h': self.states['h'].shape[1]}
        self.n_elec = self.h.molecule.n_electrons
        self.n_orb = len(self.h.act_inds)
        self.n_occ = self.n_elec // 2 - len(self.h.occ_inds)
        self.n_vir = self.n_orb - self.n_occ
        self.n_e = int(binom(2 * self.n_orb, 2 * self.n_occ + 1)) // 2
        self.n_h = int(binom(2 * self.n_orb, 2 * self.n_occ - 1)) // 2

        # Build qubit indices for system qubits.
        self.qinds_sys = {'e': transform_4q_indices(params.e_inds[self.spin]),
                          'h': transform_4q_indices(params.h_inds[self.hspin])}

        # Build qubit indices for diagonal circuits.
        self.keys_diag = ['e', 'h']
        self.qinds_anc_diag = [[1], [0]]
        self.qinds_tot_diag = dict()
        for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
            self.qinds_tot_diag[key] = self.qinds_sys[key].insert_ancilla(qind)
            # print('qinds_tot_diag', key, self.qinds_tot_diag[key])
        
        # Build qubit indices for off-diagonal circuits.
        self.keys_off_diag = ['ep', 'em', 'hp', 'hm']
        self.qinds_anc_off_diag = [[1, 0], [1, 1], [0, 0], [0, 1]]
        self.qinds_tot_off_diag = dict()
        for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
            self.qinds_tot_off_diag[key] = self.qinds_sys[key[0]].insert_ancilla(qind)

        # Transition amplitude arrays. The keys of B are 'e' and 'h'. 
        # The keys of D are 'ep', 'em', 'hp', 'hm'.
        assert self.n_e == self.n_h # XXX: This is only for this special case
        self.B = {key: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex) for key in self.keys_diag}
        self.D = {key: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex) for key in self.keys_off_diag}

        # Create Pauli dictionaries for the creation/annihilation operators.
        second_q_ops = SecondQuantizedOperators(self.h.molecule.n_electrons)
        second_q_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = second_q_ops.get_pauli_dict()
        # for key, val in self.pauli_dict.items():
        #     print(key, val.coeffs, val.table.to_labels())

        # The circuit constructor and tomography labels.
        self.constructor = CircuitConstructor(self.ansatz, anc=self.anc)
        self.tomo_labels = [''.join(x) for x in itertools.product('xyz', repeat=self.n_sys)]

        # print("----- Printing out physical quantities -----")
        # print(f"Number of electrons is {self.n_elec}")
        # print(f"Number of orbitals is {self.n_orb}")
        # print(f"Number of occupied orbitals is {self.n_occ}")
        # print(f"Number of virtual orbitals is {self.n_vir}")
        # print(f"Number of (N+1)-electron states is {self.n_e}")
        # print(f"Number of (N-1)-electron states is {self.n_h}")
        # print("--------------------------------------------")

    def build_diagonal(self) -> None:
        """Constructs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')
        
        for m in range(self.n_orb): # 0, 1
            # Build the circuit based on the creation/annihilation operator 
            # and store the QASM string in the HDF5 file.
            a_op = self.pauli_dict[(m, self.spin)]
            circ = self.constructor.build_eh_diagonal(a_op)
            circ = transpile_into_berkeley_gates(circ, str(m) + self.spin)
            write_hdf5(h5file, f'circ{m}{self.spin}', 'base', circuit_to_qasm_str(circ))

            if self.method == 'tomo':
                for label in self.tomo_labels:
                    # When using tomography, build circuits with tomography and measurement
                    # gates appended and store the QASM string in the HDF5 file.
                    tomo_circ = append_tomography_gates(circ, [1, 2], label)
                    tomo_circ = append_measurement_gates(tomo_circ)
                    write_hdf5(h5file, f'circ{m}{self.spin}', label, circuit_to_qasm_str(tomo_circ))

        h5file.close()

    def execute_diagonal(self) -> None:
        """Executes the diagonal circuits circ0 and circ1."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb): # 0, 1
            if self.method == 'exact':
                # Extract the quantum circuit from the HDF5 file.
                dset = h5file[f'circ{m}{self.spin}/base']
                circ = QuantumCircuit.from_qasm_str(dset[()].decode())
                # circ = replace_with_cixc(circ)
                
                # Execute the circuit and store the statevector in the HDF5 file.
                result = self.q_instance.execute(circ)
                psi = result.get_statevector()
                dset.attrs[f'psi{self.suffix}'] = psi
            else: # Tomography
                for label in self.tomo_labels:
                    # Extract the quantum circuit from the HDF5 file.
                    dset = h5file[f'circ{m}{self.spin}/{label}']
                    circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                    # Execute the circuit and store the counts array in the HDF5 file.
                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    counts_arr = counts_dict_to_arr(counts.int_raw, n_qubits=3)
                    dset.attrs[f'counts{self.suffix}'] = counts_arr

        h5file.close()

    run_diagonal = execute_diagonal

    def process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb): # 0, 1
            if self.method == 'exact':
                psi = h5file[f'circ{m}{self.spin}/base'].attrs[f'psi{self.suffix}']
                for key in self.keys_diag:
                    # print('key =', key)
                    psi[abs(psi) < 1e-8] = 0.
                    # print('psi =', psi)
                    psi_key = self.qinds_tot_diag[key](psi)
                    # print('psi_key =', psi_key)
                    # print('states[key] =', self.states[key])

                    # Obtain the B matrix elements by computing the overlaps.
                    self.B[key][m, m] = np.abs(self.states[key].conj().T @ psi_key) ** 2
                    if self.verbose: print(f'B[{key}][{m}, {m}] = {self.B[key][m, m]}')

            elif self.method == 'tomo':
                for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
                    # Stack counts_arr over all tomography labels together. The procedure is to 
                    # first extract the raw counts_arr, slice the counts array to the specific 
                    # label, and then stack the counts_arr_label to counts_arr_key.
                    counts_arr_key = np.array([])
                    for label in self.tomo_labels:
                        counts_arr = h5file[f'circ{m}{self.spin}/{label}'].attrs[f'counts{self.suffix}']
                        start = int(''.join([str(i) for i in qind])[::-1], 2)
                        counts_arr_label = counts_arr[start::2**self.n_anc_diag]
                        counts_arr_label = counts_arr_label / np.sum(counts_arr)
                        counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))
                    
                    # Obtain the density matrix from tomography. Slice the density matrix 
                    # based on whether we are considering 'e' or 'h' on the system qubits.
                    rho = np.linalg.lstsq(params.basis_matrix, counts_arr_key)[0]
                    rho = rho.reshape(2**self.n_sys, 2**self.n_sys, order='F')
                    rho = self.qinds_sys[key](rho)

                    # Obtain the B matrix elements by computing the overlaps between 
                    # the density matrix and the states from EHStatesSolver.
                    self.B[key][m, m] = [get_overlap(self.states[key][:, i], rho) 
                                         for i in range(self.n_states[key])]
                    if self.verbose: print(f'B[{key}][{m}, {m}] = {self.B[key][m, m]}')

        h5file.close()

    def build_off_diagonal(self) -> None:
        """Constructs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')

        a_op_0 = self.pauli_dict[(0, self.spin)]
        a_op_1 = self.pauli_dict[(1, self.spin)]
        if self.spin == 'u':
            circ = self.constructor.build_eh_off_diagonal(a_op_1, a_op_0)
        else:
            circ = self.constructor.build_eh_off_diagonal(a_op_0, a_op_1)
        circ = transpile_into_berkeley_gates(circ, '01' + self.spin)
        write_hdf5(h5file, f'circ01{self.spin}', 'base', circuit_to_qasm_str(circ))

        if self.method == 'tomo':
            for label in self.tomo_labels:
                tomo_circ = append_tomography_gates(circ, self.sys, label)
                tomo_circ = append_measurement_gates(tomo_circ)
                write_hdf5(h5file, f'circ01{self.spin}', label, circuit_to_qasm_str(tomo_circ))

        h5file.close()

    def execute_off_diagonal(self) -> None:
        """Executes the off-diagonal transition amplitude circuit circ01."""
        h5file = h5py.File(self.h5fname, 'r+')

        if self.method == 'exact':
            # Extract the quantum circuit from the HDF5 file.
            dset = h5file[f'circ01{self.spin}/base']
            circ = QuantumCircuit.from_qasm_str(dset[()].decode())
            # circ = replace_with_cixc(circ)

            result = self.q_instance.execute(circ)
            psi = result.get_statevector()
            dset.attrs[f'psi{self.suffix}'] = psi
        else: # Tomography
            for label in self.tomo_labels:
                # Extract the quantum circuit from the HDF5 file.
                dset = h5file[f'circ01{self.spin}/{label}']
                circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                result = self.q_instance.execute(circ)
                counts = result.get_counts()
                counts_arr = counts_dict_to_arr(counts.int_raw)
                dset.attrs[f'counts{self.suffix}'] = counts_arr

        h5file.close()

    run_off_diagonal = execute_off_diagonal

    def process_off_diagonal(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        if self.method == 'exact':
            psi = h5file[f'circ01{self.spin}/base'].attrs[f'psi{self.suffix}']
            for key in self.keys_off_diag:
                psi_key = self.qinds_tot_off_diag[key](psi)

                # Obtain the D matrix elements by computing the overlaps.
                self.D[key][0, 1] = self.D[key][1, 0] = abs(self.states[key[0]].conj().T @ psi_key) ** 2
                if self.verbose: print(f'D[{key}][0, 1] =', self.D[key][0, 1])
                    
        elif self.method == 'tomo':
            for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
                # Stack counts_arr over all tomography labels together. The procedure is to 
                # first extract the raw counts_arr, slice the counts array to the specific 
                # label, and then stack the counts_arr_label to counts_arr_key.
                counts_arr_key = np.array([])
                for label in self.tomo_labels:
                    counts_arr = h5file[f'circ01{self.spin}/{label}'].attrs[f'counts{self.suffix}']
                    start = int(''.join([str(i) for i in qind])[::-1], 2)
                    counts_arr_label = counts_arr[start::2**self.n_anc_off_diag] # 4 is because 2 ** 2
                    counts_arr_label = counts_arr_label / np.sum(counts_arr)
                    counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))

                # Obtain the density matrix from tomography. Slice the density matrix 
                # based on whether we are considering 'e' or 'h' on the system qubits.
                rho = np.linalg.lstsq(params.basis_matrix, counts_arr_key)[0]
                rho = rho.reshape(2**self.n_sys, 2**self.n_sys, order='F')
                rho = self.qinds_sys[key[0]](rho)

                # Obtain the D matrix elements by computing the overlaps between 
                # the density matrix and the states from EHStatesSolver.
                self.D[key][0, 1] = self.D[key][1, 0] = \
                    [get_overlap(self.states[key[0]][:, i], rho) 
                    for i in range(self.n_states[key[0]])]
                if self.verbose: print(f'D[{key}][0, 1] =', self.D[key][0, 1])

        # Unpack D values to B values based on Eq. (18) of Kosugi and Matsushita 2020.
        for key in self.keys_diag:
            self.B[key][0, 1] = self.B[key][1, 0] = \
                np.exp(-1j * np.pi/4) * (self.D[key+'p'][0, 1] - self.D[key+'m'][0, 1]) \
                + np.exp(1j * np.pi/4) * (self.D[key+'p'][1, 0] - self.D[key+'m'][1, 0])
            if self.verbose: print(f'B[{key}][0, 1] =', self.B[key][0, 1])

        h5file.close()
        
    def save_data(self) -> None:
        """Saves transition amplitudes data to HDF5 file."""
        h5file = h5py.File(self.h5fname, 'r+')

        # Saves B_e and B_h.
        write_hdf5(h5file, 'amp', f'B_e{self.suffix}', self.B['e'])
        write_hdf5(h5file, 'amp', f'B_h{self.suffix}', self.B['h'])

        # Saves orbital energies for calculating the self-energy.
        e_orb = np.diag(self.h.molecule.orbital_energies)
        act_inds = self.h.act_inds
        write_hdf5(h5file, 'amp', f'e_orb{self.suffix}', e_orb[act_inds][:, act_inds])

        h5file.close()

    def run(self,
            method: Optional[str] = None, 
            build: bool = True,
            execute: bool = True,
            process: bool = True) -> None:
        """Executes all functions to calculate transition amplitudes."""
        if method is not None: 
            self.method = method
        if build:
            self.build_diagonal()
            self.build_off_diagonal()
        if execute:
            self.execute_diagonal()
            self.execute_off_diagonal()
        if process:
            self.process_diagonal()
            self.process_off_diagonal()
            self.save_data()