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
from circuits import (CircuitConstructor, append_tomography_gates,
                      append_measurement_gates, transpile_into_berkeley_gates)
from utils import get_overlap, get_quantum_instance, counts_dict_to_arr, write_hdf5

np.set_printoptions(precision=6)

class EHAmplitudesSolver:
    """A class to calculate transition amplitudes."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 method: str = 'exact',
                 h5fname: str = 'lih',
                 anc: Sequence[int] = [0, 2],
                 suffix: str = '') -> None:
        """Initializes an EHAmplitudesSolver object.

        Args:
            h: The molecular Hamiltonian.
            method: The method for extracting the transition amplitudes.
            q_instance: The quantum instance for executing the circuits.
            h5fname: The HDF5 file name.
            anc: Location of the ancilla qubits.
        """
        assert method in ['exact', 'tomo']

        # Basic variables
        self.h = h
        self.q_instance = q_instance
        self.method = method
        self.anc = anc
        self.sys = [i for i in range(4) if i not in anc]
        self.suffix = suffix

        # Load data and initialize quantities
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
        self.n_elec = self.h.molecule.n_electrons
        self.n_orb = len(self.h.act_inds)
        self.n_occ = self.n_elec // 2 - len(self.h.occ_inds)
        self.n_vir = self.n_orb - self.n_occ
        self.n_e = int(binom(2 * self.n_orb, 2 * self.n_occ + 1)) // 2
        self.n_h = int(binom(2 * self.n_orb, 2 * self.n_occ - 1)) // 2

        self.qinds_sys = {'e': transform_4q_indices(params.ed_inds),
                          'h': transform_4q_indices(params.hu_inds)}
        
        # Build qubit indices for diagonal circuits.
        self.keys_diag = ['e', 'h']
        self.qinds_anc_diag = [[1], [0]]
        self.qinds_tot_diag = dict()
        for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
            self.qinds_tot_diag[key] = self.qinds_sys[key].insert_ancilla(qind, loc=self.anc)
        
        # Build qubit indices for off-diagonal circuits.
        self.keys_off_diag = ['ep', 'em', 'hp', 'hm']
        self.qinds_anc_off_diag = [[1, 0], [1, 1], [0, 0], [0, 1]]
        self.qinds_tot_off_diag = dict()
        for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
            self.qinds_tot_off_diag[key] = self.qinds_sys[key[0]].insert_ancilla(qind, loc=self.anc)

        # Transition amplitude arrays. The keys of B will be 'e' and 'h'. 
        # The keys of D will be 'ep', 'em', 'hp', 'hm'.
        assert self.n_e == self.n_h # XXX: This is only for this special case
        self.B = defaultdict(lambda: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex))
        self.D = defaultdict(lambda: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex))
    
        # Create Pauli dictionaries for the creation/annihilation operators.
        second_q_ops = SecondQuantizedOperators(self.h.molecule.n_electrons)
        second_q_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = second_q_ops.get_pauli_dict()

        # The circuit constructor.
        self.constructor = CircuitConstructor(self.ansatz, anc=self.anc)

        print("----- Printing out physical quantities -----")
        print(f"Number of electrons is {self.n_elec}")
        print(f"Number of orbitals is {self.n_orb}")
        print(f"Number of occupied orbitals is {self.n_occ}")
        print(f"Number of virtual orbitals is {self.n_vir}")
        print(f"Number of (N+1)-electron states is {self.n_e}")
        print(f"Number of (N-1)-electron states is {self.n_h}")
        print("--------------------------------------------")

    def build_diagonal(self) -> None:
        """Constructs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')
        
        for m in range(self.n_orb): # 0, 1
            a_op = self.pauli_dict[(m, 'd')]
            circ = self.constructor.build_eh_diagonal(a_op)
            circ = transpile_into_berkeley_gates(circ, str(m))
            write_hdf5(h5file, f'circ{m}', 'base', circ.qasm())

            if self.method == 'tomo':
                labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
                for label in labels:
                    tomo_circ = append_tomography_gates(circ, [1, 2], label)
                    tomo_circ = append_measurement_gates(tomo_circ)
                    write_hdf5(h5file, f'circ{m}', label, tomo_circ.qasm())

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
            else:
                labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
                for label in labels:
                    dset = h5file[f'circ{m}/{label}']
                    circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    counts_arr = counts_dict_to_arr(counts.int_raw, n_qubits=3)
                    dset.attrs[f'counts{self.suffix}'] = counts_arr

        h5file.close()

    def process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb): # 0, 1
            if self.method == 'exact':
                psi = h5file[f'circ{m}/base'].attrs[f'psi{self.suffix}']
                for key in self.keys_diag:
                    psi_key = self.qinds_tot_diag[key](psi)
                    self.B[key][m, m] = np.abs(self.states[key].conj().T @ psi_key) ** 2
                    print(f'B[{key}][{m}, {m}] = {self.B[key][m, m]}')

            elif self.method == 'tomo':
                basis_matrix = params.basis_matrix

                labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
                for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
                    # Stack counts_arr over all tomography labels together.
                    counts_arr_key = np.array([])
                    for label in labels:
                        counts_arr = h5file[f'circ{m}/{label}'].attrs[f'counts{self.suffix}']
                        # print('np.sum(counts_arr) =', np.sum(counts_arr))
                        start = int(''.join([str(i) for i in qind])[::-1], 2)
                        counts_arr_label = counts_arr[start::2] # 2 is because 2 ** 1
                        counts_arr_label = counts_arr_label / np.sum(counts_arr)
                        counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))
                        # print('np.sum(counts_arr) =', np.sum(counts_arr))
                        # print('np.sum(counts_arr_label) =', np.sum(counts_arr_label))
                    
                    # Obtain the density matrix from tomography.
                    rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0].reshape(4, 4, order='F')
                    rho = self.qinds_sys[key](rho)

                    self.B[key][m, m] = [get_overlap(self.states[key][:, i], rho) for i in range(2)]
                    print(f'B[{key}][{m}, {m}] = {self.B[key][m, m]}')

        h5file.close()

    def build_off_diagonal(self) -> None:
        """Constructs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')

        a_op_0 = self.pauli_dict[(0, 'd')]
        a_op_1 = self.pauli_dict[(1, 'd')]
        circ = self.constructor.build_eh_off_diagonal(a_op_1, a_op_0)
        circ = transpile_into_berkeley_gates(circ, '01')
        write_hdf5(h5file, 'circ01', 'base', circ.qasm())

        if self.method == 'tomo':
            labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
            for label in labels:
                tomo_circ = append_tomography_gates(circ, self.sys, label)
                tomo_circ = append_measurement_gates(tomo_circ)
                write_hdf5(h5file, 'circ01', label, tomo_circ.qasm())

        h5file.close()

    def run_off_diagonal(self) -> None:
        """Executes the off-diagonal transition amplitude circuit circ01."""
        h5file = h5py.File(self.h5fname, 'r+')

        if self.method == 'exact':
            dset = h5file[f'circ01/base']
            circ = QuantumCircuit.from_qasm_str(dset[()].decode())

            result = self.q_instance.execute(circ)
            psi = result.get_statevector()
            dset.attrs[f'psi{self.suffix}'] = psi
        else:
            labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
            for label in labels:
                dset = h5file[f'circ01/{label}']
                circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                result = self.q_instance.execute(circ)
                counts = result.get_counts()
                counts_arr = counts_dict_to_arr(counts.int_raw)
                dset.attrs[f'counts{self.suffix}'] = counts_arr

        h5file.close()

    def process_off_diagonal(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        if self.method == 'exact':
            psi = h5file[f'circ01/base'].attrs[f'psi{self.suffix}']
            for key in self.keys_off_diag:
                psi_key = self.qinds_tot_off_diag[key](psi)
                self.D[key][0, 1] = self.D[key][1, 0] = \
                    abs(self.states[key[0]].conj().T @ psi_key) ** 2
                print(f'self.D[{key}][0, 1] =', self.D[key][0, 1])
                    
        elif self.method == 'tomo':
            basis_matrix = params.basis_matrix
            labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
            for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
                counts_arr_key = np.array([])
                for label in labels:
                    counts_arr = h5file[f'circ01/{label}'].attrs[f'counts{self.suffix}']
                    start = int(''.join([str(i) for i in qind])[::-1], 2)
                    counts_arr_label = counts_arr[start::4] # 4 is because 2 ** 2
                    counts_arr_label = counts_arr_label / np.sum(counts_arr)
                    # assert np.sum(counts_arr) == 10000
                    counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))

                rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0].reshape(4, 4, order='F')
                rho = self.qinds_sys[key[0]](rho)
                self.D[key][0, 1] = self.D[key][1, 0] = \
                    [get_overlap(self.states[key[0]][:, i], rho) for i in range(2)]
                print(f'D[{key}][0, 1] =', self.D[key][0, 1])


        # Unpack D values to B values.
        for key in self.keys_diag:
            self.B[key][0, 1] = self.B[key][1, 0] = \
                np.exp(-1j * np.pi/4) * (self.D[key+'p'][0, 1] - self.D[key+'m'][0, 1]) \
                + np.exp(1j * np.pi/4) * (self.D[key+'p'][1, 0] - self.D[key+'m'][1, 0])
            print(f'B[{key}][0, 1] =', self.B[key][0, 1])

        h5file.close()
        
    def save_data(self) -> None:
        """Saves transition amplitudes data to hdf5 file."""
        h5file = h5py.File(self.h5fname, 'r+')
        write_hdf5(h5file, 'amp', f'B_e{self.suffix}', self.B['e'])
        write_hdf5(h5file, 'amp', f'B_h{self.suffix}', self.B['h'])
        e_orb = np.diag(self.h.molecule.orbital_energies)
        act_inds = self.h.act_inds
        write_hdf5(h5file, 'amp', f'e_orb{self.suffix}', e_orb[act_inds][:, act_inds])
        h5file.close()

    def build_all(self, method: Optional[str] = None) -> None:
        """Constructs the diagonal and off-diagonal circuits."""
        if method is not None: self.method = method
        self.build_diagonal()
        self.build_off_diagonal()

    def run_all(self, method: Optional[str] = None) -> None:
        """Executes the diagonal and off-diagonal circuits."""
        if method is not None: self.method = method
        self.run_diagonal()
        self.run_off_diagonal()

    def process_all(self, method: Optional[str] = None) -> None:
        """Post-processes data obtained from the diagonal and off-diagonal circuits."""
        if method is not None: self.method = method
        self.process_diagonal()
        self.process_off_diagonal()
        self.save_data()

    def run(self, method: Optional[str] = None) -> None:
        """Executes all functions to calculate transition amplitudes."""
        if method is not None: self.method = method
        self.build_diagonal()
        self.build_off_diagonal()
        self.run_diagonal()
        self.run_off_diagonal()
        self.process_diagonal()
        self.process_off_diagonal()
        self.save_data()
