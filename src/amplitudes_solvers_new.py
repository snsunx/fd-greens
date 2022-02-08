"""Amplitudes solver module."""

from typing import Iterable, Optional, Sequence
from functools import partial
from collections import defaultdict

import itertools
import h5py
import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate

import params
from hamiltonians import MolecularHamiltonian
from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from operators import SecondQuantizedOperators, ChargeOperators, transform_4q_pauli
from qubit_indices import QubitIndices, transform_4q_indices
from circuits import (CircuitConstructor, InstructionTuple, append_tomography_gates,
                      append_measurement_gates, transpile_into_berkeley_gates)
from utils import (get_overlap, get_quantum_instance, counts_dict_to_arr, split_counts_on_anc)

np.set_printoptions(precision=6)

class EHAmplitudesSolver:
    """A class to calculate transition amplitudes."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 method: str = 'exact',
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 h5fname: str = 'lih',
                 anc: Sequence[int] = [0, 2]) -> None:
        """Initializes an EHAmplitudesSolver object.

        Args:
            h: The molecular Hamiltonian.
            method: The method for extracting the transition amplitudes.
            q_instance: The quantum instance for executing the circuits.
            h5fname: The HDF5 file name.
        """
        assert method in ['exact', 'tomo']

        # Basic variables
        self.h = h
        self.method = method
        self.q_instance = q_instance
        self.anc = anc
        self.sys = [i for i in range(4) if i not in anc]

        # Load data and initialize quantities
        self.h5fname = h5fname + '.h5'
        self._initialize()

        self.constructor = CircuitConstructor(self.ansatz, anc=self.anc)

    def _initialize(self) -> None:
        """Loads ground state and (N+/-1)-electron states data from hdf5 file."""
        # Attributes from ground and (N+/-1)-electron state solver.
        h5file = h5py.File(self.h5fname, 'r+')
        self.ansatz = QuantumCircuit.from_qasm_str(h5file['gs/ansatz'][()].decode())
        self.states = {'e': h5file['eh/states_e'][:], 'h': h5file['eh/states_h'][:]}
        h5file.close()

        # Number of spatial orbitals and (N+/-1)-electron states
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

        # Transition amplitude arrays.
        assert self.n_e == self.n_h # XXX
        self.B = defaultdict(lambda: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex))
        self.D = defaultdict(lambda: np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex))
    
        # Create Pauli dictionaries for the creation/annihilation operators.
        second_q_ops = SecondQuantizedOperators(self.h.molecule.n_electrons)
        second_q_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = second_q_ops.get_pauli_dict()

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
        
        for m in range(self.n_orb):
            a_op = self.pauli_dict[(m, 'd')]
            circ = self.constructor.build_eh_diagonal(a_op)
            circ = transpile_into_berkeley_gates(circ, str(m))
            h5file[f'circ{m}/base'] = circ.qasm()

            if self.method == 'tomo':
                labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
                for label in labels:
                    tomo_circ = append_tomography_gates(circ, [1, 2], label)
                    tomo_circ = append_measurement_gates(tomo_circ)
                    h5file[f'circ{m}/{label}'] = tomo_circ.qasm()

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
                dset.attrs[f'psi{m}'] = psi
            else:
                labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
                for label in labels:
                    dset = h5file[f'circ{m}/{label}']
                    circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    counts_arr = counts_dict_to_arr(counts.int_raw, n_qubits=3)
                    dset.attrs[f'counts{m}'] = counts_arr

        h5file.close()

    def process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb): # 0, 1
            if self.method == 'exact':
                psi = h5file[f'circ{m}/base'].attrs[f'psi{m}']
                for key in self.keys_diag:
                    psi_key = self.qinds_tot_diag[key](psi)
                    self.B[key][m, m] = np.abs(self.states[key].conj().T @ psi_key) ** 2
            
            elif self.method == 'tomo':
                basis_matrix = params.basis_matrix
                labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
                for key in self.keys_diag:
                    # Stack counts_arr over all tomography labels together.
                    counts_arr_key = np.array([])
                    for label in labels:
                        counts_arr = h5file[f'circ{m}/{label}'].attrs[f'counts{m}']
                        counts_arr_label = split_counts_on_anc(counts_arr, n_anc=1, key=key)
                        counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))
                    
                    # Obtain the density matrix from tomography.
                    rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0].reshape(4, 4, order='F')
                    rho = self.qinds_sys[key](rho)

                    self.B[key][m, m] = [get_overlap(self.states[key][0], rho),
                                         get_overlap(self.states[key][1], rho)]
                    print(f'B[{key}][{m}, {m}] = {self.B[key][m, m]}')

        h5file.close()

    def build_off_diagonal(self) -> None:
        """Constructs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')

        a_op_0 = self.pauli_dict[(0, 'd')]
        a_op_1 = self.pauli_dict[(1, 'd')]
        circ = self.constructor.build_eh_off_diagonal(a_op_1, a_op_0)
        circ = transpile_into_berkeley_gates(circ, '01')
        h5file[f'circ01/base'] = circ.qasm()

        if self.method == 'tomo':
            labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
            for label in labels:
                tomo_circ = append_tomography_gates(circ, self.sys, label)
                tomo_circ = append_measurement_gates(tomo_circ)
                h5file[f'circ01/{label}'] = tomo_circ.qasm()

        h5file.close()

    def run_off_diagonal(self) -> None:
        """Executes the off-diagonal transition amplitude circuit circ01."""
        h5file = h5py.File(self.h5fname, 'r+')

        if self.method == 'exact':
            dset = h5file[f'circ01/base']
            circ = QuantumCircuit.from_qasm_str(dset[()].decode())

            result = self.q_instance.execute(circ)
            psi = result.get_statevector()
            dset.attrs[f'psi01'] = psi
        else:
            labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
            for label in labels:
                dset = h5file[f'circ01/{label}']
                circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                result = self.q_instance.execute(circ)
                counts = result.get_counts()
                counts_arr = counts_dict_to_arr(counts.int_raw)
                dset.attrs[f'counts01'] = counts_arr

        h5file.close()

    def process_off_diagonal(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        if self.method == 'exact':
            psi = h5file[f'circ01/base'].attrs[f'psi01']
            for key in self.keys_off_diag:
                psi_key = self.qinds_tot_off_diag[key](psi)
                self.D[key][0, 1] = self.D[key][1, 0] = \
                    abs(self.states[key[0]].conj().T @ psi_key) ** 2
                    
        elif self.method == 'tomo':
            basis_matrix = params.basis_matrix
            labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
            for key in self.keys_off_diag:
                counts_arr_key = np.array([])
                for label in labels:
                    counts_arr = h5file[f'circ01/{label}'].attrs[f'counts01']
                    counts_arr_label = split_counts_on_anc(counts_arr, n_anc=2, key=key)
                    counts_arr_key = np.hstack((counts_arr_key, counts_arr_label))

                rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0].reshape(4, 4, order='F')
                rho = self.qinds_sys[key](rho)
                self.D[key][0, 1] = self.D[key][1, 0] = \
                    [get_overlap(self.states_h[0], rho), get_overlap(self.states_h[1], rho)]

        # Unpack D values to B values.
        for key in self.keys_diag:
            self.B[key][0, 1] = self.B[key][1, 0] = \
                np.exp(-1j * np.pi / 4) * (self.D[key + 'p'][0, 1] - self.D[key + 'm'][0, 1]) \
                + np.exp(1j * np.pi / 4) * (self.D[key + 'p'][1, 0] - self.D[key + 'm'][1, 0])
            print(f'B[{key}[0, 1] =', self.B[key][0, 1])

        h5file.close()
        
    def save_data(self) -> None:
        """Saves transition amplitudes data to hdf5 file."""
        h5file = h5py.File(self.h5fname, 'r+')
        h5file['amp/B_e'] = self.B['e']
        h5file['amp/B_h'] = self.B['h']
        e_orb = np.diag(self.h.molecule.orbital_energies)
        act_inds = self.h.act_inds
        h5file['amp/e_orb'] = e_orb[act_inds][:, act_inds]
        h5file.close()

    def run(self, method: Optional[str] = None) -> None:
        """Runs the functions to compute transition amplitudes."""
        if method is not None: self.method = method
        self.build_diagonal()
        self.build_off_diagonal()
        self.run_diagonal()
        # self.run_off_diagonal()
        self.process_diagonal()
        # self.process_off_diagonal()
        # self.save_data()


class ExcitedAmplitudesSolver:
    """A class to calculate transition amplitudes."""

    def __init__(self,
                 h: MolecularHamiltonian,
                 gs_solver: GroundStateSolver,
                 es_solver: ExcitedStatesSolver,
                 method: str = 'energy',
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 ccx_inst_tups: Iterable[InstructionTuple] = [(CCXGate(), [0, 1, 2], [])],
                 add_barriers: bool = True,
                 transpiled: bool = False,
                 push: bool = False,
                 save: bool = False) -> None:
        """Initializes an ExcitedAmplitudesSolver object.

        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The hamiltonian of the molecule.
            spin: A string indicating the spin in the tapered N+/-1 electron operators. 
                Either 'euhd' (N+1 up, N-1 down) or 'edhu' (N+1 down, N-1 up).
            method: The method for extracting the transition amplitudes. Either 'energy'
                or 'tomography'.
            q_instance: The QuantumInstance for executing the transition amplitude circuits.
            ccx_inst_tups: An iterable of instruction tuple indicating how CCX gate is applied.
            recompiled: Whether the QPE circuit is recompiled.
            add_barriers: Whether to add barriers to the circuit.
            transpiled: Whether the circuit is transpiled.
            push: Whether the SWAP gates are pushed.
        """
        # Basic variables of the system
        self.h = h

        # Attributes from ground state solver
        self.gs_solver = gs_solver
        self.energy_gs = gs_solver.energy
        self.state_gs = gs_solver.state
        self.ansatz = gs_solver.ansatz

        # Attributes from excited states solver
        self.es_solver = es_solver
        self.energies_s = es_solver.energies_s
        self.states_s = es_solver.states_s
        self.energies_t = es_solver.energies_t
        self.states_t = es_solver.states_t

        # Method and quantum instance
        self.method = method
        self.q_instance = q_instance
        self.backend = self.q_instance.backend

        # Circuit construction variables
        self.transpiled = transpiled
        self.push = push
        self.save = save
        self.add_barriers = add_barriers
        self.ccx_inst_tups = ccx_inst_tups
        self.suffix = ''
        if self.transpiled: self.suffix = self.suffix + '_trans'
        if self.push: self.suffix = self.suffix + '_push'
        self.constructor = CircuitConstructor(
            self.ansatz, add_barriers=self.add_barriers, ccx_inst_tups=self.ccx_inst_tups)

        # Initialize operators and physical quantities
        self._initialize_quantities()
        self._initialize_operators()

    def _initialize_quantities(self) -> None:
        """Initializes physical quantity attributes."""
        # Occupied and active orbital indices
        self.occ_inds = self.h.occ_inds
        self.act_inds = self.h.act_inds

        self.qinds_s = transform_4q_indices(params.singlet_inds)
        self.qinds_t = transform_4q_indices(params.triplet_inds)

        # Number of spatial orbitals and N+/-1 electron states
        self.n_elec = self.h.molecule.n_electrons
        self.n_states = len(self.energies_s) + len(self.energies_t)
        self.n_orb = 2
        #self.n_occ = self.n_elec // 2 - len(self.occ_inds)
        #self.n_vir = self.n_orb - self.n_occ

        # Transition amplitude arrays
        self.L = np.zeros((self.n_orb, self.n_orb, self.n_states), dtype=complex)
    
    def _initialize_operators(self) -> None:
        """Initializes operator attributes."""
        # Create Pauli dictionaries for operators after symmetry transformation
        charge_ops = ChargeOperators(self.n_elec)
        charge_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.charge_dict_s = charge_ops.get_pauli_dict()
        #for key, val in self.charge_dict.items():
        #    print(key, val.coeffs[0], val.table)

        charge_ops = ChargeOperators(self.n_elec)
        charge_ops.transform(partial(transform_4q_pauli, init_state=[0, 0]))
        self.charge_dict_t = charge_ops.get_pauli_dict()

    def compute_diagonal(self) -> None:
        """Calculates diagonal transition amplitudes."""
        print("----- Calculating diagonal transition amplitudes -----")
        inds_anc = QubitIndices(['10'])
        inds_tot_s = self.qinds_s + inds_anc
        inds_tot_t = self.qinds_t + inds_anc
        
        for m in range(self.n_orb):
            U_op_s = self.charge_dict_s[(m, 'd')]
            U_op_t = self.charge_dict_t[(m, 'd')]
            circ_s = self.constructor.build_charge_diagonal(U_op_s)
            circ_t = self.constructor.build_charge_diagonal(U_op_t)

            if self.transpiled: 
                circ_s = transpile(circ_s, basis_gates=params.basis_gates)
                circ_t = transpile(circ_t, basis_gates=params.basis_gates)

            if self.method == 'exact' and self.backend.name() == 'statevector_simulator':
                result = self.q_instance.execute(circ_s)
                psi = result.get_statevector()
                # for i in range(len(psi)):
                #     if abs(psi[i]) > 1e-8:
                #         print(format(i, '#06b')[2:], psi[i])
                # print('inds_tot_s =', inds_tot_s)
                psi_s = psi[inds_tot_s._int]
                L_mm_s = np.abs(self.states_s.conj().T @ psi_s) ** 2
                # print('np.sum(L_mm_s) =', np.sum(L_mm_s))

                result = self.q_instance.execute(circ_t)
                psi = result.get_statevector()
                # for i in range(len(psi)):
                #     if abs(psi[i]) > 1e-8:
                #         print(format(i, '#06b')[2:], psi[i])
                # print('inds_tot_t =', inds_tot_t)
                psi_t = psi[inds_tot_t._int]
                L_mm_t = np.abs(self.states_t.conj().T @ psi_t) ** 2
                # print('np.sum(L_mm_t) =', np.sum(L_mm_t))

                L_mm = np.hstack((L_mm_s, L_mm_t))

            self.L[m, m] = L_mm

            print(f'L[{m}, {m}] = {self.L[m, m]}')
        print("------------------------------------------------------")

    def run(self, method=None) -> None:
        """Runs the functions to compute transition amplitudes."""
        if method is not None:
            self.method = method
        self.compute_diagonal()

