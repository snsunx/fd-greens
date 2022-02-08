"""Amplitudes solver module."""

from typing import Iterable
from functools import partial

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
from circuits import (CircuitConstructor, CircuitTranspiler, InstructionTuple)
from utils import (get_overlap, get_quantum_instance, counts_dict_to_arr, split_counts_on_anc)

np.set_printoptions(precision=6)

class EHAmplitudesSolver:
    """A class to calculate transition amplitudes"""

    def __init__(self,
                 h: MolecularHamiltonian,
                 spin: str = 'edhu',
                 method: str = 'exact',
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 h5fname: str = 'lih',
                 ccx_inst_tups: Iterable[InstructionTuple] = params.ccx_inst_tups,
                 add_barriers: bool = False,
                 transpiled: bool = True,
                 swap_gates_pushed: bool = True) -> None:
        """Initializes an EHAmplitudesSolver object.

        Args:
            h: The molecular Hamiltonian.
            spin: A string indicating the spin in the tapered operators. 
            method: The method for extracting the transition amplitudes.
            q_instance: The quantum instance for executing the circuits.
            h5fname: The hdf5 file name.
            dsetname: The dataset name in the hdf5 file.
            ccx_inst_tups: An iterable of instruction tuples indicating how CCX gate is applied.
            add_barriers: Whether to add barriers to the circuit.
            transpiled: Whether the circuit is transpiled.
            swap_gates_pushed: Whether the SWAP gates are pushed.
        """
        assert spin in ['euhd', 'edhu']
        assert method in ['exact', 'tomo']

        # Basic variables
        self.h = h
        self.spin = spin
        self.method = method
        self.q_instance = q_instance

        # Load data and initialize quantities
        self.h5fname = h5fname + '.hdf5'
        self._load_data_from_hdf5()
        self._initialize_quantities()
        self._initialize_operators()

        # Circuit constructor and transpiler
        self.transpiled = transpiled
        self.circuit_constructor = CircuitConstructor(
            self.ansatz,
            add_barriers=add_barriers, 
            ccx_inst_tups=ccx_inst_tups)
        self.circuit_transpiler = CircuitTranspiler(swap_gates_pushed=swap_gates_pushed)

    def _load_data_from_hdf5(self) -> None:
        """Loads ground state and (N+/-1)-electron states data from hdf5 file. """
        h5file = h5py.File(self.h5fname, 'r+')

        # Attributes from ground state solver
        self.energy_gs = h5file['gs/energy'][()]
        qasm_str = h5file['gs/ansatz'][()].decode()
        self.ansatz = QuantumCircuit.from_qasm_str(qasm_str)

        # Attributes from (N+/-1)-electron states solver
        self.energies_e = h5file['eh/energies_e'][()]
        self.energies_h = h5file['eh/energies_h'][()]
        self.states_e = h5file['eh/states_e'][:]
        self.states_h = h5file['eh/states_h'][:]

        h5file.close()

    def _initialize_quantities(self) -> None:
        """Initializes physical quantity attributes."""
        # Occupied and active orbital indices
        self.occ_inds = self.h.occ_inds
        self.act_inds = self.h.act_inds

        # Number of spatial orbitals and (N+/-1)-electron states
        self.n_elec = self.h.molecule.n_electrons
        self.n_orb = len(self.act_inds)
        self.n_occ = self.n_elec // 2 - len(self.occ_inds)
        self.n_vir = self.n_orb - self.n_occ
        self.n_e = int(binom(2 * self.n_orb, 2 * self.n_occ + 1)) // 2
        self.n_h = int(binom(2 * self.n_orb, 2 * self.n_occ - 1)) // 2

        print("----- Printing out physical quantities -----")
        print(f"Number of electrons is {self.n_elec}")
        print(f"Number of orbitals is {self.n_orb}")
        print(f"Number of occupied orbitals is {self.n_occ}")
        print(f"Number of virtual orbitals is {self.n_vir}")
        print(f"Number of (N+1)-electron states is {self.n_e}")
        print(f"Number of (N-1)-electron states is {self.n_h}")
        print("--------------------------------------------")

        if self.spin == 'euhd':
            self.inds_e = transform_4q_indices(params.eu_inds)
            self.inds_h = transform_4q_indices(params.hd_inds)
        elif self.spin == 'edhu':
            self.inds_e = transform_4q_indices(params.ed_inds)
            self.inds_h = transform_4q_indices(params.hu_inds)

        # Transition amplitude arrays
        self.B_e = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.D_em = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.B_h = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
    
    def _initialize_operators(self) -> None:
        """Initializes operator attributes."""
        self.qiskit_op = self.h.qiskit_op.copy()
        if self.spin == 'euhd': # e up h down
            self.qiskit_op_spin = transform_4q_pauli(self.qiskit_op, init_state=[0, 1])
        elif self.spin == 'edhu': # e down h up
            self.qiskit_op_spin = transform_4q_pauli(self.qiskit_op, init_state=[1, 0])

        print('qiskit_op_spin =', self.qiskit_op_spin)

        # Create Pauli dictionaries for the transformed and tapered operators
        second_q_ops = SecondQuantizedOperators(self.n_elec)
        second_q_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = second_q_ops.get_pauli_dict()

        for key, val in self.pauli_dict.items():
            print(key, val[0].table, val[1].table)

    def build_diagonal(self) -> None:
        """Constructs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')
        
        for m in range(self.n_orb):
            a_op = self.pauli_dict[(m, self.spin[1])]
            circ = self.circuit_constructor.build_eh_diagonal(a_op)
            if self.transpiled: circ = self.circuit_transpiler.transpile(circ)
            h5file[f'circ{m}/base'] = circ.qasm()

            if self.method == 'tomo':
                labels = itertools.product('xyz', repeat=2)
                for label in labels:
                    tomo_circ = CircuitConstructor.append_tomography_gates(circ, [1, 2], label)
                    if self.transpiled: tomo_circ = self.circuit_transpiler.transpile(tomo_circ)
                    tomo_circ = CircuitConstructor.append_measurement_gates(tomo_circ)
                    label_str = ''.join(label)
                    h5file[f'circ{m}/{label_str}'] = tomo_circ.qasm()

        h5file.close()

    def run_diagonal(self) -> None:
        """Executes the diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb):
            if self.method == 'exact':
                dset = h5file[f'circ{m}/base']
                qasm_str = dset[()].decode()
                circ = QuantumCircuit.from_qasm_str(qasm_str)
                result = self.q_instance.execute(circ)
                psi = result.get_statevector()
                dset.attrs[f'psi{m}'] = psi
            else:
                labels = itertools.product('xyz', repeat=2)
                for label in labels:
                    label_str = ''.join(label)
                    dset = h5file[f'circ{m}/{label_str}']
                    qasm_str = dset[()].decode()
                    circ = QuantumCircuit.from_qasm_str(qasm_str)
                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    counts_dict = counts.int_raw
                    counts_arr = counts_dict_to_arr(counts_dict)
                    dset.attrs[f'counts{m}'] = counts_arr

        h5file.close()

    def process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        inds_anc_e = QubitIndices(['1'])
        inds_anc_h = QubitIndices(['0'])
        inds_tot_e = self.inds_e + inds_anc_e
        inds_tot_h = self.inds_h + inds_anc_h

        for m in range(self.n_orb):
            if self.method == 'exact':
                psi = dset.attrs[f'psi{m}']

                psi_e = psi[inds_tot_e.int_form]
                B_e_mm = np.abs(self.states_e.conj().T @ psi_e) ** 2

                psi_h = psi[inds_tot_h.int_form]
                B_h_mm = np.abs(self.states_h.conj().T @ psi_h) ** 2
            
            elif self.method == 'tomo':
                basis_matrix = params.basis_matrix
                data_h = np.array([])
                data_e = np.array([])

                labels = itertools.product('xyz', repeat=2)
                for label in labels:
                    label_str = ''.join(label)
                    dset = h5file[f'circ{m}/{label_str}']
                    counts = dset.attrs[f'counts{m}']
                    counts_h, counts_e = split_counts_on_anc(counts, n_anc=1)

                    data_h = np.hstack((data_h, counts_h))
                    data_e = np.hstack((data_e, counts_e))
                    
                rho_h = np.linalg.lstsq(basis_matrix, data_h)[0].reshape(4, 4, order='F')
                rho_e = np.linalg.lstsq(basis_matrix, data_e)[0].reshape(4, 4, order='F')
                rho_h = rho_h[self.inds_h.int_form][:, self.inds_h.int_form]
                rho_e = rho_e[self.inds_e.int_form][:, self.inds_e.int_form]

                B_h_mm = np.zeros((self.n_h,), dtype=float)
                B_e_mm = np.zeros((self.n_e,), dtype=float)
                for i in range(self.n_h):
                    B_h_mm[i] = get_overlap(self.states_h[i], rho_h)
                for i in range(self.n_e):
                    B_e_mm[i] = get_overlap(self.states_e[i], rho_e)

            self.B_e[m, m] = B_e_mm
            self.B_h[m, m] = B_h_mm
            print(f'B_e[{m}, {m}] = {self.B_e[m, m]}')
            print(f'B_h[{m}, {m}] = {self.B_h[m, m]}')

        h5file.close()

    def build_off_diagonal(self) -> None:
        """Constructs the off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb):
            a_op_m = self.pauli_dict[(m, self.spin[1])]
            for n in range(m + 1, self.n_orb):
                a_op_n = self.pauli_dict[(n, self.spin[1])]
                circ = self.circuit_constructor.build_eh_off_diagonal(a_op_m, a_op_n)
                if self.transpiled: 
                    circ = self.circuit_transpiler.transpile_across_barriers(circ)
                h5file[f'circ{m}{n}/base'] = circ.qasm()

                if self.method == 'tomo':
                    labels = itertools.product('xyz', repeat=2)
                    for label in labels:
                        tomo_circ = CircuitConstructor.append_tomography_gates(circ, [2, 3], label)
                        if self.transpiled: 
                            tomo_circ = self.circuit_transpiler.transpile_last_section(tomo_circ)
                        tomo_circ = CircuitConstructor.append_measurement_gates(tomo_circ)
                        label_str = ''.join(label)
                        h5file[f'circ{m}{n}/{label_str}'] = tomo_circ.qasm()

        h5file.close()

    def run_off_diagonal(self) -> None:
        """Executes the off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, 'r+')

        for m in range(self.n_orb):
            for n in range(m + 1, self.n_orb):
                if self.method == 'exact':
                    dset = h5file[f'circ{m}{n}/base']
                    qasm_str = dset[()].decode()
                    circ = QuantumCircuit.from_qasm_str(qasm_str)
                    result = self.q_instance.execute(circ)
                    psi = result.get_statevector()
                    dset.attrs[f'psi{m}{n}'] = psi
                else:
                    labels = itertools.product('xyz', repeat=2)
                    for label in labels:
                        label_str = ''.join(label)
                        dset = h5file[f'circ{m}{n}/{label_str}']
                        qasm_str = dset[()].decode()
                        circ = QuantumCircuit.from_qasm_str(qasm_str)

                        result = self.q_instance.execute(circ)
                        counts = result.get_counts()
                        counts_dict = counts.int_raw
                        counts_arr = counts_dict_to_arr(counts_dict)

                        dset.attrs[f'counts{m}{n}'] = counts_arr

        h5file.close()

    def process_off_diagonal(self) -> None:
        """Post-processes off-diagonal transition amplitude results."""
        h5file = h5py.File(self.h5fname, 'r+')

        inds_tot_ep = self.inds_e + QubitIndices(['01'])
        inds_tot_em = self.inds_e + QubitIndices(['11'])
        inds_tot_hp = self.inds_h + QubitIndices(['00'])
        inds_tot_hm = self.inds_h + QubitIndices(['10'])

        for m in range(self.n_orb):
            for n in range(m + 1, self.n_orb):
                if self.method == 'exact':
                    dset = h5file[f'circ{m}{n}/base']
                    psi = dset.attrs[f'psi{m}{n}']

                    psi_ep = psi[inds_tot_ep.int_form]
                    psi_em = psi[inds_tot_em.int_form]
                    D_ep_mn = abs(self.states_e.conj().T @ psi_ep) ** 2
                    D_em_mn = abs(self.states_e.conj().T @ psi_em) ** 2

                    psi_hp = psi[inds_tot_hp.int_form]
                    psi_hm = psi[inds_tot_hm.int_form]
                    D_hp_mn = abs(self.states_h.conj().T @ psi_hp) ** 2
                    D_hm_mn = abs(self.states_h.conj().T @ psi_hm) ** 2
                            
                elif self.method == 'tomo':
                    basis_matrix = params.basis_matrix
                    data_hp = np.array([])
                    data_ep = np.array([])
                    data_hm = np.array([])
                    data_em = np.array([])

                    labels = itertools.product('xyz', repeat=2)
                    for label in labels:
                        label_str = ''.join([label[1], label[0]])
                        dset = h5file[f'circ{m}{n}/{label_str}']
                        counts = dset.attrs[f'counts{m}{n}']
                        counts_hp, counts_ep, counts_hm, counts_em = \
                            split_counts_on_anc(counts, n_anc=2)

                        data_hp = np.hstack((data_hp, counts_hp))
                        data_ep = np.hstack((data_ep, counts_ep))
                        data_hm = np.hstack((data_hm, counts_hm))
                        data_em = np.hstack((data_em, counts_em))
                        
                    rho_hp = np.linalg.lstsq(basis_matrix, data_hp)[0].reshape(4, 4, order='F')
                    rho_ep = np.linalg.lstsq(basis_matrix, data_ep)[0].reshape(4, 4, order='F')
                    rho_hm = np.linalg.lstsq(basis_matrix, data_hm)[0].reshape(4, 4, order='F')
                    rho_em = np.linalg.lstsq(basis_matrix, data_em)[0].reshape(4, 4, order='F')

                    rho_hp = rho_hp[self.inds_h.int_form][:, self.inds_h.int_form]
                    rho_ep = rho_ep[self.inds_e.int_form][:, self.inds_e.int_form]
                    rho_hm = rho_hm[self.inds_h.int_form][:, self.inds_h.int_form]
                    rho_em = rho_em[self.inds_e.int_form][:, self.inds_e.int_form]

                    D_hp_mn = np.zeros((self.n_h,), dtype=float)
                    D_ep_mn = np.zeros((self.n_e,), dtype=float)
                    D_hm_mn = np.zeros((self.n_h,), dtype=float)
                    D_em_mn = np.zeros((self.n_e,), dtype=float)
                    for i in range(self.n_h):
                        D_hp_mn[i] = get_overlap(self.states_h[i], rho_hp)
                        D_hm_mn[i] = get_overlap(self.states_h[i], rho_hm)
                    for i in range(self.n_e):
                        D_ep_mn[i] = get_overlap(self.states_e[i], rho_ep)
                        D_em_mn[i] = get_overlap(self.states_e[i], rho_em)

                self.D_ep[m, n] = self.D_ep[n, m] = D_ep_mn
                self.D_em[m, n] = self.D_em[n, m] = D_em_mn
                self.D_hp[m, n] = self.D_hp[n, m] = D_hp_mn
                self.D_hm[m, n] = self.D_hm[n, m] = D_hm_mn

        # Unpack D values to B values
        for m in range(self.n_orb):
            for n in range(m + 1, self.n_orb):
                B_e_mn = np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n])
                B_e_mn += np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m])
                self.B_e[m, n] = self.B_e[n, m] = B_e_mn

                B_h_mn = np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n])
                B_h_mn += np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m])
                self.B_h[m, n] = self.B_h[n, m] = B_h_mn
                
                print(f'B_e[{m}, {n}] = {self.B_e[m, n]}')
                print(f'B_h[{m}, {n}] = {self.B_h[m, n]}')

        h5file.close()
        

    def save_data(self) -> None:
        """Saves transition amplitudes data to hdf5 file."""
        h5file = h5py.File(self.h5fname, 'r+')
        h5file['amp/B_e'] = self.B_e
        h5file['amp/B_h'] = self.B_h
        e_orb = np.diag(self.h.molecule.orbital_energies)
        act_inds = self.h.act_inds
        h5file['amp/e_orb'] = e_orb[act_inds][:, act_inds]
        h5file.close()

    def run(self, method=None) -> None:
        """Runs the functions to compute transition amplitudes."""
        if method is not None: self.method = method
        self.build_diagonal()
        self.build_off_diagonal()
        self.run_diagonal()
        self.run_off_diagonal()
        self.process_diagonal()
        self.process_off_diagonal()
        self.save_data()


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
        self.circuit_constructor = CircuitConstructor(
            self.ansatz, add_barriers=self.add_barriers, ccx_inst_tups=self.ccx_inst_tups)

        # Initialize operators and physical quantities
        self._initialize_quantities()
        self._initialize_operators()

    def _initialize_quantities(self) -> None:
        """Initializes physical quantity attributes."""
        # Occupied and active orbital indices
        self.occ_inds = self.h.occ_inds
        self.act_inds = self.h.act_inds

        self.inds_s = transform_4q_indices(params.singlet_inds)
        self.inds_t = transform_4q_indices(params.triplet_inds)

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
        inds_tot_s = self.inds_s + inds_anc
        inds_tot_t = self.inds_t + inds_anc
        
        for m in range(self.n_orb):
            U_op_s = self.charge_dict_s[(m, 'd')]
            U_op_t = self.charge_dict_t[(m, 'd')]
            circ_s = self.circuit_constructor.build_charge_diagonal(U_op_s)
            circ_t = self.circuit_constructor.build_charge_diagonal(U_op_t)

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
                psi_s = psi[inds_tot_s.int_form]
                L_mm_s = np.abs(self.states_s.conj().T @ psi_s) ** 2
                # print('np.sum(L_mm_s) =', np.sum(L_mm_s))

                result = self.q_instance.execute(circ_t)
                psi = result.get_statevector()
                # for i in range(len(psi)):
                #     if abs(psi[i]) > 1e-8:
                #         print(format(i, '#06b')[2:], psi[i])
                # print('inds_tot_t =', inds_tot_t)
                psi_t = psi[inds_tot_t.int_form]
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


