"""Amplitudes solver module."""

from typing import Optional
from functools import partial

import os
import h5py
import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate

import params
from hamiltonians import MolecularHamiltonian
from number_states_solvers import measure_operator, EHStatesSolver, ExcitedStatesSolver
from operators import SecondQuantizedOperators, ChargeOperators
from qubit_indices import QubitIndices, transform_4q_indices
from circuits import CircuitConstructor, CircuitData, transpile_across_barrier
from z2_symmetries import transform_4q_pauli
from utils import save_circuit, state_tomography, solve_energy_probabilities, get_overlap, get_counts, get_quantum_instance
from ground_state_solvers import GroundStateSolver

np.set_printoptions(precision=6)

class EHAmplitudesSolver:
    """A class to calculate transition amplitudes"""

    def __init__(self,
                 h: MolecularHamiltonian,
                 gs_solver: GroundStateSolver,
                 es_solver: EHStatesSolver,
                 spin: str = 'edhu',
                 method: str = 'energy',
                 q_instance: QuantumInstance = None,
                 ccx_data: Optional[CircuitData] = None,
                 add_barriers: bool = True,
                 transpiled: bool = False,
                 push: bool = False,
                 save: bool = False,
                 fname: str = 'lih',
                 dsetname: str = 'eh_exact') -> None:
        """Initializes an EHAmplitudesSolver object.

        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The hamiltonian of the molecule.
            spin: A string indicating the spin in the tapered N+/-1 electron operators. 
                Either 'euhd' (N+1 up, N-1 down) or 'edhu' (N+1 down, N-1 up).
            method: The method for extracting the transition amplitudes. Either 'energy' or 'tomo'.
            q_instance: The QuantumInstance for executing the transition amplitude circuits.
            ccx_data: A CircuitData object indicating how CCX gate is applied.
            recompiled: Whether the QPE circuit is recompiled.
            add_barriers: Whether to add barriers to the circuit.
            transpiled: Whether the circuit is transpiled.
            push: Whether the SWAP gates are pushed.
        """
        assert spin in ['euhd', 'edhu']

        # Basic variables of the system
        self.h = h
        self.spin = spin

        # Attributes from ground state solver
        self.gs_solver = gs_solver
        self.energy_gs = gs_solver.energy
        self.state_gs = gs_solver.state
        self.ansatz = gs_solver.ansatz

        # Attributes from N+/-1 electron states solver
        self.es_solver = es_solver
        self.energies_e = es_solver.energies_e
        self.states_e = es_solver.states_e
        self.energies_h = es_solver.energies_h
        self.states_h = es_solver.states_h

        # Method and quantum instance
        self.method = method
        if q_instance is None:
            self.q_instance = get_quantum_instance('sv')
        else:
            self.q_instance = q_instance
        self.backend = self.q_instance.backend

        # Circuit construction variables
        self.transpiled = transpiled
        self.push = push
        self.save = save
        self.add_barriers = add_barriers
        if ccx_data is None: 
            self.ccx_data = [(CCXGate(), [0, 1, 2], [])]
        else: 
            self.ccx_data = ccx_data
        self.suffix = ''
        if self.transpiled: self.suffix = self.suffix + '_trans'
        if self.push: self.suffix = self.suffix + '_push'
        self.circuit_constructor = CircuitConstructor(
            self.ansatz, add_barriers=self.add_barriers, ccx_data=self.ccx_data)

        self.fname = fname + '.hdf5'
        self.dsetname = dsetname

        # Initialize operators and physical quantities
        self._initialize_quantities()
        self._initialize_operators()

        


    def _initialize_quantities(self) -> None:
        """Initializes physical quantity attributes."""
        # Occupied and active orbital indices
        self.occ_inds = self.h.occ_inds
        self.act_inds = self.h.act_inds

        # Number of spatial orbitals and N+/-1 electron states
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

        # Create Pauli dictionaries for the transformed and tapered operators
        second_q_ops = SecondQuantizedOperators(self.n_elec)
        second_q_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = second_q_ops.get_op_dict_all()

    def build_diagonal(self) -> None:
        """Constructs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.fname, 'r+')
        dset = h5file[self.dsetname]
        
        for m in range(self.n_orb):
            a_op = self.pauli_dict[(m, self.spin[1])]
            circ = self.circuit_constructor.build_eh_diagonal(a_op)
            #fname = f'circuits/circuit_{m}' + self.suffix
            if self.transpiled: circ = transpile(circ, basis_gates=['u3', 'swap', 'cz', 'cp'])
            #if self.save: save_circuit(circ, fname)
            dset.attrs[f'circ{m}'] = circ.qasm()

        h5file.close()

    def run_diagonal(self) -> None:
        """Executes the diagonal transition amplitude circuits."""
        h5file = h5py.File(self.fname, 'r+')
        dset = h5file[self.dsetname]

        for m in range(self.n_orb):
            qasm_str = dset.attrs[f'circ{m}']
            circ = QuantumCircuit.from_qasm_str(qasm_str)
            result = self.q_instance.execute(circ)
            if self.method == 'exact':
                psi = result.get_statevector()
                dset.attrs[f'psi{m}'] = psi
            else:
                counts = result.get_counts()
                dset.attrs[f'counts{m}'] = counts

        h5file.close()

    def build_off_diagonal(self) -> None:
        """Constructs the off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.fname, 'r+')
        dset = h5file[self.dsetname]

        for m in range(self.n_orb):
            a_op_m = self.pauli_dict[(m, self.spin[1])]
            for n in range(self.n_orb):
                if m < n:
                    a_op_n = self.pauli_dict[(n, self.spin[1])]

                    circ = self.circuit_constructor.build_eh_off_diagonal(a_op_m, a_op_n)
                    # fname = f'circuits/circuit_{m}{n}' + self.suffix
                    if self.transpiled:
                        circ = transpile_across_barrier(
                            circ, basis_gates=['u3', 'swap', 'cz', 'cp'], 
                            push=self.push, ind=(m, n))
                    # if self.save: save_circuit(circ, fname)
                    dset.attrs[f'circ{m}{n}'] = circ.qasm()

        h5file.close()

    def run_off_diagonal(self) -> None:
        """Executes the off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.fname, 'r+')
        dset = h5file[self.dsetname]

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m < n:
                    qasm_str = dset.attrs[f'circ{m}{n}']
                    circ = QuantumCircuit.from_qasm_str(qasm_str)
                    result = self.q_instance.execute(circ)
                    if self.method == 'exact':
                        psi = result.get_statevector()
                        dset.attrs[f'psi{m}{n}'] = psi
                    else:
                        counts = result.get_counts()
                        dset.attrs[f'counts{m}{n}'] = counts

        h5file.close()


    def compute_diagonal_amplitudes(self) -> None:
        """Calculates diagonal transition amplitudes."""
        print("----- Calculating diagonal transition amplitudes -----")
        inds_anc_e = QubitIndices(['1'])
        inds_anc_h = QubitIndices(['0'])
        inds_tot_e = self.inds_e + inds_anc_e
        inds_tot_h = self.inds_h + inds_anc_h
        for m in range(self.n_orb):
            a_op_m = self.pauli_dict[(m, self.spin[1])]
            circ = self.circuit_constructor.build_diagonal_circuits(a_op_m)
            fname = f'circuits/circuit_{m}' + self.suffix
            if self.transpiled: circ = transpile(circ, basis_gates=['u3', 'swap', 'cz', 'cp'])
            if self.save: save_circuit(circ, fname)

            if self.method == 'exact' and self.backend.name() == 'statevector_simulator':
                result = self.q_instance.execute(circ)
                psi = result.get_statevector()

                psi_e = psi[inds_tot_e.int_form]
                B_e_mm = np.abs(self.states_e.conj().T @ psi_e) ** 2

                psi_h = psi[inds_tot_h.int_form]
                B_h_mm = np.abs(self.states_h.conj().T @ psi_h) ** 2

            elif self.method == 'energy':
                circ_anc = circ.copy()
                circ_anc.add_register(ClassicalRegister(1))
                circ_anc.measure(0, 0)
                
                result = self.q_instance.execute(circ_anc)
                counts = get_counts(result)
                shots = sum(counts.values())
                p_e = counts[inds_anc_e.str_form[0]] / shots
                p_h = counts[inds_anc_h.str_form[0]] / shots

                energy_e = measure_operator(circ.copy(), self.qiskit_op_spin, 
                                            q_instance=self.q_instance, 
                                            anc_state=inds_anc_e.list_form[0])
                energy_h = measure_operator(circ.copy(), self.qiskit_op_spin,
                                            q_instance=self.q_instance,
                                            anc_state=inds_anc_h.list_form[0])

                B_e_mm = p_e * solve_energy_probabilities(self.energies_e, energy_e)
                B_h_mm = p_h * solve_energy_probabilities(self.energies_h, energy_h)
                
            elif self.method == 'tomo':
                rho = state_tomography(circ, q_instance=self.q_instance)

                rho_e = rho[inds_tot_e.int_form][:, inds_tot_e.int_form]
                rho_h = rho[inds_tot_h.int_form][:, inds_tot_h.int_form]

                B_e_mm = np.zeros((self.n_e,), dtype=float)
                B_h_mm = np.zeros((self.n_h,), dtype=float)
                print(self.states_e)
                print(rho_e)
                for i in range(self.n_e):
                    B_e_mm[i] = get_overlap(self.states_e[i], rho_e)
                for i in range(self.n_h):
                    B_h_mm[i] = get_overlap(self.states_h[i], rho_h)

            # B_e_mm[abs(B_e_mm) < 1e-8] = 0.
            self.B_e[m, m] = B_e_mm
            # B_h_mm[abs(B_h_mm) < 1e-8] = 0.
            self.B_h[m, m] = B_h_mm

            print(f'B_e[{m}, {m}] = {self.B_e[m, m]}')
            print(f'B_h[{m}, {m}] = {self.B_h[m, m]}')
        print("------------------------------------------------------")

    def compute_off_diagonal_amplitudes(self) -> None:
        """Calculates off-diagonal transition amplitudes."""
        print("----- Calculating off-diagonal transition amplitudes -----")

        inds_anc_ep = QubitIndices(['01'])
        inds_anc_em = QubitIndices(['11'])
        inds_anc_hp = QubitIndices(['00'])
        inds_anc_hm = QubitIndices(['10'])
        inds_tot_ep = self.inds_e + inds_anc_ep
        inds_tot_em = self.inds_e + inds_anc_em
        inds_tot_hp = self.inds_h + inds_anc_hp
        inds_tot_hm = self.inds_h + inds_anc_hm

        for m in range(self.n_orb):
            a_op_m = self.pauli_dict[(m, self.spin[1])]
            for n in range(self.n_orb):
                if m < n:
                    a_op_n = self.pauli_dict[(n, self.spin[1])]

                    circ = self.circuit_constructor.build_off_diagonal_circuits(a_op_m, a_op_n)
                    fname = f'circuits/circuit_{m}{n}' + self.suffix
                    if self.transpiled:
                        circ = transpile_across_barrier(
                            circ, basis_gates=['u3', 'swap', 'cz', 'cp'], 
                            push=self.push, ind=(m, n))
                    if self.save: save_circuit(circ, fname)

                    if self.method == 'exact' or self.backend.name() == 'statevector_simulator':
                        result = self.q_instance.execute(circ)
                        psi = result.get_statevector()

                        psi_ep = psi[inds_tot_ep.int_form]
                        psi_em = psi[inds_tot_em.int_form]
                        D_ep_mn = abs(self.states_e.conj().T @ psi_ep) ** 2
                        D_em_mn = abs(self.states_e.conj().T @ psi_em) ** 2

                        psi_hp = psi[inds_tot_hp.int_form]
                        psi_hm = psi[inds_tot_hm.int_form]
                        D_hp_mn = abs(self.states_h.conj().T @ psi_hp) ** 2
                        D_hm_mn = abs(self.states_h.conj().T @ psi_hm) ** 2

                    elif self.method == 'energy':
                        circ_anc = circ.copy()
                        circ_anc.add_register(ClassicalRegister(2))
                        circ_anc.measure([0, 1], [0, 1])
                        
                        result = self.q_instance.execute(circ_anc)
                        counts = get_counts(result)
                        shots = sum(counts.values())
                        p_ep = counts[inds_anc_ep.str_form[0]] / shots
                        p_em = counts[inds_anc_em.str_form[0]] / shots
                        p_hp = counts[inds_anc_hp.str_form[0]] / shots
                        p_hm = counts[inds_anc_hm.str_form[0]] / shots

                        energy_ep = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance,
                                                     anc_state=inds_anc_ep.list_form[0])
                        energy_em = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance,
                                                     anc_state=inds_anc_em.list_form[0])
                        energy_hp = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance,
                                                     anc_state=inds_anc_hp.list_form[0])
                        energy_hm = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance, 
                                                     anc_state=inds_anc_hm.list_form[0])

                        D_ep_mn = p_ep * solve_energy_probabilities(self.energies_e, energy_ep)
                        D_em_mn = p_em * solve_energy_probabilities(self.energies_e, energy_em)
                        D_hp_mn = p_hp * solve_energy_probabilities(self.energies_h, energy_hp)
                        D_hm_mn = p_hm * solve_energy_probabilities(self.energies_h, energy_hm)
                                
                    elif self.method == 'tomo':
                        rho = state_tomography(circ, q_instance=self.q_instance)

                        rho_ep = rho[inds_tot_ep.int_form][:, inds_tot_ep.int_form]
                        rho_em = rho[inds_tot_em.int_form][:, inds_tot_em.int_form]
                        rho_hp = rho[inds_tot_hp.int_form][:, inds_tot_hp.int_form]
                        rho_hm = rho[inds_tot_hm.int_form][:, inds_tot_hm.int_form]

                        D_ep_mn = np.zeros((self.n_e,), dtype=float)
                        D_em_mn = np.zeros((self.n_e,), dtype=float)
                        D_hp_mn = np.zeros((self.n_h,), dtype=float)
                        D_hm_mn = np.zeros((self.n_h,), dtype=float)

                        for i in range(self.n_e):
                            D_ep_mn[i] = get_overlap(self.states_e[i], rho_ep)
                            D_em_mn[i] = get_overlap(self.states_e[i], rho_em)

                        for i in range(self.n_h):
                            D_hp_mn[i] = get_overlap(self.states_h[i], rho_hp)
                            D_hm_mn[i] = get_overlap(self.states_h[i], rho_hm)

                    self.D_ep[m, n] = self.D_ep[n, m] = D_ep_mn
                    self.D_em[m, n] = self.D_em[n, m] = D_em_mn
                    self.D_hp[m, n] = self.D_hp[n, m] = D_hp_mn
                    self.D_hm[m, n] = self.D_hm[n, m] = D_hm_mn

                    print(f'D_ep[{m}, {n}] = {self.D_ep[m, n]}')
                    print(f'D_em[{m}, {n}] = {self.D_em[m, n]}')
                    print(f'D_hp[{m}, {n}] = {self.D_hp[m, n]}')
                    print(f'D_hm[{m}, {n}] = {self.D_hm[m, n]}')

        # Unpack D values to B values
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m < n:
                    B_e_mn = np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n])
                    B_e_mn += np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m])
                    # B_e_mn[abs(B_e_mn) < 1e-8] = 0
                    self.B_e[m, n] = self.B_e[n, m] = B_e_mn

                    B_h_mn = np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n])
                    B_h_mn += np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m])
                    # B_h_mn[abs(B_h_mn) < 1e-8] = 0
                    self.B_h[m, n] = self.B_h[n, m] = B_h_mn
                    
                    print(f'B_e[{m}, {n}] = {self.B_e[m, n]}')
                    print(f'B_h[{m}, {n}] = {self.B_h[m, n]}')

        print("----------------------------------------------------------")

    def run(self, method=None) -> None:
        """Runs the functions to compute transition amplitudes."""
        if method is not None:
            self.method = method
        self.build_diagonal()
        self.build_off_diagonal()
        self.run_diagonal()
        self.run_off_diagonal()
        exit()
        #self.compute_diagonal_amplitudes()
        #self.compute_off_diagonal_amplitudes()


class ExcitedAmplitudesSolver:
    """A class to calculate transition amplitudes"""

    def __init__(self,
                 h: MolecularHamiltonian,
                 gs_solver: GroundStateSolver,
                 es_solver: ExcitedStatesSolver,
                 method: str = 'energy',
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 ccx_data: CircuitData = [(CCXGate(), [0, 1, 2], [])],
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
            ccx_data: A CircuitData object indicating how CCX gate is applied.
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
        self.ccx_data = ccx_data
        self.suffix = ''
        if self.transpiled: self.suffix = self.suffix + '_trans'
        if self.push: self.suffix = self.suffix + '_push'
        self.circuit_constructor = CircuitConstructor(
            self.ansatz, add_barriers=self.add_barriers, ccx_data=self.ccx_data)

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
        self.charge_dict_s = charge_ops.get_op_dict_all()
        #for key, val in self.charge_dict.items():
        #    print(key, val.coeffs[0], val.table)

        charge_ops = ChargeOperators(self.n_elec)
        charge_ops.transform(partial(transform_4q_pauli, init_state=[0, 0]))
        self.charge_dict_t = charge_ops.get_op_dict_all()

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

            #fname = f'circuits/circuit_{m}' + self.suffix
            if self.transpiled: 
                circ_s = transpile(circ_s, basis_gates=['u3', 'swap', 'cz', 'cp'])
                circ_t = transpile(circ_t, basis_gates=['u3', 'swap', 'cz', 'cp'])
            #if self.save: save_circuit(circ, fname)

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


