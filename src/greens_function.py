"""New GreensFunction class"""

from typing import Union, Tuple, Optional, Sequence, Mapping
from functools import partial

import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, ClassicalRegister, Aer, transpile
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate
from qiskit.opflow import PauliSumOp

import params
from constants import HARTREE_TO_EV
from hamiltonians import MolecularHamiltonian
from number_state_solvers import measure_operator
from recompilation import CircuitRecompiler
from operators import SecondQuantizedOperators
from qubit_indices import QubitIndices, transform_4q_indices
from circuits import CircuitConstructor, CircuitData, transpile_across_barrier
from vqe import vqe_minimize, get_ansatz, get_ansatz_e, get_ansatz_h
from z2_symmetries import transform_4q_hamiltonian
from utils import save_circuit, state_tomography, solve_energy_probabilities, get_overlap
from helpers import get_quantum_instance
from vqe import GroundStateSolver
from number_state_solvers import EHStatesSolver


np.set_printoptions(precision=6)

class GreensFunction:
    """A class to calculate frequency-domain Green's function."""

    def __init__(self,
                 ansatz: QuantumCircuit,
                 hamiltonian: MolecularHamiltonian,
                 gs_solver: GroundStateSolver,
                 eh_solver: EHStatesSolver,
                 spin: str = 'edhu',
                 method: str = 'energy',
                 inds_e: QubitIndices = None,
                 inds_h: QubitIndices = None,
                 ccx_data: Optional[CircuitData] = None,
                 q_instance: QuantumInstance = None,
                 add_barriers: bool = True,
                 recompiled: bool = True,
                 transpiled: bool = True,
                 push: bool = False) -> None:
        """Initializes a GreensFunctionRestricted object.

        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The hamiltonian of the molecule.
            spin: A string indicating the spin in the tapered N+/-1 electron operators. 
                Either 'euhd' (N+1 up, N-1 down) or 'edhu' (N+1 down, N-1 up).
            method: The method for extracting the transition amplitudes. Either 'energy'
                or 'tomography'.
            q_instances: A sequence of QuantumInstance objects to execute the ground state, 
                N+/-1 electron state and transition amplitude circuits.
            recompiled: Whether the QPE circuit is recompiled.
            add_barriers: Whether to add barriers to the circuit.
            ccx_data: A CircuitData object indicating how CCX gate is applied.
            transpiled: Whether the circuit is transpiled.
            push: Whether the SWAP gates are pushed.
        """
        assert spin in ['euhd', 'edhu']

        # Basic variables of the system
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.spin = spin

        self.gs_solver = gs_solver
        self.energy_gs = gs_solver.energy
        self.state_gs = gs_solver.state

        self.eh_solver = eh_solver
        self.energies_e = eh_solver.energies_e
        self.states_e = eh_solver.states_e
        self.energies_h = eh_solver.energies_h
        self.states_h = eh_solver.states_h

        self.inds_e = inds_e 
        self.inds_h = inds_h

        self.method = method
        if q_instance is None:
            self.q_instance = get_quantum_instance('sv')
        else:
            self.q_instance = q_instance

        # Circuit construction variables
        self.recompiled = recompiled
        self.transpiled = transpiled
        self.push = push
        self.add_barriers = add_barriers
        if ccx_data is None:
            self.ccx_data = [(CCXGate(), [0, 1, 2], [])]
        else:
            self.ccx_data = ccx_data

        # Initialize operators and physical quantities
        self._initialize_operators()
        self._initialize_quantities()

    def _initialize_operators(self) -> None:
        """Initializes operator attributes."""
        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        if self.spin == 'euhd': # e up h down
            self.qiskit_op_spin = transform_4q_hamiltonian(self.qiskit_op, init_state=[0, 1]).reduce()
        elif self.spin == 'edhu': # e down h up
            self.qiskit_op_spin = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 0]).reduce()

        # Create Pauli dictionaries for the transformed and tapered operators
        second_q_ops = SecondQuantizedOperators(4)
        second_q_ops.transform(partial(transform_4q_hamiltonian, init_state=[1, 1]))
        self.pauli_dict_tapered = second_q_ops.get_op_dict_all()

    def _initialize_quantities(self) -> None:
        """Initializes physical quantity attributes."""
        # Occupied and active orbital indices
        self.occ_inds = self.hamiltonian.occ_inds
        self.act_inds = self.hamiltonian.act_inds

        # One-electron integrals and orbital energies
        self.int1e = self.hamiltonian.molecule.one_body_integrals * HARTREE_TO_EV
        self.int1e = self.int1e[self.act_inds][:, self.act_inds]
        self.e_orb = np.diag(self.hamiltonian.molecule.orbital_energies) * HARTREE_TO_EV
        self.e_orb = self.e_orb[self.act_inds][:, self.act_inds]

        # Number of spatial orbitals and N+/-1 electron states
        self.n_elec = self.hamiltonian.molecule.n_electrons
        self.n_orb = len(self.act_inds)
        self.n_occ = self.n_elec // 2 - len(self.occ_inds)
        self.n_vir = self.n_orb - self.n_occ
        self.n_e = int(binom(2 * self.n_orb, 2 * self.n_occ + 1)) // 2
        self.n_h = int(binom(2 * self.n_orb, 2 * self.n_occ - 1)) // 2

        print(f"Number of orbitals is {self.n_orb}")
        print(f"Number of occupied orbitals is {self.n_occ}")
        print(f"Number of virtual orbitals is {self.n_vir}")
        print(f"Number of (N+1)-electron states is {self.n_e}")
        print(f"Number of (N-1)-electron states is {self.n_h}")

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

        # Green's function arrays
        self.G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G = np.zeros((self.n_orb, self.n_orb), dtype=complex)

    def compute_diagonal_amplitudes(self,
                                    cache_read: bool = True,
                                    cache_write: bool = True
                                    ) -> None:
        """Calculates diagonal transition amplitudes.

        Args:
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("===== Start calculating diagonal transition amplitudes =====")
        q_instance = self.q_instances['amp']
        inds_e = self.inds_e.include_ancilla('1').int_form
        inds_h = self.inds_h.include_ancilla('0').int_form
        for m in range(self.n_orb):
            a_op_m = self.pauli_dict_tapered[(m, self.spin)]
            circ = self.circuit_constructor.build_diagonal_circuits(a_op_m)
            print(f"Calculating m = {m}")
            fname = f'circuits/circuit_{m}'
            if self.recompiled:
                recompiler = CircuitRecompiler()
                circ = recompiler.recompile_all(circ)
                fname += '_rec'
            if self.transpiled:
                circ = transpile(circ, basis_gates=['u3', 'swap', 'cz', 'cp'])
                fname += '_trans'
                if self.push:
                    fname += '_push'
            save_circuit(circ, fname)

            if q_instance.backend.name() == 'statevector_simulator':
                result = q_instance.execute(circ)
                psi = result.get_statevector()

                psi_e = psi[inds_e]
                B_e_mm = np.abs(self.states_e.conj().T @ psi_e) ** 2

                psi_h = psi[inds_h]
                B_h_mm = np.abs(self.states_h.conj().T @ psi_h) ** 2

            else: # QASM simulator or hardware
                if self.methods['amp'] == 'energy':
                    circ_anc = circ.copy()
                    circ_anc.add_register(ClassicalRegister(1))
                    circ_anc.measure(0, 0)
                    
                    result = q_instance.execute(circ_anc)
                    counts = {'0': 0, '1': 0}
                    counts.update(result.get_counts())
                    print('counts =', counts)
                    shots = sum(counts.values())
                    p_e = counts['1'] / shots
                    p_h = counts['0'] / shots

                    energy_e = measure_operator(
                        circ.copy(), self.qiskit_op_spin, q_instance=q_instance, anc_state=[1])
                    energy_h = measure_operator(
                        circ.copy(), self.qiskit_op_spin, q_instance=q_instance, anc_state=[0])

                    # print(f'energy_e = {energy_e * HARTREE_TO_EV:.6f} eV')
                    # print(f'energy_h = {energy_h * HARTREE_TO_EV:.6f} eV')

                    B_e_mm = p_e * solve_energy_probabilities(self.energies_e, energy_e * HARTREE_TO_EV)
                    B_h_mm = p_h * solve_energy_probabilities(self.energies_h, energy_h * HARTREE_TO_EV)
                
                elif self.methods['amp'] == 'tomo':
                    rho = state_tomography(circ, q_instance=self.q_instances['tomo'])

                    rho_e = rho[inds_e][:, inds_e]
                    rho_h = rho[inds_h][:, inds_h]

                    B_e_mm = np.zeros((self.n_e,), dtype=float)
                    B_h_mm = np.zeros((self.n_h,), dtype=float)
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
        print("===== Finish calculating diagonal transition amplitudes =====")

    def compute_off_diagonal_amplitudes(self,
                                        cache_read: bool = True,
                                        cache_write: bool = True
                                        ) -> None:
        """Calculates off-diagonal transition amplitudes.

        Args:
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("===== Start calculating off-diagonal transition amplitudes =====")
        q_instance = self.q_instances['amp']
        inds_ep = self.inds_e.include_ancilla('01').int_form
        inds_em = self.inds_e.include_ancilla('11').int_form
        inds_hp = self.inds_h.include_ancilla('00').int_form
        inds_hm = self.inds_h.include_ancilla('10').int_form
        for m in range(self.n_orb):
            a_op_m = self.pauli_dict_tapered[(m, self.spin)]
            for n in range(self.n_orb):
                if m < n:
                    a_op_n = self.pauli_dict_tapered[(n, self.spin)]
                    print(f"Calculating m = {m}, n = {n}")

                    circ = self.circuit_constructor.build_off_diagonal_circuits(a_op_m, a_op_n)
                    fname = f'circuits/circuit_{m}{n}'
                    if self.recompiled:
                        recompiler = CircuitRecompiler()
                        circ = recompiler.recompile_all(circ)
                        fname += '_rec'
                    if self.transpiled:
                        circ = transpile_across_barrier(
                            circ, basis_gates=['u3', 'swap', 'cz', 'cp'], 
                            push=self.push, ind=(m, n))
                        fname += '_trans'
                        if self.push:
                            fname += '_push'
                    save_circuit(circ, fname)

                    if self.method == 'exact' or q_instance.backend.name() == 'statevector_simulator':
                        result = q_instance.execute(circ)
                        psi = result.get_statevector()

                        psi_ep = psi[inds_ep]
                        psi_em = psi[inds_em]
                        D_ep_mn = abs(self.states_e.conj().T @ psi_ep) ** 2
                        D_em_mn = abs(self.states_e.conj().T @ psi_em) ** 2

                        psi_hp = psi[inds_hp]
                        psi_hm = psi[inds_hm]
                        D_hp_mn = abs(self.states_h.conj().T @ psi_hp) ** 2
                        D_hm_mn = abs(self.states_h.conj().T @ psi_hm) ** 2

                    elif self.method == 'energy':
                        circ_anc = circ.copy()
                        circ_anc.add_register(ClassicalRegister(2))
                        circ_anc.measure([0, 1], [0, 1])
                        
                        result = q_instance.execute(circ_anc)
                        counts = {'00': 0, '01': 0, '10': 0, '11': 0}
                        counts.update(result.get_counts())
                        shots = sum(counts.values())
                        p_ep = counts['01'] / shots
                        p_em = counts['11'] / shots
                        p_hp = counts['00'] / shots
                        p_hm = counts['10'] / shots

                        energy_ep = measure_operator(
                            circ.copy(), self.qiskit_op_spin, q_instance=q_instance, anc_state=[1, 0])
                        energy_em = measure_operator(
                            circ.copy(), self.qiskit_op_spin, q_instance=q_instance, anc_state=[1, 1])
                        energy_hp = measure_operator(
                            circ.copy(), self.qiskit_op_spin, q_instance=q_instance, anc_state=[0, 0])
                        energy_hm = measure_operator(
                            circ.copy(), self.qiskit_op_spin, q_instance=q_instance, anc_state=[0, 1])

                        D_ep_mn = p_ep * solve_energy_probabilities(self.energies_e, energy_ep * HARTREE_TO_EV)
                        D_em_mn = p_em * solve_energy_probabilities(self.energies_e, energy_em * HARTREE_TO_EV)
                        D_hp_mn = p_hp * solve_energy_probabilities(self.energies_h, energy_hp * HARTREE_TO_EV)
                        D_hm_mn = p_hm * solve_energy_probabilities(self.energies_h, energy_hm * HARTREE_TO_EV)
                                
                    elif self.method == 'tomo':
                        rho = state_tomography(circ, q_instance=q_instance)

                        rho_ep = rho[inds_ep][:, inds_ep]
                        rho_em = rho[inds_em][:, inds_em]
                        rho_hp = rho[inds_hp][:, inds_hp]
                        rho_hm = rho[inds_hm][:, inds_hm]

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
                    B_e_mn[abs(B_e_mn) < 1e-8] = 0
                    self.B_e[m, n] = self.B_e[n, m] = B_e_mn

                    B_h_mn = np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n])
                    B_h_mn += np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m])
                    B_h_mn[abs(B_h_mn) < 1e-8] = 0
                    self.B_h[m, n] = self.B_h[n, m] = B_h_mn

                    print(f'B_e[{m}, {n}] = {self.B_e[m, n]}')
                    print(f'B_h[{m}, {n}] = {self.B_h[m, n]}')

        print("===== Finish calculating off-diagonal transition amplitudes =====")

    def run(self,
            cache_write: bool = True,
            cache_read: bool = False
            ) -> None:
        """Main function to compute energies and transition amplitudes.

        Args:
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        # TODO: Some if conditions need to be checked.
        self.compute_diagonal_amplitudes(cache_read=cache_read, cache_write=cache_write)
        self.compute_off_diagonal_amplitudes(cache_read=cache_read, cache_write=cache_write)


    def get_density_matrix(self):
        """Returns the density matrix from the hole-added part of the Green's function"""
        self.rho_gf = np.sum(self.B_h, axis=2)
        return self.rho_gf

    def get_greens_function(self, omega: Union[float, complex]) -> np.ndarray:
        """Returns the values of the Green's function at frequency omega.

        Args:
            omega: The real or complex frequency at which the Green's function is calculated.

        Returns:
            The Green's function in numpy array form.
        """
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                self.G_e[m, n] = 2 * np.sum(self.B_e[m, n] / (omega + self.energy_gs - self.energies_e))
                self.G_h[m, n] = 2 * np.sum(self.B_h[m, n] / (omega - self.energy_gs + self.energies_h))
        self.G = self.G_e + self.G_h
        return self.G

    def get_spectral_function(self,
                              omega: Union[float, complex]
                              ) -> np.ndarray:
        """Returns the spectral function at frequency omega.

        Args:
            omega: The real or complex frequency at which the spectral function is calculated.
        """
        self.get_greens_function(omega)
        A = -1 / np.pi * np.imag(np.trace(self.G))
        return A

    def get_self_energy(self, omega: Union[float, complex]) -> np.ndarray:
        """Returns the self-energy at frequency omega.

        Args:
            The real or complex frequency at which the self-energy is calculated.

        Returns:
            The self-energy numpy array.
        """
        self.get_greens_function(omega)

        G_HF = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for i in range(self.n_orb):
            G_HF[i, i] = 1 / (omega - self.e_orb[i, i])

        Sigma = np.linalg.pinv(G_HF) - np.linalg.pinv(self.G)
        return Sigma