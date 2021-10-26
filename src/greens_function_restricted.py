"""GreensFunctionRestricted class"""

from typing import Union, Tuple, Optional
from functools import partial

import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, ClassicalRegister, Aer, transpile
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate
from qiskit.opflow import PauliSumOp

from constants import HARTREE_TO_EV
from hamiltonians import MolecularHamiltonian
from number_state_solvers import (number_state_eigensolver, quantum_subspace_expansion,
                                  quantum_subspace_expansion_exact, measure_operator)
from recompilation import CircuitRecompiler
from operators import SecondQuantizedOperators
from qubit_indices import QubitIndices
from circuits import CircuitConstructor, CircuitData, transpile_across_barrier
from vqe import vqe_minimize
from z2_symmetries import transform_4q_hamiltonian
from utils import save_circuit, state_tomography, solve_energy_probabilities

np.set_printoptions(precision=6)

class GreensFunctionRestricted:
    """A class to calculate frequency-domain Green's function with restricted orbitals."""

    def __init__(self,
                 ansatz: QuantumCircuit,
                 hamiltonian: MolecularHamiltonian,
                 optimizer: Optional[Optimizer] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 recompiled: bool = True,
                 add_barriers: bool = True,
                 ccx_data: Optional[CircuitData] = None,
                 transpiled: bool = True,
                 push: bool = False,                 
                 spin: str = 'up') -> None:
        """Initializes a GreensFunctionRestricted object.

        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The MolecularHamiltonian object.
            optimizer: The optimizer in VQE. Default to L_BFGS_B().
            q_instance: The QuantumInstance to execute the circuit.
            recompiled: Whether the QPE circuit is recompiled.
            add_barriers: Whether to add barriers to the circuit.
            ccx_data: A CircuitData object indicating how CCX gate is applied.
            transpiled: Whether the circuit is transpiled.
            push: Whether the SWAP gates are pushed.
            spin: Whether to use the up-spin or down-spin tapered operator.
        """
        # XXX: transform_4q_hamiltonian and init_state are hardcoded
        # Define Hamiltonian and related variables
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.molecule = hamiltonian.molecule
        self.e_orb = np.diag(self.molecule.orbital_energies) * HARTREE_TO_EV
        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        self.qiskit_op_gs = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 1]).reduce()
        self.spin = spin
        if spin == 'up': # e up h down
            self.qiskit_op_spin = transform_4q_hamiltonian(self.qiskit_op, init_state=[0, 1]).reduce()
        elif spin == 'down': # e down h up
            self.qiskit_op_spin = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 0]).reduce()
        self.qiskit_op_up = transform_4q_hamiltonian(self.qiskit_op, init_state=[0, 1]).reduce()
        self.qiskit_op_down = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 0]).reduce()

        # Create the Pauli operator dictionary
        second_q_ops = SecondQuantizedOperators(4)
        self.pauli_op_dict = second_q_ops.get_op_dict_all()

        second_q_ops = SecondQuantizedOperators(4)
        second_q_ops.transform(partial(transform_4q_hamiltonian, init_state=[1, 1], tapered=False))
        self.pauli_op_dict_trans = second_q_ops.get_op_dict_all()
        for key, (x_op, y_op) in self.pauli_op_dict_trans.items():
            print(key, x_op.table.to_labels()[0], y_op.table.to_labels()[0])

        second_q_ops = SecondQuantizedOperators(4)
        second_q_ops.transform(partial(transform_4q_hamiltonian, init_state=[1, 1]))
        # self.pauli_op_dict_tapered = second_q_ops.get_op_dict(spin=spin)
        self.pauli_op_dict_tapered = second_q_ops.get_op_dict_all()

        if optimizer is None:
            self.optimizer = L_BFGS_B()
        else:
            self.optimizer = optimizer
        if q_instance is None:
            self.q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
        else:
            self.q_instance = q_instance
        self.recompiled = recompiled
        self.transpiled = transpiled
        self.push = push
        self.method = 'energy'

        self.add_barriers = add_barriers
        if ccx_data is None:
            self.ccx_data = [(CCXGate(), [0, 1, 2], [])]
        else:
            self.ccx_data = ccx_data

        # Number of orbitals and indices
        self.n_orb = len(self.hamiltonian.act_inds)
        self.n_occ = self.hamiltonian.molecule.n_electrons // 2 - len(self.hamiltonian.occ_inds)
        self.n_vir = self.n_orb - self.n_occ
        self.inds_occ = list(range(self.n_occ))
        self.inds_vir = list(range(self.n_occ, self.n_orb))
        self.n_e = int(binom(2 * self.n_orb, 2 * self.n_occ + 1)) // 2
        self.n_h = int(binom(2 * self.n_orb, 2 * self.n_occ - 1)) // 2

        print(f"Number of orbitals is {self.n_orb}")
        print(f"Number of occupied orbitals is {self.n_occ}")
        print(f"Number of virtual orbitals is {self.n_vir}")
        print(f"Number of (N+1)-electron states is {self.n_e}")
        print(f"Number of (N-1)-electron states is {self.n_h}")

        # `swap` is whether an additional swap gate is applied on top of the two CNOTs.
        # This part is hardcoded.
        swap = True
        if not swap:
            if spin == 'up':
                self.inds_h = QubitIndices(['10', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '01'], n_qubits=2)
            elif spin == 'down':
                self.inds_h = QubitIndices(['01', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '10'], n_qubits=2)
        else:
            if spin == 'up':
                self.inds_h = QubitIndices(['01', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '10'], n_qubits=2)
            elif spin == 'down':
                self.inds_h = QubitIndices(['10', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '01'], n_qubits=2)

        # Define the transition amplitude matrices
        self.B_e = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.D_em = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)

        self.B_h = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)

        # Define the Green's function arrays
        self.G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G = np.zeros((self.n_orb, self.n_orb), dtype=complex)

        # Variables to store eigenenergies and eigenstates
        self.energy_gs = None
        self.eigenenergies_e = None
        self.eigenstates_e = None
        self.eigenenergies_h = None
        self.eigenstates_h = None

        self.circuit_constructor = None

    def compute_ground_state(self,
                             save_params: bool = False,
                             load_params: bool = False
                             ) -> None:
        """Calculates the ground state of the Hamiltonian using VQE.

        Args:
            save_params: Whether to save the VQE energy and ansatz circuit.
            load_params: Whether to load the VQE energy and ansatz circuit.
        """
        if load_params:
            print("Load VQE circuit from file")
            with open('data/vqe_energy.txt', 'r') as f: 
                self.energy_gs = float(f.read())
            with open('circuits/vqe_circuit.txt') as f:
                self.ansatz = QuantumCircuit.from_qasm_str(f.read())
        else:
            print("===== Start calculating the ground state using VQE =====")
            self.energy_gs, self.ansatz = vqe_minimize(self.qiskit_op_gs, q_instance=self.q_instance)
            self.energy_gs *= HARTREE_TO_EV
            print(f'Ground state energy = {self.energy_gs:.3f} eV')
            print("===== Finish calculating the ground state using VQE =====")

            if save_params:
                print("Save VQE circuit to file")
                with open('data/vqe_energy.txt', 'w') as f: 
                    f.write(str(self.energy_gs))
                save_circuit(self.ansatz.copy(), 'circuits/vqe_circuit')

        self.circuit_constructor = CircuitConstructor(
            self.ansatz, add_barriers=self.add_barriers, ccx_data=self.ccx_data)

    def compute_eh_states(self) -> None:
        """Calculates (N±1)-electron states of the Hamiltonian."""

        method = 'exact'

        if method == 'exact':
            print("===== Start calculating (N±1)-electron states =====")
            self.eigenenergies_e, self.eigenstates_e = number_state_eigensolver(
                self.qiskit_op_spin.to_matrix(), inds=self.inds_e.int_form)
            self.eigenenergies_h, self.eigenstates_h = number_state_eigensolver(
                self.qiskit_op_spin.to_matrix(), inds=self.inds_h.int_form)
            print("===== Finish calculating (N±1)-electron states =====")

        elif method == 'qse':
            def a_op(ind, spin, dag):
                a_op_ = (self.pauli_op_dict[(ind, spin)][0] +  (-1) ** dag * self.pauli_op_dict[(ind, spin)][1]) / 2
                a_op_ = PauliSumOp(a_op_).reduce()
                return a_op_
            qse_ops_e = [a_op(0, 'down', True), a_op(1, 'down', True)]
            qse_ops_h = [a_op(0, 'down', False), a_op(1, 'down', False)]

            # The eigenstates below are wrong. But if using energy 
            # averaging method the eigenstates do not matter.
            print("===== Start calculating (N+1)-electron states =====")
            self.eigenenergies_e, self.eigenstates_e = quantum_subspace_expansion(
                self.ansatz.copy(), self.qiskit_op, qse_ops_e, q_instance=self.q_instance)
            self.eigenenergies_h, self.eigenstates_h = quantum_subspace_expansion(
                self.ansatz.copy(), self.qiskit_op, qse_ops_h, q_instance=self.q_instance)
            print("===== Finish calculating (N±1)-electron states =====")

        self.eigenenergies_e *= HARTREE_TO_EV
        self.eigenenergies_h *= HARTREE_TO_EV
        
        print(f"(N+1)-electron energies are {self.eigenenergies_e} eV")
        print(f"(N-1)-electron energies are {self.eigenenergies_h} eV")


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
        inds_e = self.inds_e.include_ancilla('1').int_form
        inds_h = self.inds_h.include_ancilla('0').int_form
        for m in range(self.n_orb):
            a_op_m = self.pauli_op_dict_tapered[(m, self.spin)]
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

            if self.q_instance.backend.name() == 'statevector_simulator':
                result = self.q_instance.execute(circ)
                psi = result.get_statevector()

                psi_e = psi[inds_e]
                B_e_mm = np.abs(self.eigenstates_e.conj().T @ psi_e) ** 2

                psi_h = psi[inds_h]
                B_h_mm = np.abs(self.eigenstates_h.conj().T @ psi_h) ** 2

            else: # QASM simulator or hardware
                if self.method == 'energy':
                    circ_anc = circ.copy()
                    circ_anc.add_register(ClassicalRegister(1))
                    circ_anc.measure(0, 0)
                    q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8000)
                    result = q_instance.execute(circ_anc)
                    counts = result.get_counts()
                    shots = sum(counts.values())
                    p_e = counts['1'] / shots
                    p_h = counts['0'] / shots

                    energy_e = measure_operator(
                        circ.copy(), self.qiskit_op_down, q_instance=self.q_instance, anc_state=[1])
                    energy_h = measure_operator(
                        circ.copy(), self.qiskit_op_down, q_instance=self.q_instance, anc_state=[0])

                    # print(f'energy_e = {energy_e * HARTREE_TO_EV:.6f} eV')
                    # print(f'energy_h = {energy_h * HARTREE_TO_EV:.6f} eV')

                    B_e_mm = p_e * solve_energy_probabilities(self.eigenenergies_e, energy_e * HARTREE_TO_EV)
                    B_h_mm = p_h * solve_energy_probabilities(self.eigenenergies_h, energy_h * HARTREE_TO_EV)
                    
                    #print(B_e_mm)
                    #print(B_h_mm)
                    #print(sum(B_e_mm) + sum(B_h_mm))
                    #exit()
                elif self.method == 'tomography':
                    rho = state_tomography(circ, q_instance=self.q_instance)

                    rho_e = rho[inds_e][:, inds_e]
                    B_e_mm = np.diag(self.eigenstates_e.conj().T @ rho_e @ self.eigenstates_e).real

                    rho_h = rho[inds_h][:, inds_h]
                    B_h_mm = np.diag(self.eigenstates_h.conj().T @ rho_h @ self.eigenstates_h).real

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
        inds_hp = self.inds_h.include_ancilla('00').int_form
        inds_hm = self.inds_h.include_ancilla('10').int_form
        inds_ep = self.inds_e.include_ancilla('01').int_form
        inds_em = self.inds_e.include_ancilla('11').int_form
        for m in range(self.n_orb):
            a_op_m = self.pauli_op_dict_tapered[(m, self.spin)]
            for n in range(self.n_orb):
                if m < n:
                    a_op_n = self.pauli_op_dict_tapered[(n, self.spin)]
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

                    if self.q_instance.backend.name() == 'statevector_simulator':
                        result = self.q_instance.execute(circ)
                        psi = result.get_statevector()

                        psi_ep = psi[inds_ep]
                        psi_em = psi[inds_em]
                        D_ep_mn = abs(self.eigenstates_e.conj().T @ psi_ep) ** 2
                        D_em_mn = abs(self.eigenstates_e.conj().T @ psi_em) ** 2

                        psi_hp = psi[inds_hp]
                        psi_hm = psi[inds_hm]
                        D_hp_mn = abs(self.eigenstates_h.conj().T @ psi_hp) ** 2
                        D_hm_mn = abs(self.eigenstates_h.conj().T @ psi_hm) ** 2

                    else: # QASM simulator or hardware
                        if self.method == 'energy':
                            D_ep_mn = D_em_mn = D_hp_mn = D_hm_mn = 0.
                        elif self.method == 'tomography':
                            rho = state_tomography(circ, q_instance=self.q_instance)

                            rho_ep = rho[inds_ep][:, inds_ep]
                            rho_em = rho[inds_em][:, inds_em]
                            D_ep_mn = np.diag(self.eigenstates_e.conj().T @ rho_ep @ self.eigenstates_e).real
                            D_em_mn = np.diag(self.eigenstates_e.conj().T @ rho_em @ self.eigenstates_e).real

                            rho_hp = rho[inds_hp][:, inds_hp]
                            rho_hm = rho[inds_hm][:, inds_hm]
                            D_hp_mn = np.diag(self.eigenstates_h.conj().T @ rho_hp @ self.eigenstates_h).real
                            D_hm_mn = np.diag(self.eigenstates_h.conj().T @ rho_hm @ self.eigenstates_h).real

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
            compute_energies: bool = True,
            save_params: bool = True,
            load_params: bool = False,
            cache_write: bool = True,
            cache_read: bool = False
            ) -> None:
        """Main function to compute energies and transition amplitudes.

        Args:
            compute_energies: Whether to compute ground- and (N±1)-electron state energies.
            save_params: Whether to save the VQE energy and ansatz parameters.
            load_params: Whether to load the VQE energy and ansatz parameters.
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        # TODO: Some if conditions need to be checked.
        if compute_energies:
            if self.energy_gs is None:
                self.compute_ground_state(save_params=save_params, load_params=load_params)
            if (self.eigenenergies_e is None or self.eigenenergies_h is None or
                self.eigenstates_e is None or self.eigenstates_h is None):
                self.compute_eh_states()
        self.compute_diagonal_amplitudes(cache_read=cache_read, cache_write=cache_write)
        self.compute_off_diagonal_amplitudes(cache_read=cache_read, cache_write=cache_write)


    def get_density_matrix(self):
        """Returns the density matrix from the hole-added part of the Green's function"""
        self.rho_gf = np.sum(self.B_h, axis=2)
        return self.rho_gf

    def compute_greens_function(self,
                                omega: Union[float, complex]
                                ) -> np.ndarray:
        """Calculates the values of the Green's function at frequency omega.

        Args:
            omega: The real or complex frequency at which the Green's function is calculated.

        Returns:
            The Green's function in numpy array form.
        """
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                self.G_e[m, n] = 2 * np.sum(
                    self.B_e[m, n] / (omega + self.energy_gs - self.eigenenergies_e))
                self.G_h[m, n] = 2 * np.sum(
                    self.B_h[m, n] / (omega - self.energy_gs + self.eigenenergies_h))
        self.G = self.G_e + self.G_h
        return self.G

    def compute_spectral_function(self,
                                  omega: Union[float, complex]
                                  ) -> np.ndarray:
        """Calculates the spectral function at frequency omega.

        Args:
            omega: The real or complex frequency at which the spectral function is calculated.
        """
        self.compute_greens_function(omega)
        A = -1 / np.pi * np.imag(np.trace(self.G))
        return A

    def compute_self_energy(self, omega: Union[float, complex]) -> np.ndarray:
        """Calculates the self-energy at frequency omega.

        Args:
            omega: The real or complex frequency at which the self-energy is calculated.

        Returns:
            Sigma: The self-energy numpy array.
        """
        self.compute_greens_function(omega)

        G_HF = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for i in range(self.n_orb):
            G_HF[i, i] = 1 / (omega - self.molecule.orbital_energies[i // 2] * HARTREE_TO_EV)

        Sigma = np.linalg.pinv(G_HF) - np.linalg.pinv(self.G)
        return Sigma

    def compute_correlation_energy(self) -> Tuple[complex, complex]:
        """Calculates the correlation energy.

        Returns:
            E1: The one-electron part (?) of the correlation energy.
            E2: The two-electron part (?) of the correlation energy.
        """

        self.rho_hf = np.diag([1] * self.n_occ + [0] * self.n_vir)
        self.rho_gf = np.sum(self.B_h, axis=2)

        # XXX: Now filling in the one-electron integrals with a
        # checkerboard pattern. Could be done more efficiently.
        self.h = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for i in range(self.n_orb):
            # print('i =', i)
            # print(self.molecule.one_body_integrals[i // 2])
            tmp = np.vstack((self.molecule.one_body_integrals[i // 2],
                            np.zeros((self.n_orb // 2, ))))
            self.h[i] = tmp[::(-1) ** (i % 2)].reshape(self.n_orb, order='f')
        self.h *= HARTREE_TO_EV

        E1 = np.trace((self.h + self.e_orb) @
                      (self.rho_gf - self.rho_hf)) / 2

        E2 = 0
        for i in range(self.n_h):
            # print('i =', i)
            e_qp = self.energy_gs - self.eigenenergies_h[i]
            Sigma = self.compute_self_energy(e_qp + 0.000002 * HARTREE_TO_EV * 1j)
            E2 += np.trace(Sigma @ self.B_h[:,:,i]) / 2
        return E1, E2