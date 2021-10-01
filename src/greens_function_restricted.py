"""GreensFunctionRestricted class"""

from typing import Union, Tuple, Optional
from greens_function import GreensFunction

import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B
from qiskit.utils import QuantumInstance

from constants import HARTREE_TO_EV
from greens_function import GreensFunction
from hamiltonians import MolecularHamiltonian
from number_state_solvers import number_state_eigensolver
from io_utils import load_vqe_result, save_vqe_result
from operators import get_operator_dictionary
from qubit_indices import QubitIndices
#from circuits import build_diagonal_circuits, build_off_diagonal_circuits
from circuits import CircuitConstructor, CircuitData
from z2_symmetries import transform_4q_hamiltonian
from utils import state_tomography

np.set_printoptions(precision=6)
pauli_op_dict = get_operator_dictionary()

class GreensFunctionRestricted:
    """A class to perform frequency-domain Green's function calculations
    with restricted orbitals."""
    def __init__(self, 
                 ansatz: QuantumCircuit, 
                 hamiltonian: MolecularHamiltonian, 
                 optimizer: Optional[Optimizer] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 recompiled: bool = True,
                 add_barriers: bool = True,
                 cxc_data: Optional[CircuitData] = None) -> None:
        """Initializes a GreensFunctionRestricted object.
        
        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The MolecularHamiltonian object.
            optimizer: The optimizer in VQE. Default to L_BFGS_B().
            q_instance: The QuantumInstance to execute the circuit.
            recompiled: Whether the QPE circuit is recompiled.
        """
        # Define Hamiltonian and related variables
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.molecule = hamiltonian.molecule
        self.e_orb = np.diag(self.molecule.orbital_energies) * HARTREE_TO_EV
        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        self.qiskit_op_gs = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 1])
        self.qiskit_op_up = transform_4q_hamiltonian(self.qiskit_op, init_state=[0, 1])

        self.optimizer = L_BFGS_B() if optimizer is None else optimizer
        if q_instance is None:
            self.q_instance = QuantumInstance(
                Aer.get_backend('statevector_simulator'))
        else:
            self.q_instance = q_instance
        self.recompiled = recompiled

        self.add_barriers = add_barriers
        self.cxc_data = cxc_data

        # Number of orbitals and indices
        self.n_orb = len(self.hamiltonian.act_inds)
        self.n_occ = (self.hamiltonian.molecule.n_electrons // 2
                      - len(self.hamiltonian.occ_inds))
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

        self.inds_h = QubitIndices(['10', '00'], n_qubits=2)
        self.inds_e = QubitIndices(['11', '01'], n_qubits=2)

        # Define the transition amplitude matrices
        self.B_e = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.D_em = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)

        self.B_h = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)

        # Define the density matrices for obtaining the self-energy
        self.rho_hf = np.diag([1] * self.n_occ + [0] * self.n_vir)
        self.rho_gf = None

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
                             load_params: bool = False,
                             prefix: str = None
                             ) -> None:
        """Calculates the ground state of the Hamiltonian using VQE.
        
        Args:
            save_params: Whether to save the VQE energy and ansatz parameters.
            load_params: Whether to load the VQE energy and ansatz parameters.
            prefix: A string specifying the prefix for the saved files.
        """
        if prefix is None:
            prefix = 'vqe'
        if load_params:
            print("Load ground state energy and ansatz from file")
            self.energy_gs, self.ansatz = load_vqe_result(self.ansatz, prefix=prefix)
        else:
            print("Start calculating the ground state using VQE")
            vqe = VQE(self.ansatz, optimizer=self.optimizer, 
                      quantum_instance=Aer.get_backend('statevector_simulator'))
            vqe_result = vqe.compute_minimum_eigenvalue(self.qiskit_op_gs)
            if save_params:
                save_vqe_result(vqe_result, prefix=prefix)
            self.energy_gs = vqe_result.optimal_value * HARTREE_TO_EV
            self.ansatz.assign_parameters(vqe_result.optimal_parameters, inplace=True)
            print("Finish calculating the ground state using VQE")

        self.circuit_constructor = CircuitConstructor(
            self.ansatz, add_barriers=self.add_barriers, cxc_data=self.cxc_data)
        print(f'Ground state energy = {self.energy_gs:.3f} eV')
    
    def compute_eh_states(self) -> None:
        """Calculates (N±1)-electron states of the Hamiltonian."""
        print("Start calculating (N±1)-electron states")
        self.eigenenergies_e, self.eigenstates_e = number_state_eigensolver(
            self.qiskit_op_up.to_matrix(), inds=self.inds_e.int_form)
        self.eigenenergies_h, self.eigenstates_h = number_state_eigensolver(
            self.qiskit_op_up.to_matrix(), inds=self.inds_h.int_form)
        self.eigenenergies_e *= HARTREE_TO_EV
        self.eigenenergies_h *= HARTREE_TO_EV
        print("Finish calculating (N±1)-electron states")

        print(f"(N+1)-electron energies are {self.eigenenergies_e}")
        print(f"(N-1)-electron energies are {self.eigenenergies_h}")

    def compute_diagonal_amplitudes(self,
                                    cache_read: bool = True,
                                    cache_write: bool = True
                                    ) -> None:
        """Calculates diagonal transition amplitudes.
        
        Args:
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("Start calculating diagonal transition amplitudes")

        inds_e = self.inds_e.include_ancilla('1').int_form
        inds_h = self.inds_h.include_ancilla('0').int_form
        for m in range(self.n_orb):
            print(f"Calculating m = {m}")
            a_op_m = pauli_op_dict[m]
            circ = self.circuit_constructor.build_diagonal_circuits(a_op_m)
            if self.q_instance.backend.name() == 'statevector_simulator':
                result = self.q_instance.execute(circ)
                psi = result.get_statevector()
                
                psi_e = psi[inds_e]
                B_e_mm = np.abs(self.eigenstates_e.conj().T @ psi_e) ** 2

                psi_h = psi[inds_h]
                B_h_mm = np.abs(self.eigenstates_h.conj().T @ psi_h) ** 2

            else: # QASM simulator or hardware
                rho = state_tomography(circ, q_instance=self.q_instance)

                rho_e = rho[inds_e][:, inds_e]
                B_e_mm = np.diag(self.eigenstates_e.conj().T @ rho_e @ self.eigenstates_e).real

                rho_h = rho[inds_h][:, inds_h]
                B_h_mm = np.diag(self.eigenstates_h.conj().T @ rho_h @ self.eigenstates_h).real
            
            #B_e_mm[abs(B_e_mm) < 1e-8] = 0.
            self.B_e[m, m] = B_e_mm

            #B_h_mm[abs(B_h_mm) < 1e-8] = 0.
            self.B_h[m, m] = B_h_mm

            print(f'B_e[{m}, {m}] = {self.B_e[m, m]}')
            print(f'B_h[{m}, {m}] = {self.B_h[m, m]}')
        print("Finish calculating diagonal transition amplitudes")

    def compute_off_diagonal_amplitudes(self, 
                                        cache_read: bool = True, 
                                        cache_write: bool = True
                                        ) -> None:
        """Calculates off-diagonal transition amplitudes.

        Args:
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("Start calculating off-diagonal transition amplitudes")

        inds_hp = self.inds_h.include_ancilla('00').int_form
        inds_hm = self.inds_h.include_ancilla('10').int_form
        inds_ep = self.inds_e.include_ancilla('01').int_form
        inds_em = self.inds_e.include_ancilla('11').int_form
        for m in range(self.n_orb):
            a_op_m = pauli_op_dict[m]
            for n in range(self.n_orb):
                if m != n:
                    print(f"Calculating m = {m}, n = {n}")
                    a_op_n = pauli_op_dict[n]
                    circ = self.circuit_constructor.build_off_diagonal_circuits(a_op_m, a_op_n)
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
                        rho = state_tomography(circ, q_instance=self.q_instance)

                        rho_ep = rho[inds_ep][:, inds_ep]
                        rho_em = rho[inds_em][:, inds_em]
                        D_ep_mn = np.diag(self.eigenstates_e.conj().T @ rho_ep @ self.eigenstates_e).real
                        D_em_mn = np.diag(self.eigenstates_e.conj().T @ rho_em @ self.eigenstates_e).real

                        rho_hp = rho[inds_hp][:, inds_hp]
                        rho_hm = rho[inds_hm][:, inds_hm]
                        D_hp_mn = np.diag(self.eigenstates_h.conj().T @ rho_hp @ self.eigenstates_h).real
                        D_hm_mn = np.diag(self.eigenstates_h.conj().T @ rho_hm @ self.eigenstates_h).real
                    
                    #D_ep_mn[abs(D_ep_mn) < 1e-8] = 0.
                    #D_em_mn[abs(D_em_mn) < 1e-8] = 0.
                    self.D_ep[m, n] = D_ep_mn
                    self.D_em[m, n] = D_em_mn

                    #D_hp_mn[abs(D_hp_mn) < 1e-8] = 0.
                    #D_hm_mn[abs(D_hm_mn) < 1e-8] = 0.
                    self.D_hp[m, n] = D_hp_mn
                    self.D_hm[m, n] = D_hm_mn

        # Unpack D values to B values
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    B_e_mn = np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n])
                    B_e_mn += np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m])
                    B_e_mn[abs(B_e_mn) < 1e-8] = 0
                    self.B_e[m, n] = B_e_mn

                    B_h_mn = np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n])
                    B_h_mn += np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m])
                    B_h_mn[abs(B_h_mn) < 1e-8] = 0
                    self.B_h[m, n] = B_h_mn

                    print(f'B_e[{m}, {n}] = {self.B_e[m, n]}')
                    print(f'B_h[{m}, {n}] = {self.B_h[m, n]}')
        
        print("Finish calculating off-diagonal transition amplitudes")

    @property
    def density_matrix(self):
        """Obtains the density matrix from the hole-added part of the Green's
        function"""
        self.rho_gf = np.sum(self.B_h, axis=2)
        return self.rho_gf

    def run(self, 
            compute_energies: bool = True,
            save_params: bool = True,
            load_params: bool = False,
            cache_write: bool = True,
            cache_read: bool = False
            ) -> None:
        """Main function to compute energies and transition amplitudes.
        
        Args:
            compute_energies: Whether to compute ground- and (N±1)-electron 
                state energies.
            save_params: Whether to save the VQE energy and ansatz parameters.
            load_params: Whether to load the VQE energy and ansatz parameters.
            cache_read: Whether to read recompiled circuits from io_utils files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        # TODO: Some if conditions need to be checked.
        if compute_energies:
            if self.energy_gs is None:
                self.compute_ground_state(save_params=save_params, 
                                          load_params=load_params)
            if (self.eigenenergies_e is None or self.eigenenergies_h is None or 
                self.eigenstates_e is None or self.eigenstates_h is None):
                self.compute_eh_states()
        self.compute_diagonal_amplitudes(
            cache_read=cache_read, cache_write=cache_write)
        self.compute_off_diagonal_amplitudes(
            cache_read=cache_read, cache_write=cache_write)

    def compute_greens_function(self, 
                                omega: Union[float, complex]
                                ) -> np.ndarray:
        """Calculates the values of the Green's function at frequency omega.
        
        Args:
            omega: The real or complex frequency at which the Green's function
                is calculated.
                
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
            omega: The real or complex frequency at which the spectral 
                function is calculated.
        """
        #print("Start calculating the spectral function")
        self.compute_greens_function(omega)
        A = -1 / np.pi * np.imag(np.trace(self.G))
        #print("Finish calculating the spectral function")
        return A

    def compute_self_energy(self, omega: Union[float, complex]) -> np.ndarray:
        """Calculates the self-energy at frequency omega.
        
        Args:
            omega: The real or complex frequency at which the self-energy 
                is calculated.
            
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
        self.get_density_matrix()

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