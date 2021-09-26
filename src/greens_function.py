"""GreensFunction class"""

from typing import Union, Tuple, Optional

import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, Aer, ClassicalRegister
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B
from qiskit.utils import QuantumInstance

from constants import HARTREE_TO_EV
from hamiltonians import MolecularHamiltonian
from number_state_solvers import (get_number_state_indices,
                                  number_state_eigensolver)
from tools import (get_a_operator,
                   load_vqe_result,
                   save_vqe_result)
from circuits import append_qpe_circuit, build_diagonal_circuits, build_off_diagonal_circuits

np.set_printoptions(precision=6)

class GreensFunction:
    """A class to perform frequency-domain Green's Function calculations."""
    def __init__(self, 
                 ansatz: QuantumCircuit, 
                 hamiltonian: MolecularHamiltonian, 
                 optimizer: Optional[Optimizer] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 scaling: float = 1.,
                 shift: float = 0.,
                 recompiled: bool = True,
                 states: Optional[str] = None) -> None:
        """Initializes a GreensFunction object.
        
        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The MolecularHamiltonian object.
            optimizer: The optimizer in VQE. Default to L_BFGS_B().
            q_instance: The quantum instance for execution of the quantum 
                circuit. Default to statevector simulator.
            scaling: Scaling factor of the Hamiltonian.
            shift: Constant shift factor of the Hamiltonian.
            recompiled: Whether the QPE circuit is recompiled or not.
            states: 'e' or 'h' indicating whether (N+1)- or (N-1)-electron
                states are to be calculated.
        """
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.molecule = hamiltonian.molecule

        # Extract the orbital energies
        e_orb = self.molecule.orbital_energies
        e_orb = np.vstack([e_orb, e_orb]).reshape(e_orb.shape[0] * 2, order='f')
        self.e_orb = np.diag(e_orb) * HARTREE_TO_EV

        # Extract the Hamiltonian operators and apply shift and scaling factors
        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        self.qiskit_op *= scaling
        self.qiskit_op.primitive.coeffs[0] += shift / HARTREE_TO_EV
        self.hamiltonian_arr = self.qiskit_op.to_matrix()
        
        self.optimizer = L_BFGS_B() if optimizer is None else optimizer
        if q_instance is None:
            self.q_instance = QuantumInstance(
                Aer.get_backend('statevector_simulator'))
        else:
            self.q_instance = q_instance
        self.recompiled = recompiled

        # Number of orbitals and indices
        self.n_orb = 2 * len(self.hamiltonian.active_inds)
        self.n_qubits = self.n_orb # XXX
        self.n_occ = (self.hamiltonian.molecule.n_electrons 
                      - 2 * len(self.hamiltonian.occupied_inds))
        self.n_vir = self.n_orb - self.n_occ
        self.inds_occ = list(range(self.n_occ))
        self.inds_vir = list(range(self.n_occ, self.n_orb))
        self.n_e = int(binom(self.n_orb, self.n_occ + 1))
        self.n_h = int(binom(self.n_orb, self.n_occ - 1))

        # Define the transition amplitude matrices
        self.B_e = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.B_h = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h))
        self.D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h))
        self.D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e))
        self.D_em = np.zeros((self.n_orb, self.n_orb, self.n_e))

        # Define the density matrices for obtaining the self-energy
        self.rho_hf = np.diag([1] * self.n_occ + [0] * self.n_vir)
        self.rho_gf = None

        # Define the Green's function arrays
        self.G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G = np.zeros((self.n_orb, self.n_orb), dtype=complex)

        # Define the arrays indicating whether the (N±1)-electron states
        # are to be calculated
        self.states = states
        self.states_arr = np.empty((self.n_orb, self.n_orb), dtype=np.object)
        # print('states_arr\n', self.states_arr)
        # self.states_arr = np.array([[[None] for i in range(self.n_orb)] for j in range(self.n_orb)])

        # Variables to store eigenenergies and eigenstates
        self.energy_gs = None
        self.eigenenergies_e = None
        self.eigenstates_e = None
        self.eigenenergies_h = None
        self.eigenstates_h = None

    def compute_ground_state(self, 
                             save_params: bool = False,
                             load_params: bool = False,
                             prefix: str = None
                             ) -> Optional[QuantumCircuit]:
        """Calculates the ground state of the Hamiltonian using VQE.
        
        Args:
            save_params: Whether to save the VQE energy and ansatz parameters.
            load_params: Whether to load the VQE energy and ansatz parameters.
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
            vqe_result = vqe.compute_minimum_eigenvalue(self.qiskit_op)
            if save_params:
                save_vqe_result(vqe_result, prefix=prefix)
            self.energy_gs = vqe_result.optimal_value * HARTREE_TO_EV
            self.ansatz.assign_parameters(vqe_result.optimal_parameters, inplace=True)
            print("Finish calculating the ground state using VQE")
        
        print(f'Ground state energy = {self.energy_gs:.3f} eV')


    def compute_eh_states(self) -> None:
        """Calculates (N±1)-electron states of the Hamiltonian."""
        print("Start calculating (N±1)-electron states")
        self.eigenenergies_e, self.eigenstates_e = \
            number_state_eigensolver(self.hamiltonian_arr, self.n_occ + 1, reverse=True)
        self.eigenenergies_h, self.eigenstates_h = \
            number_state_eigensolver(self.hamiltonian_arr, self.n_occ - 1, reverse=True)
        self.eigenenergies_e *= HARTREE_TO_EV
        self.eigenenergies_h *= HARTREE_TO_EV
        print("eigenstates_e\n", self.eigenstates_e)
        print("eigenstates_h\n", self.eigenstates_h)

        '''
        eigenenergies_2, eigenstates_2 = number_state_eigensolver(self.hamiltonian_arr, 2)
        if True:
            np.set_printoptions(precision=2)
            eigenstates = np.zeros((16, 6), dtype=complex)
            inds = get_number_state_indices(4, 2)
            for j in range(6):
                eigenstates[inds, j] = eigenstates_2[:, j]
            ZIZI = Pauli('ZIZI').to_matrix()
            IZIZ = Pauli('IZIZ').to_matrix()

            res = eigenstates.conj().T @ ZIZI @ eigenstates
            res[abs(res) < 1e-8] = 0
            print('ZIZI values')
            print(res)

            res = eigenstates.conj().T @ IZIZ @ eigenstates
            res[abs(res) < 1e-8] = 0
            print('IZIZ values')
            print(res)
        

        # print(self.eigenstates_e.shape)
        if True:
            eigenstates = np.zeros((16, 4), dtype=complex)
            inds = get_number_state_indices(4, 3)
            inds = [15 - k for k in inds]
            for j in range(4):
                eigenstates[inds, j] = self.eigenstates_e[:, j]
            ZIZI = Pauli('ZIZI').to_matrix()
            IZIZ = Pauli('IZIZ').to_matrix()
            print('ZIZI values\n')
            print(eigenstates.conj().T @ ZIZI @ eigenstates)
            print('IZIZ values\n')
            print(eigenstates.conj().T @ IZIZ @ eigenstates)
        print("Finish calculating (N±1)-electron states")
    '''

    def compute_diagonal_amplitudes(self,
                                    cache_read: bool = True,
                                    cache_write: bool = True) -> None:
        """Calculates diagonal transition amplitudes.
        
        Args:
            cache_read: Whether to read recompiled circuits from cache files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("Start calculating diagonal transition amplitudes")
        for m in range(self.n_orb):
            print('-' * 30, f'm = {m}', '-' * 30)
            a_op_m = get_a_operator(self.n_qubits, m)
            circ = build_diagonal_circuits(self.ansatz.copy(), a_op_m)
            if self.states is None or self.states in self.states_arr[m, m]:
                if self.q_instance.backend.name() == 'statevector_simulator':
                    result = self.q_instance.execute(circ)
                    psi = result.get_statevector()
                    print('Index of largest element is', format(np.argmax(np.abs(psi)), '05b'))
                    
                    if self.states in ('e', None):
                        inds_e = get_number_state_indices(
                            self.n_orb, self.n_occ + 1, anc='1', reverse=True)
                        probs_e = np.abs(self.eigenstates_e.conj().T @ psi[inds_e]) ** 2
                        probs_e[abs(probs_e) < 1e-8] = 0.
                        self.B_e[m, m] = probs_e

                    if self.states in ('h', None):                     
                        inds_h = get_number_state_indices(
                            self.n_orb, self.n_occ - 1, anc='0', reverse=True)
                        probs_h = np.abs(self.eigenstates_h.conj().T @ psi[inds_h]) ** 2
                        probs_h[abs(probs_h) < 1e-8] = 0.
                        self.B_h[m, m] = probs_h

                    if self.states is None:
                        states_elem = []
                        if sum(probs_h) > 1e-3:
                            states_elem.append('h')
                        if sum(probs_e) > 1e-3:
                            states_elem.append('e')
                        self.states_arr[m, m] = states_elem

                else: # QASM simulator or hardware
                    circ = append_qpe_circuit(circ, self.qiskit_op.to_matrix(), 1, 
                                              recompiled=self.recompiled, 
                                              n_gate_rounds=6,
                                              cache_options = 
                                              {'read': cache_read, 'write': cache_write, 
                                               'hamiltonian': self.hamiltonian,
                                               'index': str(m), 'states': self.states})
                    circ.add_register(ClassicalRegister(2))
                    circ.measure([0, 1], [0, 1])
                    
                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    for key in list(counts):
                        # If calculating (N-1)-electron states, need the first
                        # qubit to be in |0>
                        if int(key[-1]) == (self.states == 'h'):
                            del counts[key]
                    shots = sum(counts.values())

                    probs = {}
                    for t in ['0', '1']:
                        key = ''.join(t) + ('0' if self.states == 'h' else '1')
                        if key in counts.keys():
                            probs[key] = counts[key] / shots
                        else:
                            probs[key] = 0.0
                    if self.states == 'h':
                        self.B_h[m, m] = [probs['00'], 0., probs['10'], 0.]
                    
                    if self.states == 'e':
                        self.B_e[m, m] = [probs['01'], 0., probs['11'], 0.]
            print(f'B_e[{m}, {m}] = {self.B_e[m, m]}')
            print(f'B_h[{m}, {m}] = {self.B_h[m, m]}')
        print("Finish calculating diagonal transition amplitudes")

    def compute_off_diagonal_amplitudes(self, 
                                        cache_read: bool = True, 
                                        cache_write: bool = True) -> None:
        """Calculates off-diagonal transition amplitudes.

        Args:
            cache_read: Whether to read recompiled circuits from cache files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("Start calculating off-diagonal transition amplitudes")
        for m in range(self.n_orb):
            a_op_m = get_a_operator(self.n_qubits, m)
            for n in range(self.n_orb):
                a_op_n = get_a_operator(self.n_qubits, n)
                circ = build_off_diagonal_circuits(self.ansatz.copy(), 
                                                   a_op_m, a_op_n)
                if m != n and (self.states is None or 
                               self.states in self.states_arr[m, n]):
                    if self.q_instance.backend.name() == 'statevector_simulator':
                        result = self.q_instance.execute(circ)
                        psi = result.get_statevector()

                        if self.states in ('h', None):
                            inds_hp = get_number_state_indices(
                                self.n_orb, self.n_occ - 1, anc='00', reverse=True)
                            inds_hm = get_number_state_indices(
                                self.n_orb, self.n_occ - 1, anc='01', reverse=True)
                            probs_hp = abs(self.eigenstates_h.conj().T @ psi[inds_hp]) ** 2
                            probs_hm = abs(self.eigenstates_h.conj().T @ psi[inds_hm]) ** 2
                            self.D_hp[m, n] = probs_hp
                            self.D_hm[m, n] = probs_hm

                        if self.states in ('e', None):
                            inds_ep = get_number_state_indices(
                                self.n_orb, self.n_occ + 1, anc='10', reverse=True)
                            inds_em = get_number_state_indices(
                                self.n_orb, self.n_occ + 1, anc='11', reverse=True)
                            probs_ep = abs(self.eigenstates_e.conj().T @ psi[inds_ep]) ** 2
                            probs_em = abs(self.eigenstates_e.conj().T @ psi[inds_em]) ** 2
                            self.D_ep[m, n] = probs_ep
                            self.D_em[m, n] = probs_em

                        if self.states is None:
                            states_elem = []
                            if sum(probs_hp) > 1e-3 or sum(probs_hm) > 1e-3:
                                states_elem.append('h')
                            if sum(probs_ep) > 1e-3 or sum(probs_em) > 1e-3:
                                states_elem.append('e')
                            self.states_arr[m, n] = states_elem

                    else:
                        circ = append_qpe_circuit(circ, self.qiskit_op.to_matrix(), 2, 
                                                recompiled=self.recompiled, 
                                                n_gate_rounds=7,
                                                cache_options = 
                                                {'read': cache_read, 'write': cache_write, 
                                                'hamiltonian': self.hamiltonian,
                                                'index': str(m), 'states': self.states})
                        circ.add_register(ClassicalRegister(3))
                        circ.measure([0, 1, 2], [0, 1, 2])

                        result = self.q_instance.execute(circ)
                        counts = result.get_counts()
                        for key in list(counts):
                            if int(key[-1]) == (self.states == 'h'):
                                del counts[key]
                        shots = sum(counts.values())
                        
                        from itertools import product
                        probs = {}
                        for t in product(['0', '1'], repeat=2):
                            key = ''.join(t) + ('0' if self.states == 'h' else '1')
                            if key in counts.keys():
                                probs[key] = counts[key] / shots / len(self.states_arr[m, n]) # XXX
                            else:
                                probs[key] = 0.0

                        if self.states == 'h':
                            self.D_hp[m, n] = [probs['000'], 0.0, probs['100'], 0.0]
                            self.D_hm[m, n] = [probs['010'], 0.0, probs['110'], 0.0]
                        
                        if self.states == 'e':
                            self.D_ep[m, n] = [probs['001'], 0., probs['101'], 0.]
                            self.D_em[m, n] = [probs['011'], 0., probs['111'], 0.]
                        print('')

        # Unpack D values to B values
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    print('-' * 30, f'm = {m}, n = {n}', '-' * 30)
                    B_e_mn = np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n])
                    B_e_mn += np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m])
                    B_e_mn[abs(B_e_mn) < 1e-8] = 0
                    self.B_e[m, n] = B_e_mn

                    B_h_mn = np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n])
                    B_h_mn = np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m])
                    B_h_mn[abs(B_h_mn) < 1e-8] = 0
                    self.B_h[m, n] = B_h_mn
                    print(f'B_e[{m}, {n}] = {self.B_e[m, n]}')
                    print(f'B_h[{m}, {n}] = {self.B_h[m, n]}')

        '''
        print('B_e\n')
        #self.B_e[abs(self.B_e) < 1e-8] = 0.
        B_e = self.B_e.copy()
        for m in [0, 2]:
            for m_ in range(B_e.shape[1]):
                for lam in range(B_e.shape[2]):
                    if abs(B_e[m, m_, lam]) > 1e-4:
                        print(f'm = {m}, m\' = {m_}, lambda = {lam}: {B_e[m, m_, lam]:.6f}')
        '''

        print("Finish calculating off-diagonal transition amplitudes")

    # TODO: Write this as a property function
    def get_density_matrix(self):
        """Obtains the density matrix from the hole-added part of the Green's
        function"""
        self.rho_gf = np.sum(self.B_h, axis=2)
        return self.rho_gf

    def run(self, 
            compute_energies: bool = True,
            save_params: bool = True,
            load_params: bool = False,
            cache_write: bool = True,
            cache_read: bool = False) -> None:
        """Main function call to compute energies and transition amplitudes.
        
        Args:
            compute_energies: Whether to compute ground- and (N±1)-electron 
                state energies.
            save_params:
            load_params: 
            cache_read: Whether to read recompiled circuits from cache files.
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

    @classmethod
    def initialize_eh(cls, 
                      gf: 'GreensFunction', 
                      states: str,
                      q_instance: Optional[QuantumInstance] = None
                      ) -> 'GreensFunction':
        """Creates a GreensFunction object for calculating the (N±1)-electron
        states transition amplitudes.

        Args:
            gf: The GreensFunction from statevector simulator calculation.
            states: A string indicating whether the e or h states are 
                to be calculated. Must be 'e' or 'h'.
            q_instance: The QuantumInstance for executing the calculation.
        
        Returns:
            gf_new: The new GreensFunction object for calculating the 
                (N±1)-electron states.
        """
        assert states in ['e', 'h']
        if q_instance is None:
            q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
        
        # XXX: The 0 and 2 are hardcoded
        # Obtain the scaling factor
        gf_scaling = cls(None, gf.hamiltonian)
        gf_scaling.compute_eh_states()
        if states == 'h':
            E_low = gf_scaling.eigenenergies_h[0]
            E_high = gf_scaling.eigenenergies_h[2]
        elif states == 'e':
            E_low = gf_scaling.eigenenergies_e[0]
            E_high = gf_scaling.eigenenergies_e[2]
        scaling = np.pi / (E_high - E_low)

        # Obtain the shift factor
        gf_shift = cls(None, gf.hamiltonian, scaling=scaling)
        gf_shift.compute_eh_states()
        if states == 'h':
            shift = -gf_shift.eigenenergies_h[0]
        elif states == 'e':
            shift = -gf_shift.eigenenergies_e[0]

        # Create a new GreensFunction object with scaling and shift
        gf_new = cls(gf.ansatz.copy(), gf.hamiltonian, 
                     q_instance=q_instance, 
                     scaling=scaling, shift=shift)
        gf_new.states = states
        gf_new.states_arr = gf.states_arr
        # XXX: Do we really not need the eigenstates?
        if q_instance.backend.name() == 'statevector_simulator':
            if states == 'e':
                gf_new.eigenstates_e = gf.eigenstates_e
            elif states == 'h':
                gf_new.eigenstates_h = gf.eigenstates_h
        return gf_new

    @classmethod
    def initialize_final(cls, 
                         gf_sv: 'GreensFunction',
                         gf_e: 'GreensFunction',
                         gf_h: 'GreensFunction',
                         q_instance: Optional[QuantumInstance] = None
                         ) -> 'GreensFunction':
        """Creates a GreensFunction object for calculating final 
        observables.
        
        Args:
            gf_sv: The GreensFunction from statevector simulator calculation.
            gf_e: The GreensFunction for e states calculation.
            gf_h: The GreensFunction for h states calculation.
            q_instance: The QuantumInstance for executing the calculation.

        Returns:
            gf_new: The new GreensFunction object for calculating final 
                observables.
        """
        # Creates the new GreensFunction object
        if q_instance is None:
            q_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
        gf_new = cls(None, gf_sv.hamiltonian, q_instance=q_instance)

        # Assign the eigenenergies
        gf_new.energy_gs = gf_sv.energy_gs
        gf_new.eigenenergies_e = gf_sv.eigenenergies_e
        gf_new.eigenenergies_h = gf_sv.eigenenergies_h

        # Add the e states transition amplitudes
        gf_new.B_e += gf_e.B_e
        gf_new.D_ep += gf_e.D_ep
        gf_new.D_em += gf_e.D_em

        # Add the h states transition amplitudes
        gf_new.B_h += gf_h.B_h
        gf_new.D_hp += gf_h.D_hp
        gf_new.D_hm += gf_h.D_hm
        return gf_new

    def compute_greens_function(self, 
                                omega: Union[float, complex]
                                ) -> np.ndarray:
        """Calculates the values of the Green's function at frequency omega.
        
        Args:
            omega: The real or complex frequency at which the Green's function
                is calculated.
                
        Returns:
            self.G: The Green's function Numpy array.
        """
        # print("Start calculating the Green's function")
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                self.G_e[m, n] = np.sum(
                    self.B_e[m, n] / (omega + self.energy_gs - self.eigenenergies_e))
                self.G_h[m, n] = np.sum(
                    self.B_h[m, n] / (omega - self.energy_gs + self.eigenenergies_h))
        self.G = self.G_e + self.G_h
        # print("Finish calculating the Green's function")
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
        #print("Start calculating the self-energy")
        self.compute_greens_function(omega)

        G_HF = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for i in range(self.n_orb):
            G_HF[i, i] = 1 / (omega - self.molecule.orbital_energies[i // 2] * HARTREE_TO_EV)

        Sigma = np.linalg.pinv(G_HF) - np.linalg.pinv(self.G)
        #print("Finish calculating the self-energy")
        return Sigma

    def compute_correlation_energy(self) -> Tuple[complex, complex]:
        """Calculates the corelation energy.
        
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