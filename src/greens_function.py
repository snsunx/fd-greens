import numpy as np
from scipy.special import binom
from qiskit import *
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B
from qiskit.aqua import QuantumInstance
from qiskit.utils import QuantumInstance
from qiskit.extensions import UnitaryGate
from scipy.linalg import expm
from openfermion.linalg import get_sparse_operator
from math import pi
import copy
import pickle

from constants import *
from hamiltonians import MolecularHamiltonian
from tools import (get_number_state_indices,
                   number_state_eigensolver, 
                   reverse_qubit_order,
                   get_statevector)
from circuits import (build_diagonal_circuits, 
                      build_diagonal_circuits_with_qpe,
                      build_off_diagonal_circuits, 
                      build_off_diagonal_circuits_with_qpe)
from recompilation import recompile_with_statevector, apply_quimb_gates


class GreensFunction:
    def __init__(self, 
                 ansatz: QuantumCircuit, 
                 hamiltonian: MolecularHamiltonian, 
                 optimizer: Optimizer = None,
                 q_instance: QuantumInstance = None,
                 scaling_factor: float = 1.,
                 constant_shift: float = 0.,
                 recompiled: bool = True):
        """Initializes an object to calculate frequency-domain Green's function.
        
        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The Hamiltonian object.
            optimizer: The optimzier in VQE.
            q_instance: The quantum instance for execution of the quantum circuit.
            scaling_factor: Scaling factor of the Hamiltonian.
            constant_shift: Constant shift of the Hamiltonian.
            recompiled: Whether the QPE circuit is recompiled or not.
        """ 
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.molecule = hamiltonian.molecule

        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        self.openfermion_op = copy.deepcopy(self.hamiltonian.openfermion_op)
        self.qiskit_op *= scaling_factor
        self.qiskit_op.primitive.coeffs[0] += constant_shift / HARTREE_TO_EV
        self.openfermion_op *= scaling_factor
        self.openfermion_op.terms[()] += constant_shift / HARTREE_TO_EV
        self.hamiltonian_arr = get_sparse_operator(self.openfermion_op).toarray()

        self.recompiled = recompiled
        
        self.optimizer = L_BFGS_B() if optimizer is None else optimizer
        if q_instance is None:
            self.q_instance = QuantumInstance(
                Aer.get_backend('statevector_simulator'))
        else:
            self.q_instance = q_instance

        self.n_orb = 2 * len(self.hamiltonian.active_inds)
        self.n_occ = (self.hamiltonian.molecule.n_electrons 
                      - 2 * len(self.hamiltonian.occupied_inds))

        self.n_vir = self.n_orb - self.n_occ
        self.inds_occ = list(range(self.n_occ))
        self.inds_vir = list(range(self.n_occ, self.n_orb))
        self.n_e = int(binom(self.n_orb, self.n_occ + 1))
        self.n_h = int(binom(self.n_orb, self.n_occ - 1))

        self.B_e = np.zeros((self.n_orb, self.n_orb, self.n_e), dtype=complex)
        self.B_h = np.zeros((self.n_orb, self.n_orb, self.n_h), dtype=complex)
        self.G_e = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G_h = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        self.G = np.zeros((self.n_orb, self.n_orb), dtype=complex)

        self.D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h))
        self.D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h))
        self.D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e))
        self.D_em = np.zeros((self.n_orb, self.n_orb, self.n_e))

        self.states = None
        self.states_arr = np.empty((self.n_orb, self.n_orb), dtype=np.object)
        # self.states_arr = np.array([[[None] for i in range(self.n_orb)] for j in range(self.n_orb)])

        self.energies_e = None
        self.eigenstates_e = None
        self.energies_h = None
        self.eigenstates_h = None


    def compute_ground_state(self, save_params=True, return_ansatz=False):
        """Calculates the ground state of the Hamiltonian using VQE."""
        # TODO: Need to save the coefficient somewhere instead of passing in 
        # statevector simulator all the time.
        vqe = VQE(self.ansatz, optimizer=self.optimizer, 
                  quantum_instance=Aer.get_backend('statevector_simulator'))
        result = vqe.compute_minimum_eigenvalue(self.qiskit_op)
        
        self.energy_gs = result.optimal_value * HARTREE_TO_EV
        self.ansatz.assign_parameters(result.optimal_parameters, inplace=True)
        if save_params:
            f = open('ansatz_params.pkl', 'wb')
            pickle.dump(result.optimal_parameters, f)
            f.close()
        print(f'Ground state energy = {self.energy_gs:.3f} eV')
        if return_ansatz:
            return self.ansatz
    

    def compute_eh_states(self):
        """Calculates (N+/-1)-electron states of the Hamiltonian."""
        self.energies_e, self.eigenstates_e = \
            number_state_eigensolver(self.hamiltonian_arr, self.n_occ + 1)
        self.energies_h, self.eigenstates_h = \
            number_state_eigensolver(self.hamiltonian_arr, self.n_occ - 1)
        self.energies_e *= HARTREE_TO_EV
        self.energies_h *= HARTREE_TO_EV
        print("Calculations of (N+/-1)-electron states finished.")

    def compute_Npm1_electron_states(self):
        self.compute_eh_states()


    def compute_diagonal_amplitudes(self, cache_read: bool = True, cache_write: bool = True):
        """Calculates diagonal transition amplitudes in the Green's function.
        
        Args:
           cache_read: Whether to attempt to read recompiled circuits from cache files.
           cache_write: Whether to save recompiled circuits to cache files.
        """
        
        for m in range(self.n_orb):
            if self.states is None or self.states in self.states_arr[m, m]:
                print('m =', m)
                if self.q_instance.backend.name() == 'statevector_simulator':
                    print("Statevector simulator mode.")
                    circ = build_diagonal_circuits(self.ansatz.copy(), m)
                    result = self.q_instance.execute(circ)
                    psi = reverse_qubit_order(result.get_statevector())
                    
                    if self.states in ('e', None):
                        inds_e = get_number_state_indices(self.n_orb, self.n_occ + 1, anc='1')
                        probs_e = np.abs(self.eigenstates_e.conj().T @ psi[inds_e]) ** 2
                        probs_e[abs(probs_e) < 1e-8] = 0.
                        # print('probs_e =', probs_e)
                        self.B_e[m, m] = probs_e

                    if self.states in ('h', None):                     
                        inds_h = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='0')
                        probs_h = np.abs(self.eigenstates_h.conj().T @ psi[inds_h]) ** 2
                        probs_h[abs(probs_h) < 1e-8] = 0.
                        # print('probs_h =', probs_h)
                        self.B_h[m, m] = probs_h

                    if self.states is None:
                        states_elem = []
                        if sum(probs_h) > 1e-3:
                            states_elem.append('h')
                        if sum(probs_e) > 1e-3:
                            states_elem.append('e')
                        self.states_arr[m, m] = states_elem

                else:
                    print("QASM simulator mode.") # XXX: Not necessarily QASM simulator
                    circ = build_diagonal_circuits_with_qpe(self.ansatz.copy(), m)
                    Umat = expm(1j * self.hamiltonian_arr * HARTREE_TO_EV)
                    if self.recompiled:
                        cUmat = np.kron(np.eye(2), 
                            np.kron(np.diag([1, 0]), np.eye(2 ** 4))
                            + np.kron(np.diag([0, 1]), Umat))

                        circ.barrier()
                        circ.h(1)
                        statevector = reverse_qubit_order(get_statevector(circ))
                        quimb_gates = recompile_with_statevector(
                            statevector, cUmat, n_gate_rounds=6,
                            cache_options = {'read': cache_read, 'write': cache_write, 
                                             'hamiltonian': self.hamiltonian,
                                             'type': f'greens-diag-({m})'})
                        circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
                        circ.h(1)
                        circ.barrier()

                        # circ_recompiled = get_statevector(circ)
                        # np.save('circ_recompiled.npy', circ_recompiled)
                    else:
                        cUgate = UnitaryGate(Umat).control(1)

                        circ.barrier()
                        circ.h(1)
                        circ.append(cUgate, [1, 5, 4, 3, 2])
                        circ.h(1)
                        circ.barrier()

                        # circ_unrecompiled = get_statevector(circ)
                        # np.save('circ_unrecompiled.npy', circ_unrecompiled)

                    circ.measure([0, 1], [0, 1])
                    # print('circ depth =', circ.depth())
                    # circ_transpiled = self.q_instance.transpile(circ)
                    # print('circ transpiled depth =', circ_transpiled[0].depth())
                    
                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    for key in list(counts):
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
                    # counts = dict(sorted(counts.items(), 
                    #               key=lambda item: item[1], 
                    #               reverse=True))
                    if self.states == 'h':
                        self.B_h[m, m] = [probs['00'], 0., probs['10'], 0.]
                        print("self.B_h =", self.B_h[m, m])
                    
                    if self.states == 'e':
                        self.B_e[m, m] = [probs['01'], 0., probs['11'], 0.]
                        print("self.B_e =", self.B_e[m, m])
                    print('')

        print("Calculations of diagonal transition amplitudes finished.")


    def compute_off_diagonal_amplitudes(self, cache_read: bool = True, cache_write: bool = True):
        """Calculates off-diagonal transition amplitudes in the Green's function.

        Args:
           cache_read: Whether to attempt to read recompiled circuits from cache files.
           cache_write: Whether to save recompiled circuits to cache files.
        """
        
        for m in range(self.n_orb):
           for n in range(self.n_orb):
                if m != n and (self.states is None or self.states in self.states_arr[m, n]):
                    print('(m, n) =', (m, n))
                    if self.q_instance.backend.name() == 'statevector_simulator':
                        print("Statevector simulator mode.")
                        circ = build_off_diagonal_circuits(self.ansatz.copy(), m, n, measure=False)
                        result = self.q_instance.execute(circ)
                        psi = reverse_qubit_order(result.get_statevector())

                        if self.states in ('h', None):
                            inds_hp = get_number_state_indices(
                                self.n_orb, self.n_occ - 1, anc='00')
                            inds_hm = get_number_state_indices(
                                self.n_orb, self.n_occ - 1, anc='01')
                            probs_hp = abs(self.eigenstates_h.conj().T @ psi[inds_hp]) ** 2
                            probs_hm = abs(self.eigenstates_h.conj().T @ psi[inds_hm]) ** 2
                            # print("probs_hp =", probs_hp)
                            # print("probs_hm =", probs_hm)
                            self.D_hp[m, n] = probs_hp
                            self.D_hm[m, n] = probs_hm

                        if self.states in ('e', None):
                            inds_ep = get_number_state_indices(
                                self.n_orb, self.n_occ + 1, anc='10')
                            inds_em = get_number_state_indices(
                                self.n_orb, self.n_occ + 1, anc='11')
                            probs_ep = abs(self.eigenstates_e.conj().T @ psi[inds_ep]) ** 2
                            probs_em = abs(self.eigenstates_e.conj().T @ psi[inds_em]) ** 2
                            # print("probs_ep =", probs_ep)
                            # print("probs_em =", probs_em)
                            self.D_ep[m, n] = probs_ep
                            self.D_em[m, n] = probs_em

                        if self.states is None:
                            states_elem = []
                            if sum(probs_hp) > 1e-3 or sum(probs_hm) > 1e-3:
                                states_elem.append('h')
                            if sum(probs_ep) > 1e-3 or sum(probs_em) > 1e-3:
                                states_elem.append('e')
                            self.states_arr[m, n] = states_elem

                        if (m, n) == (0, 1):
                            fig = circ.draw(output='mpl')
                            fig.savefig('circ_off_diag_statevector.png')
                            circ_statevector = get_statevector(circ)
                            np.save('circ_off_diag_statevector.npy', circ_statevector)

                    else:
                        print("QASM simulator mode.") # XXX: Not necessarily QASM simulator
                        circ = build_off_diagonal_circuits_with_qpe(self.ansatz.copy(), m, n, measure=False)
                        Umat = expm(1j * self.hamiltonian_arr * HARTREE_TO_EV)
                        if self.recompiled:
                            cUmat = np.kron(np.eye(4),
                                np.kron(np.diag([1, 0]), np.eye(2 ** 4))
                                + np.kron(np.diag([0, 1]), Umat))

                            circ.barrier()
                            circ.h(2)
                            statevector = reverse_qubit_order(get_statevector(circ))
                            quimb_gates = recompile_with_statevector(
                                statevector, cUmat, n_gate_rounds=7,
                                cache_options = {'read': cache_read, 'write': cache_write,
                                                 'hamiltonian': self.hamiltonian,
                                                 'type': f'greens-offdiag-({m}, {n})'})
                            circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
                            circ.h(2)
                            circ.barrier()

                            fig = circ.draw(output='mpl')
                            fig.savefig('circ_off_diag_recompiled.png')
                            # circ_recompiled = get_statevector(circ)
                            # np.save('circ_off_diag_recompiled.npy', circ_recompiled)
                                        
                        else:
                            cUgate = UnitaryGate(Umat).control(1)

                            circ.barrier()
                            circ.h(2)
                            circ.append(cUgate, [2, 6, 5, 4, 3])
                            circ.h(2)
                            circ.barrier()

                            if (m, n) == (0, 1):
                                fig = circ.draw(output='mpl')
                                fig.savefig('circ_off_diag_unrecompiled.png')
                                circ_unrecompiled = get_statevector(circ)
                                np.save('circ_off_diag_unrecompiled.npy', circ_unrecompiled)

                        circ.measure([0, 1, 2], [0, 1, 2])
                        # circ.measure(2, 2)
                        # print('circ depth =', circ.depth())
                        # circ_transpiled = self.q_instance.transpile(circ)
                        # print('circ transpiled depth =', circ_transpiled[0].depth())

                        result = self.q_instance.execute(circ)
                        counts = result.get_counts()
                        print(counts)
                        for key in list(counts):
                            if int(key[-1]) == (self.states == 'h'):
                                del counts[key]
                        print(counts)
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
                            print("self.D_hp =", self.D_hp[m, n])
                            print("self.D_hm =", self.D_hm[m, n])
                        
                        if self.states == 'e':
                            self.D_ep[m, n] = [probs['001'], 0., probs['101'], 0.]
                            self.D_em[m, n] = [probs['011'], 0., probs['111'], 0.]
                            print("self.D_ep =", self.D_ep[m, n])
                            print("self.D_em =", self.D_em[m, n])
                        print('')

        print("Calculations of off-diagonal transition amplitudes finished.")


    def compute_greens_function(self, z):
        """Calculates the value of the Green's function at z = omega + 1j * delta."""
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    self.B_e[m, n] = (
                        np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n]) +
                        np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m]))
                    self.B_h[m, n] = (
                        np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n]) + \
                        np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m]))
        
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                self.G_e[m, n] = np.sum(
                    self.B_e[m, n] / (z + self.energy_gs - self.energies_e))
                self.G_h[m, n] = np.sum(
                    self.B_h[m, n] / (z - self.energy_gs + self.energies_h))
        self.G = self.G_e + self.G_h
        print("Calculations of Green's functions finished.")

    def compute_spectral_function(self, z):
        """Calculates the spectral function at z = omega + 1j * delta."""
        if np.sum(self.G) == 0:
            self.compute_greens_function(z)
        A = - 1 / np.pi * np.imag(np.trace(self.G))
        print("Calculations of spectral function finished.")
        return A

    def compute_self_energy(self, z):
        """Calculates the self-energy."""
        #if np.sum(self.G) == 0:
        self.compute_greens_function(z)

        G_HF = np.zeros((self.n_orb, self.n_orb), dtype=complex)
        for i in range(self.n_orb):
            G_HF[i, i] = 1 / (z - self.molecule.orbital_energies[i // 2] * HARTREE_TO_EV)

        #print(np.diag(np.linalg.pinv(G_HF))[::2])
        #print(np.diag(np.linalg.pinv(self.G))[::2])

        Sigma = np.linalg.pinv(G_HF) - np.linalg.pinv(self.G)
        return Sigma

    @staticmethod
    def get_hamiltonian_shift_parameters(hamiltonian, states='e'):
        """Obtains the scaling factor and constant shift of the Hamiltonian 
        for phase estimation."""
        # Obtains the scaling factor.
        greens_function = GreensFunction(None, hamiltonian)
        greens_function.compute_eh_states()
        if states == 'h':
            E_low = greens_function.energies_h[0]
            E_high = greens_function.energies_h[2]
        elif states == 'e':
            E_low = greens_function.energies_e[0]
            E_high = greens_function.energies_e[2]

        scaling_factor = np.pi / (E_high - E_low)

        # Obtains the constant shift.
        greens_function = GreensFunction(
            None, hamiltonian, scaling_factor=scaling_factor)
        greens_function.compute_eh_states()
        if states == 'h':
            constant_shift = -greens_function.energies_h[0]
        elif states == 'e':
            constant_shift = -greens_function.energies_e[0]

        return scaling_factor, constant_shift
