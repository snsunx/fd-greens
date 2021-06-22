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

from constants import *
from hamiltonians import MolecularHamiltonian
from tools import (get_number_state_indices, get_unitary, 
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

        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        self.openfermion_op = copy.deepcopy(self.hamiltonian.openfermion_op)
        self.qiskit_op *= scaling_factor
        self.qiskit_op.primitive.coeffs[0] += constant_shift / HARTREE_TO_EV
        self.openfermion_op *= scaling_factor
        self.openfermion_op.terms[()] += constant_shift / HARTREE_TO_EV
        self.hamiltonian_arr = get_sparse_operator(self.openfermion_op).toarray()

        self.recompiled = recompiled
        
        
        self.optimizer = optimizer if optimizer is not None else L_BFGS_B()
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


    def compute_ground_state(self):
        """Calculates the ground state of the Hamiltonian using VQE."""
        # TODO: Need to save the coefficient somewhere instead of passing in 
        # statevector simulator all the time.
        vqe = VQE(self.ansatz, optimizer=self.optimizer, 
                  quantum_instance=Aer.get_backend('statevector_simulator'))
        result = vqe.compute_minimum_eigenvalue(self.qiskit_op)
        
        self.energy_gs = result.optimal_value * HARTREE_TO_EV
        self.ansatz.assign_parameters(result.optimal_parameters, inplace=True)
        print(f'Ground state energy = {self.energy_gs:.3f} eV')
    
    # TODO: Should use quantum subspace expansion in the general case
    def compute_eh_states(self):
        """Calculates the electron-added or hole-added states of the Hamiltonian."""
        self.energies_e, self.eigenstates_e = \
            number_state_eigensolver(self.hamiltonian_arr, self.n_occ + 1)
        self.energies_h, self.eigenstates_h = \
            number_state_eigensolver(self.hamiltonian_arr, self.n_occ - 1)
        self.energies_e *= HARTREE_TO_EV
        self.energies_h *= HARTREE_TO_EV
        print("Calculations of quasiparticle states finished.")

    def compute_diagonal_amplitudes(self, states='h'):
        """Calculates diagonal transition amplitudes of the Green's function."""
        if states == 'h':
            orbs = range(self.n_occ)
        else:
            orbs = range(self.n_occ, self.n_orb)
        
        for m in orbs:
            if self.q_instance.backend.name() == 'statevector_simulator':
                print("Statevector simulator mode")
                circ = build_diagonal_circuits(self.ansatz.copy(), m)
                result = self.q_instance.execute(circ)

                psi = reverse_qubit_order(result.get_statevector())
                
                inds_e = get_number_state_indices(self.n_orb, self.n_occ + 1, anc='1')
                inds_h = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='0')
                probs_e = np.abs(self.eigenstates_e.conj().T @ psi[inds_e]) ** 2
                probs_h = np.abs(self.eigenstates_h.conj().T @ psi[inds_h]) ** 2
                probs_e[abs(probs_e) < 1e-8] = 0.
                probs_h[abs(probs_h) < 1e-8] = 0.
                print('probs_e =', probs_e)
                print('probs_h =', probs_h)
            else:
                print("QASM Simulator Mode.") # XXX: Not necessarily QASM simulator
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
                        statevector, cUmat, n_gate_rounds=6)
                    circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
                    circ.h(1)
                    circ.barrier()

                    circ_recompiled = get_statevector(circ)
                    np.save('circ_recompiled.npy', circ_recompiled)
                else:
                    cUgate = UnitaryGate(Umat).control(1)

                    circ.barrier()
                    circ.h(1)
                    circ.append(cUgate, [1, 5, 4, 3, 2])
                    circ.h(1)
                    circ.barrier()

                    circ_unrecompiled = get_statevector(circ)
                    np.save('circ_unrecompiled.npy', circ_unrecompiled)

                circ.measure([0, 1], [0, 1])
                # print(circ)
                print('circ depth =', circ.depth())

                circ_transpiled = self.q_instance.transpile(circ)
                print('circ transpiled depth =', circ_transpiled[0].depth())

                
                result = self.q_instance.execute(circ)
                counts = result.get_counts()
                counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
                shots = sum(counts.values())
                for key, val in counts.items():
                    print(key, val / shots)
                print('')

            # self.B_e[m, m] = probs_e
            # self.B_h[m, m] = probs_h
        print("Calculations of diagonal transition amplitudes finished.")

    def compute_off_diagonal_amplitudes(self):
        """Calculates off-diagonal transition amplitudes of the Green's function."""
        D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h))
        D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h))
        D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e))
        D_em = np.zeros((self.n_orb, self.n_orb, self.n_e))

        D_obj = np.empty((self.n_orb, self.n_orb), dtype=np.object)

        for m in range(self.n_orb):
           for n in range(self.n_orb):
        # for m in [0]:
        #     for n in [1]:
                if m != n:
                    if self.q_instance.backend.name() == 'statevector_simulator':
                        print("Statevector simulator mode.")
                        circ = build_off_diagonal_circuits(self.ansatz.copy(), m, n)
                        result = self.q_instance.execute(circ)
                        psi = reverse_qubit_order(result.get_statevector())

                        inds_hp = get_number_state_indices(
                            self.n_orb, self.n_occ - 1, anc='00')
                        inds_hm = get_number_state_indices(
                            self.n_orb, self.n_occ - 1, anc='01')
                        inds_ep = get_number_state_indices(
                            self.n_orb, self.n_occ + 1, anc='10')
                        inds_em = get_number_state_indices(
                            self.n_orb, self.n_occ + 1, anc='11')
                        probs_hp = abs(self.eigenstates_h.conj().T @ psi[inds_hp]) ** 2
                        probs_hm = abs(self.eigenstates_h.conj().T @ psi[inds_hm]) ** 2
                        probs_ep = abs(self.eigenstates_e.conj().T @ psi[inds_ep]) ** 2
                        probs_em = abs(self.eigenstates_e.conj().T @ psi[inds_em]) ** 2
                        if sum(probs_hp) > 1e-8 or sum(probs_hm) > 1e-8:
                            D_obj[m, n] = 'h'
                            assert(sum(probs_ep) + sum(probs_em) < 1e-8)
                        elif sum(probs_ep) > 1e-8 or sum(probs_em) > 1e-8:
                            D_obj[m, n] = 'e'
                            assert(sum(probs_hp) + sum(probs_hm) < 1e-8)

                        '''
                        print("probs_hp =", probs_hp)
                        print("probs_hm =", probs_hm)
                        print("probs_ep =", probs_ep)
                        print("probs_em =", probs_em)

                        fig = circ.draw(output='mpl')
                        fig.savefig('circ_off_diag_statevector.png')
                        circ_statevector = get_statevector(circ)
                        np.save('circ_off_diag_statevector.npy', circ_statevector)
                        '''
                    else:
                        print("QASM simulator mode.")
                        circ = build_off_diagonal_circuits_with_qpe(self.ansatz.copy(), m, n)
                        Umat = expm(1j * self.hamiltonian_arr * HARTREE_TO_EV)
                        if self.recompiled:
                            cUmat = np.kron(np.eye(4),
                                np.kron(np.diag([1, 0]), np.eye(2 ** 4))
                                + np.kron(np.diag([0, 1]), Umat))

                            
                            circ.barrier()
                            circ.h(2)
                            statevector = reverse_qubit_order(get_statevector(circ))
                            quimb_gates = recompile_with_statevector(
                                statevector, cUmat, n_gate_rounds=7)
                            circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
                            circ.h(2)
                            circ.barrier()
                            

                            fig = circ.draw(output='mpl')
                            fig.savefig('circ_off_diag_recompiled.png')
                            circ_recompiled = get_statevector(circ)
                            np.save('circ_off_diag_recompiled.npy', circ_recompiled)
                                        
                        else: # TODO: Should apply the whole unitary here
                            cUgate = UnitaryGate(Umat).control(1)

                            circ.barrier()
                            circ.h(2)
                            circ.append(cUgate, [2, 6, 5, 4, 3])
                            circ.h(2)
                            circ.barrier()

                            circ_unrecompiled = get_statevector(circ)
                            np.save('circ_off_diag_unrecompiled.npy', circ_unrecompiled)

                        circ.measure([0, 1, 2], [0, 1, 2])
                        print('circ depth =', circ.depth())
                        print(circ)

                        circ_transpiled = self.q_instance.transpile(circ)
                        print('circ transpiled depth =', circ_transpiled[0].depth())
                    '''
                    result = self.q_instance.execute(circ)
                    counts = result.get_counts()
                    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
                    shots = sum(counts.values())
                    for key, val in counts.items():
                        print(key, val / shots)
                    print('')
                    '''

                    
        print(D_obj)
        """
                    D_hp[m, n] = probs_hp
                    D_hm[m, n] = probs_hm
                    D_ep[m, n] = probs_ep
                    D_em[m, n] = probs_em

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    self.B_e[m, n] = (
                        np.exp(-1j * np.pi / 4) * (D_ep[m, n] - D_em[m, n]) +
                        np.exp(1j * np.pi / 4) * (D_ep[n, m] - D_em[n, m]))
                    self.B_h[m, n] = (
                        np.exp(-1j * np.pi / 4) * (D_hp[m, n] - D_hm[m, n]) + \
                        np.exp(1j * np.pi / 4) * (D_hp[n, m] - D_hm[n, m]))
        """
        print("Calculations of off-diagonal transition amplitudes finished.")
    
    def compute_greens_function(self, z):
        """Calculates the value of the Green's function at z = omega + 1j * delta."""
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
        self.compute_greens_function(z)
        A = - 1 / np.pi * np.imag(np.trace(self.G))
        print("Calculations of spectral function finished.")
        return A
    
    @staticmethod
    def get_hamiltonian_shift_parameters(ansatz, hamiltonian, states='e'):
        """Obtains the scaling factor and constant shift for phase estimation."""
        greens_function = GreensFunction(ansatz.copy(), hamiltonian)
        greens_function.compute_ground_state()
        ansatz = greens_function.ansatz

        # Obtains the scaling factor.
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
            ansatz.copy(), hamiltonian, scaling_factor=scaling_factor)
        greens_function.compute_eh_states()
        if states == 'h':
            constant_shift = -greens_function.energies_h[0]
        elif states == 'e':
            constant_shift = -greens_function.energies_e[0]

        return scaling_factor, constant_shift
