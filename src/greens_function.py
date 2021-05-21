import numpy as np
from scipy.special import binom
from qiskit import *
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.aqua import QuantumInstance
from qiskit.utils import QuantumInstance
from constants import *
from tools import get_number_state_indices, number_state_eigensolver, reverse_qubit_order
from circuits import build_diagonal_circuits, build_off_diagonal_circuits

class GreensFunction:
    def __init__(self, ansatz, hamiltonian, optimizer=None, q_instance=None):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer if optimizer is not None else L_BFGS_B()
        self.backend = Aer.get_backend('statevector_simulator')
        self.q_instance = QuantumInstance(backend=Aer.get_backend('statevector_simulator'))

        #self.n_orb = self.hamiltonian.molecule.n_qubits
        self.n_orb = 2 * len(self.hamiltonian.active_inds)
        self.n_occ = 2 # XXX: Why is molecule.n_electrons 0 in an example I tried?
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
        vqe = VQE(self.ansatz, optimizer=self.optimizer, quantum_instance=self.backend)
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian.to_qiskit_qubit_operator())
        
        self.energy_gs = result.optimal_value * HARTREE_TO_EV
        self.ansatz.assign_parameters(result.optimal_parameters, inplace=True)
        print(f'Ground state energy = {self.energy_gs:.3f} eV')
    
    # TODO: Should use quantum subspace expansion in the general case
    def compute_eh_states(self):
        self.energies_e, self.eigenstates_e = \
            number_state_eigensolver(self.hamiltonian, self.n_occ + 1)
        self.energies_h, self.eigenstates_h = \
            number_state_eigensolver(self.hamiltonian, self.n_occ - 1)
        self.energies_e *= HARTREE_TO_EV
        self.energies_h *= HARTREE_TO_EV
        #print(self.n_e, len(self.energies_e))
        #print(self.n_h, len(self.energies_h))

    def compute_diagonal_amplitudes(self):
        for m in range(self.n_orb):
            circ = build_diagonal_circuits(self.ansatz.copy(), m)
            print('len(circ) =', len(circ))
            result = self.q_instance.execute(circ)
            psi = reverse_qubit_order(result.get_statevector()) # XXX: Only works for statevector_simulator
            # TODO: Need to truncate the first index of psi
            inds_e = get_number_state_indices(self.n_orb, self.n_occ + 1, anc='1')
            inds_h = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='0')
            inds_h_bin = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='0', return_type='binary')
            #for i in range(10):
            #    print(inds_h_bin[i])
            #print('')
            inds_all = inds_e + inds_h
            #for i in range(psi.shape[0]):
            #    if abs(psi[i]) > 1e-8:
            #        b = bin(i)[2:]
            #        print('0' * (13 - len(b)) + b, abs(psi[i]))
            # print(np.linalg.norm(psi))
            # print(np.linalg.norm(psi[inds_all]))
            #inds_e1 = get_number_state_indices(self.n_orb, self.n_occ + 1, anc='0')
            #inds_h1 = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='1')
            #print(self.eigenstates_e.shape, psi[inds_e].shape)
            #print(self.eigenstates_h.shape, psi[inds_h].shape)
            #psi_abs2 = np.abs(psi) ** 2
            #for i in range(psi_abs2.shape[0]):
            #    if psi_abs2[i] > 1e-8:
            #        print(i, bin(i)[2:])
            #print('norm(psi) = {:.3f}'.format(np.linalg.norm(psi)))
            #print('norm(psi_e) = {:.3f}'.format(np.linalg.norm(psi[inds_e])))
            #print('norm(psi_h) = {:.3f}'.format(np.linalg.norm(psi[inds_h])))
            #print('norm(psi_e1) =', np.linalg.norm(psi[inds_e1]))
            #print('norm(psi_h1) =', np.linalg.norm(psi[inds_h1]))
            probs_e = np.abs(self.eigenstates_e.conj().T @ psi[inds_e]) ** 2
            probs_h = np.abs(self.eigenstates_h.conj().T @ psi[inds_h]) ** 2
            # print('probs_e =', np.sum(probs_e))
            # rint('probs_h =', np.sum(probs_h))
            #print('Total p_e = {:.3f}'.format(np.sum(probs_e)))
            #print('Total p_h = {:.3f}'.format(np.sum(probs_h)))
            #print('')
            self.B_e[m, m] = probs_e
            self.B_h[m, m] = probs_h

    def compute_off_diagonal_amplitudes(self):
        D_hp = np.zeros((self.n_orb, self.n_orb, self.n_h))
        D_hm = np.zeros((self.n_orb, self.n_orb, self.n_h))
        D_ep = np.zeros((self.n_orb, self.n_orb, self.n_e))
        D_em = np.zeros((self.n_orb, self.n_orb, self.n_e))

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    circ = build_off_diagonal_circuits(self.ansatz.copy(), m, n)
                    print('len(circ) =', len(circ))
                    result = self.q_instance.execute(circ)
                    psi = reverse_qubit_order(result.get_statevector()) # XXX: Only works for statevector_simulator
                    inds_hp = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='00')
                    inds_hm = get_number_state_indices(self.n_orb, self.n_occ - 1, anc='01')
                    inds_ep = get_number_state_indices(self.n_orb, self.n_occ + 1, anc='10')
                    inds_em = get_number_state_indices(self.n_orb, self.n_occ + 1, anc='11')
                    # TODO: Need to truncate the first two indices of psi
                    probs_hp = abs(self.eigenstates_h.conj().T @ psi[inds_hp]) ** 2
                    probs_hm = abs(self.eigenstates_h.conj().T @ psi[inds_hm]) ** 2
                    probs_ep = abs(self.eigenstates_e.conj().T @ psi[inds_ep]) ** 2
                    probs_em = abs(self.eigenstates_e.conj().T @ psi[inds_em]) ** 2

                    #print('Total p_hp = {:.3f}'.format(np.sum(probs_hp)))
                    #print('Total p_hm = {:.3f}'.format(np.sum(probs_hm)))
                    #print('Total p_ep = {:.3f}'.format(np.sum(probs_ep)))
                    #print('Total p_em = {:.3f}'.format(np.sum(probs_em)))
                    #print('')                
                    D_hp[m, n] = probs_hp
                    D_hm[m, n] = probs_hm
                    D_ep[m, n] = probs_ep
                    D_em[m, n] = probs_em

        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    self.B_e[m, n] = np.exp(-1j * np.pi / 4) * (D_ep[m, n] - D_em[m, n]) + \
                                    np.exp(1j * np.pi / 4) * (D_ep[n, m] - D_em[n, m])
                    self.B_h[m, n] = np.exp(-1j * np.pi / 4) * (D_hp[m, n] - D_hm[m, n]) + \
                                    np.exp(1j * np.pi / 4) * (D_hp[n, m] - D_hm[n, m])
    
    def compute_greens_function(self, z):
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                self.G_e[m, n] = np.sum(self.B_e[m, n] / (z + self.energy_gs - self.energies_e))
                self.G_h[m, n] = np.sum(self.B_h[m, n] / (z - self.energy_gs + self.energies_h))
        self.G = self.G_e + self.G_h

    def compute_spectral_function(self, z):
        self.compute_greens_function(z)
        A = - 1 / np.pi * np.imag(np.trace(self.G))
        return A





    


