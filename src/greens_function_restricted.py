from typing import Union, Tuple, Optional

import numpy as np
from scipy.linalg import expm
from scipy.special import binom

from qiskit import QuantumCircuit, Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B
from qiskit.utils import QuantumInstance
from qiskit.extensions import UnitaryGate

from constants import HARTREE_TO_EV
from hamiltonians import MolecularHamiltonian
from number_state_solvers import (get_number_state_indices,
                                  number_state_eigensolver)
from tools import (get_indices_with_ancilla, 
                   get_pauli_tuple_dictionary,
                   reverse_qubit_order,
                   get_statevector, 
                   load_vqe_result,
                   save_vqe_result)
from circuits import build_diagonal_circuits, build_off_diagonal_circuits
from recompilation import recompile_with_statevector, apply_quimb_gates
from z2_symmetries import transform_4q_hamiltonian

np.set_printoptions(precision=6)
pauli_tuple_dict = get_pauli_tuple_dictionary()
print(pauli_tuple_dict)

class GreensFunctionRestricted:
    def __init__(self, 
                 ansatz: QuantumCircuit, 
                 hamiltonian: MolecularHamiltonian, 
                 optimizer: Optional[Optimizer] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 scaling: float = 1.,
                 shift: float = 0.,
                 recompiled: bool = True,
                 states: Optional[str] = None) -> None:
        """Creates an object to calculate frequency-domain Green's function.
        
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
        self.e_orb = np.diag(e_orb) * HARTREE_TO_EV

        # Extract the Hamiltonian operators and apply shift and scaling factors
        self.qiskit_op = self.hamiltonian.qiskit_op.copy()
        self.qiskit_op *= scaling
        self.qiskit_op.primitive.coeffs[0] += shift / HARTREE_TO_EV
        
        self.qiskit_op_transformed = transform_4q_hamiltonian(self.qiskit_op)
        self.hamiltonian_arr = self.qiskit_op_transformed.to_matrix()
        
        self.qiskit_op_up = transform_4q_hamiltonian(self.qiskit_op, spin='up')
        self.hamiltonian_arr_up = self.qiskit_op_up.to_matrix()
        
        self.optimizer = L_BFGS_B() if optimizer is None else optimizer
        if q_instance is None:
            self.q_instance = QuantumInstance(
                Aer.get_backend('statevector_simulator'))
        else:
            self.q_instance = q_instance
        self.recompiled = recompiled

        # Number of orbitals and indices
        self.n_orb = len(self.hamiltonian.active_inds)
        print('self.n_orb =', self.n_orb)
        self.n_occ = (self.hamiltonian.molecule.n_electrons 
                      - 2 * len(self.hamiltonian.occupied_inds))
        self.n_vir = self.n_orb - self.n_occ
        self.inds_occ = list(range(self.n_occ))
        self.inds_vir = list(range(self.n_occ, self.n_orb))
        self.n_e = int(binom(self.n_orb, self.n_occ + 1))
        self.n_h = int(binom(self.n_orb, self.n_occ - 1))

        #self.inds_h = [int(x, 2) for x in ['10', '00']]
        #self.inds_e = [int(x, 2) for x in ['11', '01']]

        self.inds_h = ['10', '00']
        self.inds_e = ['11', '01']

        # XXX: Dimensions are hardcoded for now
        # Define the transition amplitude matrices
        self.B_e = np.zeros((2, 2, 2), dtype=complex)
        self.B_h = np.zeros((2, 2, 2), dtype=complex)
        self.D_hp = np.zeros((2, 2, 2), dtype=complex)
        self.D_hm = np.zeros((2, 2, 2), dtype=complex)
        self.D_ep = np.zeros((2, 2, 2), dtype=complex)
        self.D_em = np.zeros((2, 2, 2), dtype=complex)

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
        inds_e = [int(x, 2) for x in self.inds_e]
        inds_h = [int(x, 2) for x in self.inds_h]
        self.eigenenergies_e, self.eigenstates_e = \
            number_state_eigensolver(self.hamiltonian_arr_up, inds=inds_e)
        self.eigenenergies_h, self.eigenstates_h = \
            number_state_eigensolver(self.hamiltonian_arr_up, inds=inds_h)
        self.eigenenergies_e *= HARTREE_TO_EV
        self.eigenenergies_h *= HARTREE_TO_EV
        print('eigenenergies_e\n', self.eigenenergies_e)
        print('eigenenergies_h\n', self.eigenenergies_h)
        print("eigenstates_e\n", self.eigenstates_e)
        print("eigenstates_h\n", self.eigenstates_h)

    def compute_diagonal_amplitudes(self,
                                    cache_read: bool = True,
                                    cache_write: bool = True
                                    ) -> None:
        """Calculates diagonal transition amplitudes.
        
        Args:
            cache_read: Whether to read recompiled circuits from cache files.
            cache_write: Whether to save recompiled circuits to cache files.
        """
        print("Start calculating diagonal transition amplitudes")
        for m in range(self.n_orb):
            print('m =', m)
            tup_xy = pauli_tuple_dict[m]
            if self.states is None or self.states in self.states_arr[m, m]:
                if self.q_instance.backend.name() == 'statevector_simulator':
                    circ = build_diagonal_circuits(self.ansatz.copy(), tup_xy, with_qpe=False)
                    result = self.q_instance.execute(circ)
                    psi = result.get_statevector()
                    print('BINNNNNNNNNNNN =', bin(np.argmax(np.abs(psi)))[2:])
                    
                    if self.states in ('e', None):
                        inds_e = get_indices_with_ancilla(self.inds_e, anc='1')
                        print('inds_e =', inds_e)
                        probs_e = np.abs(self.eigenstates_e.conj().T @ psi[inds_e]) ** 2
                        probs_e[abs(probs_e) < 1e-8] = 0.
                        print('@@@@@ probs_e =', probs_e)
                        self.B_e[m, m] = probs_e

                    if self.states in ('h', None):
                        inds_h = get_indices_with_ancilla(self.inds_h, anc='0')
                        print('inds_h =', inds_h)
                        probs_h = np.abs(self.eigenstates_h.conj().T @ psi[inds_h]) ** 2
                        probs_h[abs(probs_h) < 1e-8] = 0.
                        print('@@@@@ probs_h =', probs_h)
                        self.B_h[m, m] = probs_h

                    if self.states is None:
                        states_elem = []
                        if sum(probs_h) > 1e-3:
                            states_elem.append('h')
                        if sum(probs_e) > 1e-3:
                            states_elem.append('e')
                        self.states_arr[m, m] = states_elem

                else: # QASM simulator or hardware
                    circ = build_diagonal_circuits(self.ansatz.copy(), )
                    Umat = expm(1j * self.hamiltonian_arr * HARTREE_TO_EV)
                    if self.recompiled:
                        # Construct the unitary |0><0|⊗I + |1><1|⊗e^{iHt} 
                        # on the last N-2 qubits
                        cUmat = np.kron(np.eye(2), 
                            np.kron(np.diag([1, 0]), np.eye(2 ** 4))
                            + np.kron(np.diag([0, 1]), Umat))

                        # Append single-qubit QPE circuit
                        circ.barrier()
                        circ.h(1)
                        statevector = reverse_qubit_order(get_statevector(circ))
                        quimb_gates = recompile_with_statevector(
                            statevector, cUmat, n_gate_rounds=3,
                            cache_options = {'read': cache_read, 'write': cache_write, 
                                             'hamiltonian': self.hamiltonian,
                                             'index': str(m), 'states': self.states})
                        circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
                        circ.h(1)
                        circ.barrier()

                        # circ_recompiled = get_statevector(circ)
                        # np.save('circ_recompiled.npy', circ_recompiled)
                    else:
                        # Construct the unitary |0><0|⊗I + |1><1|⊗e^{iHt} 
                        # on the last N-2 qubits
                        cUgate = UnitaryGate(Umat).control(1)

                        # Append single-qubit QPE circuit
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
                        # If calculating (N-1)-electron states, need the first 
                        # qubit to be in |0>, hence deleting the counts with
                        # the first qubit in |1>
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
            tup_xy_left = pauli_tuple_dict[m]
            for n in range(self.n_orb):
                tup_xy_right = pauli_tuple_dict[n]
                if m != n and (self.states is None or 
                               self.states in self.states_arr[m, n]): # XXX
                    if self.q_instance.backend.name() == 'statevector_simulator':
                        circ = build_off_diagonal_circuits(
                            self.ansatz.copy(), tup_xy_left, tup_xy_right, with_qpe=False)
                        result = self.q_instance.execute(circ)
                        psi = result.get_statevector()

                        if self.states in ('h', None):
                            inds_hp = get_indices_with_ancilla(self.inds_h, anc='00')
                            inds_hm = get_indices_with_ancilla(self.inds_h, anc='10')
                            probs_hp = abs(self.eigenstates_h.conj().T @ psi[inds_hp]) ** 2
                            probs_hm = abs(self.eigenstates_h.conj().T @ psi[inds_hm]) ** 2
                            self.D_hp[m, n] = probs_hp
                            self.D_hm[m, n] = probs_hm

                        if self.states in ('e', None):
                            inds_ep = get_indices_with_ancilla(self.inds_e, anc='01')
                            inds_em = get_indices_with_ancilla(self.inds_e, anc='11')
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
                        circ = build_off_diagonal_circuits(
                            self.ansatz.copy(), tup_xy_left, tup_xy_right)
                        Umat = expm(1j * self.hamiltonian_arr_up * HARTREE_TO_EV)

                        # Append QPE circuit
                        if self.recompiled:
                            cUmat = np.kron(np.eye(4),
                                np.kron(np.diag([1, 0]), np.eye(2 ** 2))
                                + np.kron(np.diag([0, 1]), Umat))
                            circ.barrier()
                            circ.h(2)
                            statevector = reverse_qubit_order(
                                get_statevector(circ))
                            quimb_gates = recompile_with_statevector(
                                statevector, cUmat, n_gate_rounds=3,
                                cache_options = {'read': cache_read, 
                                                 'write': cache_write,
                                                 'hamiltonian': self.hamiltonian,
                                                 'index': f'{m}-{n}', 
                                                 'states': self.states})
                            circ = apply_quimb_gates(
                                quimb_gates, circ.copy(), reverse=False)
                            circ.h(2)
                            circ.barrier()
                                        
                        else:
                            cUgate = UnitaryGate(Umat).control(1)

                            circ.barrier()
                            circ.h(2)
                            circ.append(cUgate, [2, 6, 5, 4, 3])
                            circ.h(2)
                            circ.barrier()

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
                            # print("self.D_hp =", self.D_hp[m, n])
                            # print("self.D_hm =", self.D_hm[m, n])
                        
                        if self.states == 'e':
                            self.D_ep[m, n] = [probs['001'], 0., probs['101'], 0.]
                            self.D_em[m, n] = [probs['011'], 0., probs['111'], 0.]
                            # print("self.D_ep =", self.D_ep[m, n])
                            # print("self.D_em =", self.D_em[m, n])
                        print('')

        # Unpack D values to B values
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m != n:
                    self.B_e[m, n] = (
                        np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n]) +
                        np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m]))
                    self.B_h[m, n] = (
                        np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n]) +
                        np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m]))

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
        """Main function to compute energies and transition amplitudes.
        
        Args:
            compute_energies: Whether to compute ground- and (N±1)-electron 
                state energies.
            save_params: Whether to save the VQE energy and ansatz parameters.
            load_params: Whether to load the VQE energy and ansatz parameters.
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
                      gf: 'GreensFunctionRestricted', 
                      states: str,
                      q_instance: Optional[QuantumInstance] = None
                      ) -> 'GreensFunctionRestricted':
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
            E_high = gf_scaling.eigenenergies_h[1]
        elif states == 'e':
            E_low = gf_scaling.eigenenergies_e[0]
            E_high = gf_scaling.eigenenergies_e[1]
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
                         gf_sv: 'GreensFunctionRestricted',
                         gf_e: 'GreensFunctionRestricted',
                         gf_h: 'GreensFunctionRestricted',
                         q_instance: Optional[QuantumInstance] = None
                         ) -> 'GreensFunctionRestricted':
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