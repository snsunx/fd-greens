"""New GreensFunction class"""

from typing import Optional
from functools import partial

import numpy as np
from scipy.special import binom

from qiskit import ClassicalRegister, transpile
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate

import params
from hamiltonians import MolecularHamiltonian
from number_state_solvers import measure_operator
from operators import SecondQuantizedOperators, ChargeOperators
from qubit_indices import QubitIndices, transform_4q_indices
from circuits import CircuitConstructor, CircuitData, transpile_across_barrier
from z2_symmetries import transform_4q_pauli
from utils import save_circuit, state_tomography, solve_energy_probabilities, get_overlap, get_counts
from helpers import get_quantum_instance
from vqe import GroundStateSolver
from number_state_solvers import ExcitedStatesSolver

np.set_printoptions(precision=6)

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
        """Initializes a AmplitudesSolver object.

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
        self.charge_dict_s = charge_ops.get_pauli_dict()
        #for key, val in self.charge_dict.items():
        #    print(key, val.coeffs[0], val.table)

        charge_ops = ChargeOperators(self.n_elec)
        charge_ops.transform(partial(transform_4q_pauli, init_state=[0, 0]))
        self.charge_dict_t = charge_ops.get_pauli_dict()

    def compute_charge_diagonal(self) -> None:
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
                for i in range(len(psi)):
                    if abs(psi[i]) > 1e-8:
                        print(format(i, '#06b')[2:], psi[i])
                print('inds_tot_s =', inds_tot_s)
                psi_s = psi[inds_tot_s.int_form]
                L_mm_s = np.abs(self.states_s.conj().T @ psi_s) ** 2
                print('np.sum(L_mm_s) =', np.sum(L_mm_s))

                result = self.q_instance.execute(circ_t)
                psi = result.get_statevector()
                for i in range(len(psi)):
                    if abs(psi[i]) > 1e-8:
                        print(format(i, '#06b')[2:], psi[i])
                print('inds_tot_t =', inds_tot_t)
                psi_t = psi[inds_tot_t.int_form]
                L_mm_t = np.abs(self.states_t.conj().T @ psi_t) ** 2
                print('np.sum(L_mm_t) =', np.sum(L_mm_t))

                L_mm = np.hstack((L_mm_s, L_mm_t))

            self.L[m, m] = L_mm

            print(f'L[{m}, {m}] = {self.L[m, m]}')
        print("------------------------------------------------------")

    def run(self, method=None) -> None:
        """Runs the functions to compute transition amplitudes."""
        if method is not None:
            self.method = method
        self.compute_charge_diagonal()

