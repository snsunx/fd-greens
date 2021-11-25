"""ResponseFunction class."""

from typing import Union, Tuple, Optional, Sequence, Mapping
from functools import partial

import numpy as np
from scipy.special import binom

from qiskit import QuantumCircuit, ClassicalRegister, Aer, transpile
# from qiskit.algorithms import VQE
# from qiskit.algorithms.optimizers import Optimizer, L_BFGS_B
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate
from qiskit.opflow import PauliSumOp

from constants import HARTREE_TO_EV
from hamiltonians import MolecularHamiltonian
from number_state_solvers import (number_state_eigensolver, 
                                  quantum_subspace_expansion,
                                  measure_operator)
from recompilation import CircuitRecompiler
from operators import SecondQuantizedOperators
from qubit_indices import QubitIndices
from circuits import CircuitConstructor, CircuitData, transpile_across_barrier
from vqe import vqe_minimize, get_ansatz, get_ansatz_e, get_ansatz_h
from z2_symmetries import transform_4q_hamiltonian
from utils import save_circuit, state_tomography, solve_energy_probabilities, get_overlap
from helpers import get_quantum_instance


class ResponseFunction:
    def __init__(self,
                 ansatz: QuantumCircuit,
                 hamiltonian: MolecularHamiltonian,
                 spin: str = 'up',
                 methods: Optional[dict] = None,
                 q_instances: Optional[Mapping[str, QuantumInstance]] = None,
                 add_barriers: bool = True,
                 ccx_data: Optional[CircuitData] = None,
                 recompiled: bool = True,
                 transpiled: bool = True,
                 push: bool = False) -> None:
        """Initializes a GreensFunctionRestricted object.

        Args:
            ansatz: The parametrized circuit as an ansatz to VQE.
            hamiltonian: The hamiltonian of the molecule.
            spin: A string indicating the spin in the tapered (N+/-1)-electron operators. 
                Either 'up' or 'down'.
            methods: The method for extracting the transition amplitudes. Either 'energy'
                or 'tomography'.
            q_instances: A dictionary of QuantumInstance objects to execute the ground state, 
                (N+/-1)-electron state and transition amplitude circuits.
            recompiled: Whether the QPE circuit is recompiled.
            add_barriers: Whether to add barriers to the circuit.
            ccx_data: A CircuitData object indicating how CCX gate is applied.
            transpiled: Whether the circuit is transpiled.
            push: Whether the SWAP gates are pushed.
        """
        assert spin in ['up', 'down']

        # Basic variables of the system
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.spin = spin

        # The methods for each of gs, eh, and amp stages of the algorithm
        self.methods = {'gs': 'exact', 'eh': 'exact', 'amp': 'exact'}
        if methods is not None:
            self.methods.update(methods)
        assert self.methods['gs'] in ['exact', 'vqe']
        assert self.methods['eh'] in ['exact', 'qse', 'ssvqe']
        assert self.methods['amp'] in ['exact', 'energy', 'tomo']

        # The quantum instances
        self.q_instances = {'gs': get_quantum_instance('sv'),
                            'eh': get_quantum_instance('sv'),
                            'amp': get_quantum_instance('sv')}
        if q_instances is not None:
            if isinstance(q_instances, QuantumInstance):
                self.q_instances = {'gs': q_instances,
                                    'eh': q_instances,
                                    'amp': q_instances}
            else:
                self.q_instances.update(q_instances)
        # if self.methods['amp'] == 'tomography' and 'tomography' not in self.q_instances.keys():
        self.q_instances['tomo'] = self.q_instances['amp']

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
        self.qiskit_op_gs = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 1]).reduce()
        if self.spin == 'up': # e up h down
            self.qiskit_op_spin = transform_4q_hamiltonian(self.qiskit_op, init_state=[0, 1]).reduce()
        elif self.spin == 'down': # e down h up
            self.qiskit_op_spin = transform_4q_hamiltonian(self.qiskit_op, init_state=[1, 0]).reduce()

        # print('qiskit_op_gs\n', self.qiskit_op_gs)
        # print('qiskit_op_spin\n', self.qiskit_op_spin)

        # Create Pauli dictionaries for the original operators
        second_q_ops = SecondQuantizedOperators(4)
        self.pauli_op_dict = second_q_ops.get_op_dict_all()

        # Create Pauli dictionaries for the transformed operators
        second_q_ops = SecondQuantizedOperators(4)
        second_q_ops.transform(partial(transform_4q_hamiltonian, init_state=[1, 1], tapered=False))
        self.pauli_op_dict_trans = second_q_ops.get_op_dict_all()
        #for key, (x_op, y_op) in self.pauli_op_dict_trans.items():
        #    print(key, x_op.table.to_labels()[0], y_op.table.to_labels()[0])

        # Create Pauli dictionaries for the transformed and tapered operators
        second_q_ops = SecondQuantizedOperators(4)
        second_q_ops.transform(partial(transform_4q_hamiltonian, init_state=[1, 1]))
        # self.pauli_op_dict_tapered = second_q_ops.get_op_dict(spin=spin)
        self.pauli_op_dict_tapered = second_q_ops.get_op_dict_all()

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

        # `swap` is whether an additional swap gate is applied on top of the two CNOTs.
        # This part is hardcoded.
        swap = True
        if not swap:
            if self.spin == 'up':
                self.inds_h = QubitIndices(['10', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '01'], n_qubits=2)
            elif self.spin == 'down':
                self.inds_h = QubitIndices(['01', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '10'], n_qubits=2)
        else:
            if self.spin == 'up':
                self.inds_h = QubitIndices(['01', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '10'], n_qubits=2)
            elif self.spin == 'down':
                self.inds_h = QubitIndices(['10', '00'], n_qubits=2)
                self.inds_e = QubitIndices(['11', '01'], n_qubits=2)

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

        # Variables created in VQE
        self.energy_gs = None
        self.circuit_constructor = None

        # Variables created in solving N+/-1 electron states
        self.eigenenergies_e = None
        self.eigenstates_e = None
        self.eigenenergies_h = None
        self.eigenstates_h = None
