import numpy as np

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp

import copy

class MolecularHamiltonian:
    def __init__(self, geometry, basis, multiplicity=1, charge=0, 
                run_pyscf_options={}, occupied_inds=None, active_inds=None):
        """Initializes a MolecularHamiltonian object."""
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.run_pyscf_options = run_pyscf_options
        molecule = MolecularData(
            self.geometry, self.basis, 
            self.multiplicity, self.charge)
        run_pyscf(molecule)
        self.molecule = molecule

        self.occupied_inds = [] if occupied_inds is None else occupied_inds
        self.active_inds = range(self.molecule.n_orbitals) if active_inds is None else active_inds

        self.openfermion_op= None
        self.qiskit_op = None
        self.build()
        
    def _build_openfermion_operator(self):
        """Constructs the Openfermion qubit operator."""
        # if self.occupied_inds is None and self.active_inds is None:
        #     self.active_inds = range(self.molecule.n_orbitals)
        hamiltonian = self.molecule.get_molecular_hamiltonian(
            occupied_indices=self.occupied_inds, active_indices=self.active_inds)
        fermion_op = get_fermion_operator(hamiltonian)
        qubit_op = jordan_wigner(fermion_op)
        qubit_op.compress()
        self.openfermion_op= qubit_op

    def _build_qiskit_operator(self):
        """Constructs the Qiskit qubit operator."""
        if self.openfermion_op is None:
            self._build_openfermion_operator()
        table = []
        coeffs = []
        n_qubits = 0
        for key in self.openfermion_op.terms:
            if key == ():
                continue
            num = max([t[0] for t in key])
            if num > n_qubits:
                n_qubits = num
        n_qubits += 1

        for key, val in self.openfermion_op.terms.items():
            coeffs.append(val)
            label = ['I'] * n_qubits
            for i, s in key:
                label[i] = s
            label = ''.join(label)
            pauli = Pauli(label[::-1]) # because Qiskit qubit order is reversed
            mask = list(pauli.x) + list(pauli.z)
            table.append(mask)
        primitive = SparsePauliOp(table, coeffs)
        qiskit_op = PauliSumOp(primitive)
        self.qiskit_op = qiskit_op

    def build(self):
        """Constructs both the Openfermion and the Qiskit qubit operators."""
        self._build_openfermion_operator()
        self._build_qiskit_operator()

    def to_openfermion_operator(self):
        """Returns the Openfermion qubit operator."""
        if self.openfermion_op is None:
            self._build_openfermion_operator()
        return self.openfermion_op

    def to_qiskit_operator(self):
        """Returns the Qiskit qubit operator."""
        if self.qiskit_op is None:
            self._build_qiskit_operator()
        return self.qiskit_op

    def to_array(self, array_type='ndarray'):
        """Returns the array form of the Hamiltonian."""
        assert array_type in ['sparse', 'matrix', 'array', 'ndarray']
        if self.openfermion_op is None:
            self._build_openfermion_operator()
        array = get_sparse_operator(self.openfermion_op)
        if array_type == 'sparse':
            return array
        elif array_type == 'matrix':
            return array.todense()
        elif array_type == 'array' or array_type == 'ndarray':
            return array.toarray()

    def copy(self):
        return copy.copy(self)
