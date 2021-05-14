import numpy as np

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp

class MolecularHamiltonian:
    def __init__(self, geometry, basis, multiplicity=1, charge=0, 
                run_pyscf_options={}, occupied_inds=None, active_inds=None):
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.run_pyscf_options = run_pyscf_options
        self.occupied_inds = occupied_inds
        self.active_inds = active_inds
        print('occupied_inds =', self.occupied_inds)
        print('active_inds =', self.active_inds)

        self.molecule = None
        self.openfermion_qubit_op = None
        self.qiskit_qubit_op = None

        molecule = MolecularData(self.geometry, self.basis, self.multiplicity, self.charge)
        run_pyscf(molecule)
        self.molecule = molecule

        self.build()
        
    def _build_openfermion_qubit_operator(self):
        if self.occupied_inds is None and self.active_inds is None:
            self.active_inds = range(self.molecule.n_orbitals)
        hamiltonian = self.molecule.get_molecular_hamiltonian(
            occupied_indices=self.occupied_inds, active_indices=self.active_inds)
        fermion_op = get_fermion_operator(hamiltonian)
        qubit_op = jordan_wigner(fermion_op)
        qubit_op.compress()
        self.openfermion_qubit_op = qubit_op

    def _build_qiskit_qubit_operator(self):
        if self.openfermion_qubit_op is None:
            self._build_openfermion_qubit_operator()
        table = []
        coeffs = []
        n_qubits = 0
        for key in self.openfermion_qubit_op.terms:
            if key == ():
                continue
            num = max([t[0] for t in key])
            if num > n_qubits:
                n_qubits = num
        n_qubits += 1

        for key, val in self.openfermion_qubit_op.terms.items():
            coeffs.append(val)
            label = ['I'] * n_qubits
            for i, s in key:
                label[i] = s
            label = ''.join(label)
            pauli = Pauli(label[::-1]) # because Qiskit qubit order is reversed
            mask = list(pauli.x) + list(pauli.z)
            table.append(mask)
        primitive = SparsePauliOp(table, coeffs)
        qiskit_qubit_op = PauliSumOp(primitive)
        self.qiskit_qubit_op = qiskit_qubit_op

    def build(self):
        self._build_openfermion_qubit_operator()
        self._build_qiskit_qubit_operator()

    def to_openfermion_qubit_operator(self):
        if self.openfermion_qubit_op is None:
            self._build_openfermion_qubit_operator()
        return self.openfermion_qubit_op

    def to_qiskit_qubit_operator(self):
        if self.qiskit_qubit_op is None:
            self._build_qiskit_qubit_operator()
        return self.qiskit_qubit_op

    def to_array(self, array_type='ndarray'):
        assert array_type in ['sparse', 'matrix', 'array', 'ndarray']
        if self.openfermion_qubit_op is None:
            self._build_openfermion_qubit_operator()
        array = get_sparse_operator(self.openfermion_qubit_op)
        if array_type == 'sparse':
            return array
        elif array_type == 'matrix':
            return array.todense()
        elif array_type == 'array' or array_type == 'ndarray':
            return array.toarray()
