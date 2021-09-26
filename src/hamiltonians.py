"""The """

from typing import Union, Sequence, Optional

import numpy as np

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp

from openfermion import MolecularData, QubitOperator
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator

class MolecularHamiltonian:
    """A class to hold a molecular Hamiltonian."""
    def __init__(self, 
                 geometry: Union[Sequence], 
                 basis: str, 
                 multiplicity: int = 1, 
                 charge: int = 0, 
                 name: Optional[str] = None,
                 run_pyscf_options: dict = {}, 
                 occ_inds: Optional[Sequence[int]] = None, 
                 act_inds: Optional[Sequence[int]] = None,
                 build_ops: bool = True):
        """Initializes a MolecularHamiltonian object.

        Args:
            geometry: A list of tuples giving the coordinates of each atom. 
                An example is [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]].
            basis: A string giving the basis set.
            multiplicity: An integer giving the spin multiplicity.
            charge: An integer giving the total molecular charge. Defaults to 0.
            name: The string identifer used for saving and loading 
                cached circuits.
            run_pyscf_options: A dictionary of keyword arguments passed to the
                run_pyscf function.
            occ_inds: A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
            act_inds: A list of spatial orbital indices indicating
                which orbitals should be considered active.
        """
        self.geometry = geometry
        self.basis = basis
        if name is None:
            self.name = ''.join(
                [geometry[i][0] for i in range(len(geometry))])
        else:
            self.name = name

        self.multiplicity = multiplicity
        self.charge = charge
        self.run_pyscf_options = run_pyscf_options
        molecule = MolecularData(
            self.geometry, self.basis, 
            multiplicity=self.multiplicity, charge=self.charge)
        run_pyscf(molecule)
        self.molecule = molecule

        if occ_inds is None:
            self.occ_inds = [] 
        else:
            self.occ_inds = occ_inds
        
        if act_inds is None:
            self.act_inds = range(self.molecule.n_orbitals)
        else:
            self.act_inds = act_inds

        self._openfermion_op = None
        self._qiskit_op = None
        if build_ops:
            self.build()

    def _build_openfermion_operator(self):
        """A private method for constructing the Openfermion qubit operator
        in QubitOperator form. Called by the `build` function."""
        # if self.occ_inds is None and self.act_inds is None:
        #     self.act_inds = range(self.molecule.n_orbitals)
        hamiltonian = self.molecule.get_molecular_hamiltonian(
            occupied_indices=self.occ_inds, 
            active_indices=self.act_inds)
        fermion_op = get_fermion_operator(hamiltonian)
        qubit_op = jordan_wigner(fermion_op)
        qubit_op.compress()
        self._openfermion_op = qubit_op

    def _build_qiskit_operator(self):
        """A private method for constructing the Qiskit qubit operator 
        in PauliSumOp form. Called by the `build` function."""
        if self._openfermion_op is None:
            self._build_openfermion_operator()
        table = []
        coeffs = []
        n_qubits = 0
        for key in self._openfermion_op.terms:
            if key == ():
                continue
            num = max([t[0] for t in key])
            if num > n_qubits:
                n_qubits = num
        n_qubits += 1

        for key, val in self._openfermion_op.terms.items():
            coeffs.append(val)
            label = ['I'] * n_qubits
            for i, s in key:
                label[i] = s
            label = ''.join(label)
            pauli = Pauli(label[::-1]) # because Qiskit qubit order is reversed
            mask = list(pauli.x) + list(pauli.z)
            table.append(mask)
        primitive = SparsePauliOp(table, coeffs)
        qubit_op = PauliSumOp(primitive)
        self._qiskit_op = qubit_op

    def build(self, 
              build_openfermion_op: bool = True, 
              build_qiskit_op: bool = True):
        """Constructs both the Openfermion and the Qiskit qubit operators.
        
        Args:
            build_openfermion_op: Whether to build the Openfermion qubit operator.
            build_qiskit_op: Whether to build the Qiskit qubit operator.
        """
        if build_openfermion_op:
            self._build_openfermion_operator()
        if build_qiskit_op:
            self._build_qiskit_operator()

    @property
    def openfermion_op(self) -> QubitOperator:
        """Returns the Openfermion qubit operator."""
        if self._openfermion_op is None:
            self._build_openfermion_operator()
        return self._openfermion_op

    @property
    def qiskit_op(self) -> PauliSumOp:
        """Returns the Qiskit qubit operator."""
        if self._qiskit_op is None:
            self._build_qiskit_operator()
        return self._qiskit_op

    def to_array(self, array_type: str = 'ndarray') -> np.ndarray:
        """Converts the molecular Hamiltonian to an array form.
        
        Args:
            array_type: A string indicating the type of the array.
        """
        assert array_type in ['sparse', 'matrix', 'array', 'ndarray']
        if self._openfermion_op is None:
            self._build_openfermion_operator()
        array = get_sparse_operator(self._openfermion_op)
        if array_type == 'sparse':
            return array
        elif array_type == 'matrix':
            return array.todense()
        elif array_type == 'array' or array_type == 'ndarray':
            return array.toarray()
