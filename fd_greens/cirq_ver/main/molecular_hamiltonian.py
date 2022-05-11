"""
===================================================================
Molecular Hamiltonian (:mod:`fd_greens.main.molecular_hamiltonian`)
===================================================================
"""

from typing import Sequence, List, Tuple, Optional

import numpy as np
import cirq
from openfermion import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf

from .operators import OperatorsBase

HARTREE_TO_EV = 27.211386245988


class MolecularHamiltonian(OperatorsBase):
    """Molecular Hamiltonian."""

    def __init__(
        self,
        qubits: Sequence[cirq.Qid],
        geometry: List[Tuple[str, Sequence[float]]],
        basis: str,
        multiplicity: int = 1,
        charge: int = 0,
        run_pyscf_options: dict = {},
        occupied_indices: Optional[Sequence[int]] = None,
        active_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Initializes a ``MolecularHamiltonian`` object.

        Args:
            geometry: The coordinates of all atoms in the molecule, e.g. ``[['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]]``.
            basis: The basis set.
            multiplicity: The spin multiplicity.
            charge: The total molecular charge. Defaults to 0.
            run_pyscf_options: Keyword arguments passed to the run_pyscf function.
            occupied_indices: A list of spatial orbital indices indicating which orbitals 
                should be considered doubly occupied.
            active_indices: A list of spatial orbital indices indicating which orbitals
                should be considered active.
        """
        self.qubits = qubits
        self.n_qubits = len(qubits)

        molecule = MolecularData(geometry, basis, multiplicity=multiplicity, charge=charge)
        run_pyscf(molecule, **run_pyscf_options)
        self.n_electrons = molecule.n_electrons
        self.orbital_energies = molecule.orbital_energies

        self.occupied_indices = [] if occupied_indices is None else occupied_indices
        self.active_indices = range(molecule.n_orbitals) if active_indices is None else active_indices

        hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=self.occupied_indices, active_indices=self.active_indices)
        fermion_operator = get_fermion_operator(hamiltonian)
        qubit_operator = jordan_wigner(fermion_operator) * HARTREE_TO_EV
        qubit_operator.compress()

        pauli_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}
        self.pauli_strings = []
        for key, value in qubit_operator.terms.items():
            pauli_string = float(value) * cirq.PauliString()
            for index, symbol in key:
                pauli_string *= pauli_dict[symbol](qubits[index])
            self.pauli_strings.append(pauli_string)

    def transform(self, method_indices_pairs, tapered_state=None) -> None:
        if tapered_state is None:
            tapered_state = [1] * (self.n_qubits // 2)
        OperatorsBase.transform(self, method_indices_pairs, tapered_state=tapered_state)
        
    @property
    def matrix(self) -> np.ndarray:
        return sum(self.pauli_strings).matrix()

def get_lih_hamiltonian(bond_distance: float) -> MolecularHamiltonian:
    """Returns the HOMO-LUMO LiH Hamiltonian with bond length r.
    
    Args:
        bond_distance: The bond length of the molecule in Angstrom.
    
    Returns:
        hamiltonian: The molecular Hamiltonian.
    """
    hamiltonian = MolecularHamiltonian(
        cirq.LineQubit.range(4),
        [["Li", (0, 0, 0)], ["H", (0, 0, bond_distance)]], 
        "sto3g", 
        occupied_indices=[0],
        active_indices=[1, 2])
    return hamiltonian