"""
==============================================
Qubit Indices (:mod:`fd_greens.qubit_indices`)
==============================================
"""

from itertools import combinations
from typing import Mapping, Iterable, Optional, Tuple, Sequence, List

import copy
import numpy as np


class QubitIndices:
    """Qubit indices of system and ancilla qubits."""

    def __init__(self, system_indices: List[List[int]], ancilla_indices: List[List[int]] = [[]]) -> None:
        """Initializes a ``QubitIndices`` object. 
        
        Example:
            qubit_indices = QubitIndices([[0, 1], [1, 1]], [[0]])

        Args:
            system_indices: A sequence of system qubits indices.
            ancilla_indices: A sequence of ancilla qubit indices.
        """
        self.system_indices = system_indices
        self.ancilla_indices = ancilla_indices
        self._build()

    def _build(self) -> None:
        # Builds str, int and list forms of the qubit indices.
        self.list = []
        for sys in self.system_indices:
            for anc in self.ancilla_indices:
                self.list.append(anc + sys)
    
        # XXX: I don't think [::-1] is correct here.
        self.str = [''.join([str(i) for i in l[::-1]]) for l in self.list]
        self.int = [int(s, 2) for s in self.str]

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Slices a 1D or 2D array by the qubit indices."""
        if len(array.shape) == 1:
            return array[self.int]
        elif len(array.shape) == 2:
            return array[self.int][:, self.int]
        else:
            raise TypeError("The input array must be a 1D or 2D array.")

    def copy(self) -> "QubitIndices":
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return str(self.list)

    def __eq__(self, other: "QubitIndices") -> bool:
        return self.str == other.str

    def __iter__(self) -> str:
        for s in self.str:
            yield s

    @property
    def system(self) -> "QubitIndices":
        """Returns a ``QubitIndices`` object of the system qubit indices."""
        return self.__class__(system_indices=self.system_indices)

    @property
    def ancilla(self) -> "QubitIndices":
        """Returns a ``QubitIndices`` object of the ancilla qubit indices."""
        return self.__class__([[]], ancilla_indices=self.ancilla_indices)

    def cnot(self, control: int, target: int) -> None:
        """Applies a CNOT transformation to qubit indices."""
        # print(f"{self.system_indices = }")
        # print(f"{control = }, {target = }")
        for qubit_index in self.system_indices:
            qubit_index[target] = (qubit_index[control] + qubit_index[target]) % 2

    def swap(self, index1: int, index2: int) -> None:
        """Applies a SWAP transformation to qubit indices."""
        for qubit_index in self.system_indices:
            qubit_index[index2], qubit_index[index1] = qubit_index[index1], qubit_index[index2]

    def taper(self, *tapered_indices: Iterable[int]) -> None:
        """Tapers off certain qubit indices."""
        for i, qubit_index in enumerate(self.system_indices):
            self.system_indices[i] = [q for j, q in enumerate(qubit_index) if j not in tapered_indices]

    def transform(self, method_indices_pairs: Iterable[Tuple[str, Sequence[int]]]) -> None:
        """Transforms a ``QubitIndices`` object."""
        method_dict = {"cnot": self.cnot, "swap": self.swap, "taper": self.taper}
        for method_key, indices in method_indices_pairs:
            method_dict[method_key](*indices)
        self._build()

    def _get_indices(n_qubits: int, states: str, spin: Optional[str] = None) -> List[List[int]]:
        assert states in ['e', 'h', 's', 't']

        up_orbitals = range(0, n_qubits, 2)
        down_orbitals = range(1, n_qubits, 2)
        if states == 'e':
            n_up_electrons = n_qubits // 4 + (spin == 'u')
            n_down_electrons = n_qubits // 4 + (spin == 'd')
        elif states == 'h':
            n_up_electrons = n_qubits // 4 - 1 + (spin == 'd')
            n_down_electrons = n_qubits // 4 - 1 + (spin == 'u')
        elif states == 's':
            n_up_electrons = n_qubits // 4
            n_down_electrons = n_qubits // 4
        elif states == 't':
            n_up_electrons = n_qubits // 4 + 1
            n_down_electrons = n_qubits // 4 - 1

        up_locations = [list(x) for x in combinations(up_orbitals, n_up_electrons)]
        down_locations = [list(x) for x in combinations(down_orbitals, n_down_electrons)]
        all_locations = [x + y for x in up_locations for y in down_locations]
        if states == 't':
            up_locations = [list(x) for x in combinations(up_orbitals, n_down_electrons)]
            down_locations = [list(x) for x in combinations(down_orbitals,  n_up_electrons)]
            all_locations += [x + y for x in up_locations for y in down_locations]

        indices = []
        for location in all_locations:
            indices.append([1 if i in location else 0 for i in range(n_qubits)])
        return indices

    @staticmethod
    def get_eh_qubit_indices_dict(
        n_qubits: int,
        spin: str,
        method_indices_pairs: Iterable[Tuple[str, Sequence[int]]] = [],
        system_only: bool = False
    ) -> Mapping[str, 'QubitIndices']:
        """Returns the qubit indices of (N+-1)-electron states.
        
        Args:
            n_qubits: Number of qubits.
            spin: The spin of the second-quantized operators.
            method_indices_pairs: Pairs of Z2 transform methods and indices.
            system_only: Whether to return only system qubit indices.

        Returns:
            qubits_indices_dict: A dictionary from subscripts to qubit indices.
        """
        system_indices_dict = {'e': QubitIndices._get_indices(n_qubits, 'e', spin), 
                               'h': QubitIndices._get_indices(n_qubits, 'h', spin)}

        qubit_indices_dict = dict()
        if system_only:
            # Build qubit indices only on system indices.
            for subscript, system_indices in system_indices_dict.items():
                qubit_indices = QubitIndices(system_indices)
                qubit_indices.transform(method_indices_pairs)
                qubit_indices_dict[subscript] = qubit_indices
        else:
            # Create dictionaries of ancilla indices.
            ancilla_indices_dict = {
                "e": [[1]], "h": [[0]], 
                "ep": [[1, 0]], "em": [[1, 1]], "hp": [[0, 0]], "hm": [[0, 1]]
            }

            # Build the qubit indices for all six subscripts.
            for subscript, ancilla_indices in ancilla_indices_dict.items():
                qubit_indices = QubitIndices(copy.deepcopy(system_indices_dict[subscript[0]]), ancilla_indices)
                qubit_indices.transform(method_indices_pairs)
                qubit_indices_dict[subscript] = qubit_indices

        return qubit_indices_dict
        

    @staticmethod
    def get_excited_qubit_indices_dict(
        n_qubits: int,
        method_indices_pairs: Iterable[Tuple[str, Sequence[int]]],
        system_only: bool = False
    ) -> Mapping[str, 'QubitIndices']:
        """Returns the qubit indices of N-electron excited states.
        
        Args:
            n_qubits: Number of qubits.
            method_indices_pairs: Pairs of Z2 transform methods and indices.
            system_only: Whether to return only system qubit indices.

        Returns:
            qubit_indices_dict: A dictionary from subscripts ot qubit indices.
        """
        system_indices_dict = {'s': QubitIndices._get_indices(n_qubits, 's'),
                               't': QubitIndices._get_indices(n_qubits, 't')}
        
        qubit_indices_dict = dict()
        if system_only:
            for subscript, system_indices in system_indices_dict.items():
                qubit_indices = QubitIndices(system_indices)
                qubit_indices.transform(method_indices_pairs)
                qubit_indices_dict[subscript] = qubit_indices
        else:
            ancilla_indices_dict = {'n': [[1]], 'np': [[1, 0]], 'nm': [[1, 1]]}
            for subscript, ancilla_indices in ancilla_indices_dict.items():
                # XXX: 's' is hardcoded.
                qubit_indices = QubitIndices(copy.deepcopy(system_indices_dict['s']), ancilla_indices)
                qubit_indices.transform(method_indices_pairs)
                qubit_indices_dict[subscript] = qubit_indices

        return qubit_indices_dict
