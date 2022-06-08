"""
==============================================
Qubit Indices (:mod:`fd_greens.qubit_indices`)
==============================================
"""

from itertools import combinations
from os import system
from typing import Mapping, Iterable, Optional, List

import copy
import numpy as np

from .parameters import REVERSE_QUBIT_ORDER, MethodIndicesPairs


class QubitIndices:
    """Qubit indices of system and ancilla qubits."""

    def __init__(self, system_indices: List[List[int]], ancilla_indices: List[List[int]] = [[]]) -> None:
        """Initializes a ``QubitIndices`` object. 
        
        For example, if we are considering ``[0, 1]`` and ``[1, 1]`` on the system qubits, and ``[0]`` 
        on the ancilla, we would initialize an ``QubitIndices`` object as ::
            
            qubit_indices = QubitIndices([[0, 1], [1, 1]], [[0]])

        Args:
            system_indices: A sequence of system qubits indices.
            ancilla_indices: A sequence of ancilla qubit indices.
        """
        self.system_indices = system_indices
        self.ancilla_indices = ancilla_indices
        self.n_qubits = len(system_indices[0]) + len(ancilla_indices[0])
        self._build()

    def _build(self) -> None:
        """Builds str, int and list forms of the qubit indices."""
        self.list = []
        self.str = []
        self.int = []

        for sys in self.system_indices:
            for anc in self.ancilla_indices:
                self.list.append(anc + sys)
                string = ''.join(map(str, anc)) + ' ' + ''.join(map(str, sys))
                if REVERSE_QUBIT_ORDER:
                    string = string[::-1]
                self.str.append(string.strip())
                self.int.append(int(string.replace(' ', ''), 2))

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Slices a 1D or 2D array by the qubit indices."""
        if len(array.shape) == 1:
            return array[self.int]
        elif len(array.shape) == 2:
            return array[self.int][:, self.int]
        else:
            raise TypeError("The input array must be a 1D or 2D array.")

    def expand(self, array: np.ndarray) -> np.ndarray:
        """Expands a 1D or 2D array by the qubit indices."""
        if len(array.shape) == 1:
            print("Expand 1d array")
            array_new = np.zeros((2 ** self.n_qubits,), dtype=complex)
            print(f'{self.int = }')
            array_new[self.int] = array
        elif len(array.shape) == 2:
            print("Expand 2d array")
            array_new = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits), dtype=complex)
            array_new[np.repeat(self.int, self.n_qubits), self.int * self.n_qubits] = array.flatten()
        else:
            raise TypeError("The input array must be a 1D or 2D array.")
        return array_new


    def copy(self) -> "QubitIndices":
        """Returns a copy of itself."""
        return self.__class__(self.system_indices, ancilla_indices=self.ancilla_indices)

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

    def _cnot(self, control: int, target: int) -> None:
        """Applies a CNOT transformation to qubit indices."""
        # print(f"{self.system_indices = }")
        # print(f"{control = }, {target = }")
        for qubit_index in self.system_indices:
            qubit_index[target] = (qubit_index[control] + qubit_index[target]) % 2

    def _swap(self, index1: int, index2: int) -> None:
        """Applies a SWAP transformation to qubit indices."""
        for qubit_index in self.system_indices:
            qubit_index[index2], qubit_index[index1] = qubit_index[index1], qubit_index[index2]

    def _taper(self, *tapered_indices: Iterable[int]) -> None:
        """Tapers off certain qubit indices."""
        for i, qubit_index in enumerate(self.system_indices):
            self.system_indices[i] = [q for j, q in enumerate(qubit_index) if j not in tapered_indices]

    def transform(self, method_indices_pairs: MethodIndicesPairs) -> None:
        """Transforms a ``QubitIndices`` object."""
        method_dict = {"cnot": self._cnot, "swap": self._swap, "taper": self._taper}
        for method_key, indices in method_indices_pairs:
            method_dict[method_key](*indices)
        self._build()

    def _get_indices(n_qubits: int, states: str, spin: Optional[str] = None) -> List[List[int]]:
        """Returns the indices of given states and spin."""
        # ``states`` should be one of 'e' ((N+1)-electron states), 'h' ((N-1)-electron states),
        # 's' (N-electron singlet states), 't' (N-electron triplet states).
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
        method_indices_pairs: MethodIndicesPairs = [],
        system_only: bool = False
    ) -> Mapping[str, 'QubitIndices']:
        """Returns the qubit indices of (N±1)-electron states.
        
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
        method_indices_pairs: MethodIndicesPairs,
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
        system_indices_dict = {'n': QubitIndices._get_indices(n_qubits, 's')} # Only singlet states
        
        qubit_indices_dict = dict()
        if system_only:
            for subscript, system_indices in system_indices_dict.items():
                qubit_indices = QubitIndices(system_indices)
                qubit_indices.transform(method_indices_pairs)
                qubit_indices_dict[subscript] = qubit_indices
        else:
            ancilla_indices_dict = {'n': [[1]], 'np': [[1, 0]], 'nm': [[1, 1]]}
            for subscript, ancilla_indices in ancilla_indices_dict.items():
                qubit_indices = QubitIndices(copy.deepcopy(system_indices_dict[subscript[0]]), ancilla_indices)
                qubit_indices.transform(method_indices_pairs)
                qubit_indices_dict[subscript] = qubit_indices

        return qubit_indices_dict
