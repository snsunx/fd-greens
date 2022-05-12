"""
===================================================
Qubit Indices (:mod:`fd_greens.main.qubit_indices`)
===================================================
"""

from typing import Mapping, Iterable, Tuple

import copy
import numpy as np
from typing import Union, Sequence, List, Optional


class QubitIndices:
    """Qubit indices of system and ancilla qubits."""

    def __init__(self, system_indices: List[List[int]], ancilla_indices: List[List[int]] = [[]]) -> None:
        """Initializes a ``QubitIndices`` object. 
        
        Example:
            qubit_indices = QubitIndices([[0, 1], [1, 1]], [[0]])

        Args:
            system_indices: A sequence of qubit indices in string, integer or list form.
            n_qubits: The number padded zeroes when the input is in int form. If the input is in
                str or list form this is not needed.
        """

        self.system_indices = system_indices
        self.ancilla_indices = ancilla_indices
        self._build()

    def _build(self) -> None:
        """Builds str, int and list forms of the qubit indices."""

        self.list = []
        for sys in self.system_indices:
            for anc in self.ancilla_indices:
                self.list.append(anc + sys)
    
        self.str = [''.join([str(i) for i in l[::-1]]) for l in self.list]
        self.int = [int(s, 2) for s in self.str]

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """Slices a 1D or 2D array by the qubit indices."""
        if len(arr.shape) == 1:
            return arr[self.int]
        elif len(arr.shape) == 2:
            return arr[self.int][:, self.int]
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

"""
# Qubit indices for Green's functions
eu_inds = QubitIndices(["1101", "0111"])
ed_inds = QubitIndices(["1110", "1011"])
hu_inds = QubitIndices(["0100", "0001"])
hd_inds = QubitIndices(["1000", "0010"])
e_inds = {"u": eu_inds, "d": ed_inds}
h_inds = {"u": hu_inds, "d": hd_inds}

# Qubit indices for charge-charge correlation functions
singlet_inds = QubitIndices(["0011", "0110", "1001", "1100"])
triplet_inds = QubitIndices(["0101", "1010"])
"""

e_inds = h_inds = singlet_inds = triplet_inds = None

def get_qubit_indices_dict(
    n_qubits: int, 
    spin: str, 
    method_indices_pairs: Iterable[Tuple[str, Sequence[int]]] = [],
    system_only: bool = False,
) -> Mapping[str, QubitIndices]:
    """Returns the ``QubitIndices`` dictionary of a certain number of qubits and spin.
    
    Args:
        n_qubits: The number of qubits.
        spin: The spin state of the electron-added states. Either "u" or "d".
        method_indices_pairs: A dictionary of transform methods to indices.
        system_only: Whether to only build system qubit indices.
    
    Returns:
        qubit_indices_dict: A dictionary from subscripts to qubit indices.
    """
    assert spin in ["u", "d"]

    def get_electron_index(index):
        electron_index = [1] * n_qubits
        electron_index[index] = 0
        return electron_index

    def get_hole_index(index):
        hole_index = [0] * n_qubits
        hole_index[index] = 1
        return hole_index

    # Create indices of electron- and hole-added states.
    empty_indices = list(range(spin == "u", n_qubits, 2))
    electron_indices = sorted([get_electron_index(i) for i in empty_indices])
    hole_indices = sorted([get_hole_index(i) for i in empty_indices])
    system_indices_dict = {"e": electron_indices, "h": hole_indices}

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
        qubit_indices_dict = dict()
        for subscript, ancilla_indices in ancilla_indices_dict.items():
            qubit_indices = QubitIndices(copy.deepcopy(system_indices_dict[subscript[0]]), ancilla_indices)
            qubit_indices.transform(method_indices_pairs)
            qubit_indices_dict[subscript] = qubit_indices

    return qubit_indices_dict
