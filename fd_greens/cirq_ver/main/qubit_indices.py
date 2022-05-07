"""
===================================================
Qubit Indices (:mod:`fd_greens.main.qubit_indices`)
===================================================
"""

import copy
import numpy as np
from typing import Union, Sequence, List, Optional

# QubitIndicesData = Sequence[Union[int, str, Sequence[int]]]


class QubitIndices:
    """Qubit indices of system and ancilla qubits."""

    def __init__(self, system_indices: List[List[int]], ancilla_indices: List[List[int]] = [[]]) -> None:
        """Initializes a ``QubitIndices`` object. 
        
        Each qubit index can be represented in three forms: str, int or list forms. For example, 
        '110', [0, 1, 1] and 6 refer to the same qubit index. Note that the indices in str or int 
        form follows Qiskit qubit order, but in list form follows normal qubit order.

        Args:
            qubit_inds_data: A sequence of qubit indices in string, integer or list form.
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
    
        self.str = ["".join([str(i) for i in l]) for l in self.list]
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

    @property
    def system(self) -> "QubitIndices":
        """Returns a ``QubitIndices`` object of the system qubit indices."""
        return self.__class__(system_indices=self.system_indices)

    @property
    def ancilla(self) -> "QubitIndices":
        """Returns a ``QubitIndices`` object of the ancilla qubit indices."""
        return self.__class__([[]], ancilla_indices=self.ancilla_indices)


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

eu_inds = ed_inds = hu_inds = hd_inds = e_inds = h_inds = singlet_inds = triplet_inds = None
