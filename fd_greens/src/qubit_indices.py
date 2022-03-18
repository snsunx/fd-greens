"""Utility class for handling qubit indices."""

import copy
import numpy as np
from typing import Union, Sequence, List, Optional
from math import ceil, log2

QubitIndicesData = Sequence[Union[int, str, Sequence[int]]]

class QubitIndices:
    """A class to handle qubit indices in str, int and list forms."""

    def __init__(self,
                 qubit_inds_data: QubitIndicesData,
                 n_qubits: Optional[int] = None
                ) -> None:
        """Initializes a QubitIndices object. 
        
        Each qubit index can be represented in three forms: str, int or list forms. For example, 
        '110', [0, 1, 1] and 6 refer to the same qubit index. Note that the indices in str or int 
        form follows Qiskit qubit order, but in list form follows normal qubit order.

        Args:
            qubit_inds_data: A sequence of qubit indices in string, integer or list form.
            n_qubits: The number padded zeroes when the input is in int form. If the input is in
                str or list form this is not needed.
        """
        self._int = None
        self._list = None
        self._str = None

        if isinstance(qubit_inds_data[0], int):
            self._int = qubit_inds_data
        elif isinstance(qubit_inds_data[0], list) or isinstance(qubit_inds_data[0], tuple):
            self._list = [list(d) for d in qubit_inds_data]
        elif isinstance(qubit_inds_data[0], str):
            self._str = qubit_inds_data
        self.n_qubits = n_qubits

        self._build()
        
        self.n = len(self.str_form)

    def _build(self) -> None:
        """Builds str, int and list forms of the qubit indices."""
        if self._str is None:
            self._build_str_form()
        if self._int is None:
            self._build_int_form() # build from str form
        if self._list is None:
            self._build_list_form() # build from str form

    def _build_str_form(self) -> None:
        """Builds the str form of the qubit indices from int or list form."""
        if self._int is not None:
            if self.n_qubits is None:
                self.n_qubits = max([ceil(log2(i)) for i in self._int])
            self._str = [format(i, f'0{self.n_qubits}b') for i in self._int]
        elif self._list is not None:
            self._str =  [''.join([str(i) for i in l[::-1]]) for l in self._list]

    def _build_int_form(self) -> None:
        """Builds int form of the qubit indices from str form."""
        assert self._str is not None
        self._int = [int(s, 2) for s in self._str]

    def _build_list_form(self) -> None:
        """Builds list form of the qubit indices from str form."""
        assert self._str is not None
        self._list = [[int(c) for c in s[::-1]] for s in self._str]

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """Slices a 1D or 2D array by the qubit indices."""
        if len(arr.shape) == 1:
            return arr[self._int]
        elif len(arr.shape) == 2:
            return arr[self._int][:, self._int]
    
    def insert_ancilla(self,
                       qind_anc: Union[str, Sequence[int]],
                       loc: Optional[Sequence[int]] = None
                       ) -> 'QubitIndices':
        """Inserts ancilla qubit indices into a QubitIndices object.

        Args:
            qind_anc: The ancilla qubit states in string or list form.
            loc: Location of the ancilla indices. Default to the first few locations.
        
        Returns:
            The new qubit indices with ancilla qubits.
        """
        # Convert qind_anc to list form.
        if isinstance(qind_anc, str):
            qind_anc = [int(c) for c in qind_anc[::-1]]
        
        qinds_list = copy.deepcopy(self._list)
        qinds_data_new = []
        for qind in qinds_list:
            if loc is None: # Default to the first few locations
                qind = qind_anc + qind
            else: # Insert into the given locations
                for i, q in zip(loc, qind_anc):
                    qind.insert(i, q)
            qinds_data_new.append(qind)
        
        qinds_new = QubitIndices(qinds_data_new)
        return qinds_new

    # XXX: Isn't this same as include_ancilla?
    def __add__(self, other: 'QubitIndices') -> 'QubitIndices':
        """Includes a single ancilla qubit index into the qubit indices."""
        assert other.n == 1
        anc = other.str_form[0]
        data_with_anc  = [s + anc for s in self.str_form]
        qubit_inds_new = QubitIndices(data_with_anc)
        return qubit_inds_new

    def __or__(self, other: 'QubitIndices') -> 'QubitIndices':
        """Combines qubit indices from two QubitIndices objects."""
        self_str = self.str_form
        other_str = other.str_form
        str_form = sorted(self_str + other_str)
        qubit_inds_new = QubitIndices(str_form)
        return qubit_inds_new

    def copy(self) -> 'QubitIndices':
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return str(self._str)
    
    def __eq__(self, other: 'QubitIndices') -> bool:
        return self.str_form == other.str_form

    @property
    def int_form(self) -> List[int]:
        """Returns the int form of the qubit indices."""
        return self._int

    @property
    def list_form(self) -> List[List[int]]:
        """Returns the list form of the qubit indices."""
        return self._list

    @property
    def str_form(self) -> List[str]:
        """Returns the str form of the qubit indices."""
        return self._str