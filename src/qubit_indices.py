"""Utility class for handling qubit indices."""

import copy
from typing import Union, Sequence, Optional
from math import ceil, log2

QubitIndicesForm = Sequence[Union[int, str, Sequence[int]]]

class QubitIndices:
    """A class to handle qubit indices in string, integer and list forms."""
    def __init__(self, 
                 data: QubitIndicesForm,
                 n_qubits: Optional[int] = None
                ) -> None:
        """Initializes a QubitIndices object.
        
        Args:   
            data: A sequence of qubit indices in string, integer or list form.
            n_qubits: The number of qubits used in padding zeroes 
                in string and list forms.
        """
        self._int = None
        self._list = None
        self._str = None
        if isinstance(data[0], int):
            self._int = data
        elif isinstance(data[0], list) or isinstance(data[0], tuple):
            self._list = [list(d) for d in data]
        elif isinstance(data[0], str):
            self._str = data
        self.n_qubits = n_qubits

        self._build()

    def _build(self):
        if self._str is None:
            self._build_str_form()
        if self._int is None:
            self._build_int_form() # build from string
        if self._list is None:
            self._build_list_form() # build from string

    def _build_str_form(self):
        if self._int is not None:
            if self.n_qubits is None:
                self.n_qubits = max([ceil(log2(i)) for i in self._int])
            self._str = [format(i, f'0{self.n_qubits}b') 
                         for i in self._int]
        elif self._list is not None:
            self._str =  [''.join([str(i) for i in l[::-1]]) 
                          for l in self._list]

    def _build_int_form(self):
        assert self._str is not None
        self._int = [int(s, 2) for s in self._str]

    def _build_list_form(self):
        assert self._str is not None
        self._list = [[int(c) for c in s[::-1]] for s in self._str]

    def include_ancilla(self, anc: str, inplace=False):
        """Includes ancilla qubits as the first few qubits into a set
        of qubit indices.
        
        Args:
            anc: The ancilla qubit states in string form.
            inplace: Whether the indices are modified inplace.

        Returns:
            (Optional) The new qubit indices with ancilla qubits.
        """
        str_with_anc  = [s + anc for s in self._str]
        if inplace:
            self._str = str_with_anc
            self._build_int_form()
            self._build_list_form()
        else:
            qubit_inds_with_anc = self.copy()
            qubit_inds_with_anc._str = str_with_anc
            qubit_inds_with_anc._build_int_form()
            qubit_inds_with_anc._build_list_form()
            return qubit_inds_with_anc

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return self._str

    @property
    def int_form(self):
        return self._int

    @property
    def list_form(self):
        return self._list

    @property
    def str_form(self):
        return self._str