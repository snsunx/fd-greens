"""Utility class for handling qubit indices."""

import copy
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

    def include_ancilla(self, anc: str, inplace: bool = False) -> Optional['QubitIndices']:
        """Includes ancilla qubits as the first few qubits into a set of qubit indices.

        Args:
            anc: The ancilla qubit states in str form.
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

def cnot_indices(qubit_inds: 'QubitIndices', ctrl: int, targ: int) -> 'QubitIndices':
    """Applies a CNOT operation to qubit indices.
    
    Args:
        qubit_inds: The input QubitIndices object.
        ctrl: Index of the control qubit.
        targ: Index of the target qubit.

    Returns:
        The QubitIndices object after CNOT is applied.
    """
    data_list = qubit_inds.list_form
    data_list_new = []
    for q_ind in data_list:
        q_ind_new = q_ind.copy()
        q_ind_new[targ] = (q_ind[ctrl] + q_ind[targ]) % 2
        data_list_new.append(q_ind_new)
    qubit_inds_new = QubitIndices(data_list_new)
    return qubit_inds_new

def swap_indices(qubit_inds: 'QubitIndices', q1: int, q2: int) -> 'QubitIndices':
    """Applies a SWAP operation to qubit indices.
    
    Args:
        qubit_inds: The input QubitIndices object.
        q1: Index of the first qubit.
        q2: Index of the second qubit.

    Returns:
        The QubitIndices object after SWAP is applied.
    """
    data_list = qubit_inds.list_form
    data_list_new = []
    for q_ind in data_list:
        q_ind_new = q_ind.copy()
        q_ind_new[q2] = q_ind[q1]
        q_ind_new[q1] = q_ind[q2]
        data_list_new.append(q_ind_new)
    qubit_inds_new = QubitIndices(data_list_new)
    return qubit_inds_new

def taper_indices(qubit_inds: 'QubitIndices', inds_tapered: Sequence[int]) -> 'QubitIndices':
    """Tapers certain qubits off in a QubitIndices object.
    
    Args:
        qubit_inds: The input QubitIndices object.
        inds_tapered: The tapered qubit indices.
    
    Returns:
        The QubitIndices object after tapering.
    """
    qubit_inds_list = qubit_inds.list_form
    qubit_inds_list_new = []
    for q_ind in qubit_inds_list:
        q_ind_new = [q for i, q in enumerate(q_ind) if i not in inds_tapered]
        qubit_inds_list_new.append(q_ind_new)
    
    qubit_inds_new = QubitIndices(qubit_inds_list_new)
    return qubit_inds_new

def transform_4q_indices(
        qubit_inds: 'QubitIndices',
        swap: bool = True,
        tapered: bool = True
    ) -> 'QubitIndices':
    qubit_inds_new = cnot_indices(qubit_inds, 2, 0)
    qubit_inds_new = cnot_indices(qubit_inds_new, 3, 1)
    if swap:
        qubit_inds_new = swap_indices(qubit_inds_new, 2, 3)
    if tapered:
        qubit_inds_new = taper_indices(qubit_inds_new, [0, 1])
    return qubit_inds_new