"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from dataclasses import dataclass
from typing import List, Sequence

HARTREE_TO_EV = 27.211386245988

REVERSE_QUBIT_ORDER = False

WRAP_Z_AROUND_ITOFFOLI = True

@dataclass
class MethodIndicesPairs:
    methods: List[str]
    indices: List[Sequence[int]]

    @property
    def n_tapered(self):
        if 'taper' in self.methods:
            index  = self.methods.index('taper')
            return len(self.indices[index])
        else:
            return 0

    def __iter__(self):
        return zip(self.methods, self.indices)


def get_method_indices_pairs(spin: str) -> MethodIndicesPairs:
    assert spin in ['u', 'd']

    if spin == 'u':
        methods = ['cnot', 'cnot', 'taper']
        indices = [(2, 0), (3, 1), (0, 1)]

    elif spin == 'd':
        methods = ['cnot', 'cnot', 'swap', 'taper']
        indices = [(2, 0), (3, 1), (2, 3), (0, 1)]

    method_indices_pairs = MethodIndicesPairs(methods, indices)
    return method_indices_pairs

# XXX: For import issues.
method_indices_pairs = None
basis_matrix = None
