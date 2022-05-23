"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from dataclasses import dataclass
from itertools import product
from typing import List, Sequence

import numpy as np

HARTREE_TO_EV = 27.211386245988

REVERSE_QUBIT_ORDER = True

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

# TODO: This should be moved to another place.
# Basis matrix for tomography
basis_matrix = []
if REVERSE_QUBIT_ORDER:
    bases = [(f'{x[0]}{x[3]}', f'{x[1]}{x[2]}') for x in product('xyz', 'xyz', '01', '01')]
else:
    bases = [(f'{x[0]}{x[2]}', f'{x[1]}{x[3]}') for x in product('xyz', 'xyz', '01', '01')]

states = {
    "x0": np.array([1.0, 1.0]) / np.sqrt(2),
    "x1": np.array([1.0, -1.0]) / np.sqrt(2),
    "y0": np.array([1.0, 1.0j]) / np.sqrt(2),
    "y1": np.array([1.0, -1.0j]) / np.sqrt(2),
    "z0": np.array([1.0, 0.0]),
    "z1": np.array([0.0, 1.0]),
}

for basis in bases:
    if REVERSE_QUBIT_ORDER:
        basis_state = np.kron(states[basis[1]], states[basis[0]])
    else:
        basis_state = np.kron(states[basis[0]], states[basis[1]])
    basis_vectorized = np.outer(basis_state, basis_state.conj()).reshape(-1)
    basis_matrix.append(basis_vectorized)

basis_matrix = np.array(basis_matrix)
