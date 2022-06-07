"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from dataclasses import dataclass
from typing import List, Sequence

HARTREE_TO_EV = 27.211386245988

# Whether to reverse qubit order for Qiskit-order postprocessing.
REVERSE_QUBIT_ORDER = False

# Whether to wrap Z rotation gates around iToffoli for hardware runs.
WRAP_Z_AROUND_ITOFFOLI = True

# Whether to put CS dagger gates on qubits 4 and 5.
CSD_IN_ITOFFOLI_ON_45 = True

# Whether to check original and transpiled circuits are equal.
CHECK_CIRCUIT_EQUAL = True

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
