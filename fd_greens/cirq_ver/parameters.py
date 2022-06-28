"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from dataclasses import dataclass
from typing import List, Sequence

HARTREE_TO_EV = 27.211386245988 # Hartree to electron volts conversion.
REVERSE_QUBIT_ORDER = False     # Whether to reverse qubit order for Qiskit qubit-order postprocessing.
PROJECT_DENSITY_MATRICES = True # Whether to project density matrices to be positive semidefinite.
PURIFY_DENSITY_MATRICES = True  # Whether to purify density matrices.

# Circuit construction parameters.
ADJUST_CS_CSD_GATES = True      # Whether to apply native CS/CSD gates on corresponding qubit pairs.
WRAP_Z_AROUND_ITOFFOLI = True   # Whether to wrap Z rotation gates around iToffoli for hardware runs.
ITOFFOLI_Z_ANGLE = 20.67        # Adjustment Z gate angles in degree.
CSD_IN_ITOFFOLI_ON_45 = False   # Whether to force plutting CS dagger gates on qubits 4 and 5.
SPLIT_SIMULTANEOUS_CZS = True   # Whether to split simultaneous CZ/CS/CSD onto different moments.
CHECK_CIRCUITS = True           # Whether to check circuits in transpilation.


@dataclass
class MethodIndicesPairs:
    """Method indices pairs used in Z2 symmetry conversion."""
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
    """Returns the method indices pairs corresponding to a given spin label.
    
    Args:
        spin: The spin label. Either ``'u'`` or ``'d'``.
    
    Returns:
        method_indices_pairs: The method indices pairs.
    """
    assert spin in ['u', 'd', '']

    if spin == 'u':
        methods = ['cnot', 'cnot', 'taper']
        indices = [(2, 0), (3, 1), (0, 1)]

    elif spin == 'd':
        methods = ['cnot', 'cnot', 'swap', 'taper']
        indices = [(2, 0), (3, 1), (2, 3), (0, 1)]

    elif spin == '':
        methods = ['cnot', 'cnot', 'swap', 'taper']
        indices = [(2, 0), (3, 1), (2, 3), (0, 1)]

    method_indices_pairs = MethodIndicesPairs(methods, indices)
    return method_indices_pairs
