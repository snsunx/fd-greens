"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from dataclasses import dataclass
from typing import List, Sequence

# Hartree to electron volts conversion.
HARTREE_TO_EV = 27.211386245988

# Whether to reverse qubit order for Qiskit qubit-order postprocessing. Should be set to False.
REVERSE_QUBIT_ORDER = False

# Whether to apply native CS/CSD gates on qubit pairs for hardware runs. Should be set to True.
ADJUST_CS_CSD_GATES = True

# Whether to wrap Z rotation gates around iToffoli for hardware runs. Should be set to True.
WRAP_Z_AROUND_ITOFFOLI = True

# Whether to force putting CS dagger gates on qubits 4 and 5. Should be set to False.
CSD_IN_ITOFFOLI_ON_45 = False

# Whether to check original and transpiled circuits are equal. Should be set to True.
CHECK_CIRCUITS = True

# Linestyles for spectral function, trace of self-energy and response function.
linestyles_A = [{}, {"ls": "--", "marker": "x", "markevery": 30}]
linestyles_TrSigma = [
    {"color": "xkcd:red"},
    {"color": "xkcd:blue"},
    {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:rose pink"},
    {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:azure"},
]
linestyles_chi = [
    {"color": "xkcd:red"},
    {"color": "xkcd:blue"},
    {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:rose pink"},
    {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:azure"},
]



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
    assert spin in ['u', 'd']

    if spin == 'u':
        methods = ['cnot', 'cnot', 'taper']
        indices = [(2, 0), (3, 1), (0, 1)]

    elif spin == 'd':
        methods = ['cnot', 'cnot', 'swap', 'taper']
        indices = [(2, 0), (3, 1), (2, 3), (0, 1)]

    method_indices_pairs = MethodIndicesPairs(methods, indices)
    return method_indices_pairs
