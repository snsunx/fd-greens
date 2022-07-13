"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from dataclasses import dataclass
from typing import List, Sequence
import h5py

HARTREE_TO_EV = 27.211386245988 # Hartree to electron volts conversion.
REVERSE_QUBIT_ORDER = False     # Whether to reverse qubit order for Qiskit qubit-order postprocessing.
CHECK_CIRCUITS: bool = True     # Whether to check circuits in each transpilation step.
QUBIT_OFFSET = 4

@dataclass
class CircuitConstructionParameters:
    """Circuit construction parameters.
    
    Args:
        WRAP_Z_AROUND_ITOFFOLI: Whether to wrap Z rotation gates around iToffoli for hardware runs.
        ITOFFOLI_Z_ANGLE: Adjustment Z gate angles in degree.
        CONSTRAIN_CS_CSD: Whether to force putting CS dagger gates on qubits 4 and 5 etc.
        CONVERT_CCZ_TO_ITOFFOLI: Whether to convert CCZ gates to iToffoli gates.
        ADJUST_CS_CSD: Whether to adjust CS/CSDs to the native CS/CSDs on corresponding qubit pairs.
        SPLIT_SIMULTANEOUS_CZS: Whether to split simultaneous CZ/CS/CSD onto different moments.
    """
    WRAP_Z_AROUND_ITOFFOLI: bool = True
    ITOFFOLI_Z_ANGLE: float = 20.67
    CONSTRAIN_CS_CSD: bool = False
    CONVERT_CCZ_TO_ITOFFOLI: bool = True
    
    ADJUST_CS_CSD: bool = True
    SPLIT_SIMULTANEOUS_CZS: bool = True

    def write(self, fname: str) -> None:
        with h5py.File(fname + '.h5', 'r+') as h5file:
            dset = h5file['params/circ']
            variables = vars(self)
            for key, value in variables.items():
                dset.attrs[key] = value

@dataclass
class ErrorMitigationParameters:
    """Error mitigation parameters.
    
    Args:
        PROJECT_DENSITY_MATRICES: Whether to project density matrices to being positive semidefinite.
        PURIFY_DENSITY_MATRICES: Whether to purify density matrices.
        USE_EXACT_TRACES: Use exact traces (probabilities) of the ancilla bitstrings.
    """
    PROJECT_DENSITY_MATRICES: bool = True
    PURIFY_DENSITY_MATRICES: bool = True
    USE_EXACT_TRACES: bool = False

    def write(self, fname: str) -> None:
        with h5py.File(fname + '.h5', 'r+') as h5file:
            dset = h5file['params/miti']
            variables = vars(self)
            for key, value in variables.items():
                dset.attrs[key] = value

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

    @classmethod
    def get_pairs(cls, spin: str) -> "MethodIndicesPairs":
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

        method_indices_pairs = cls(methods, indices)
        return method_indices_pairs
