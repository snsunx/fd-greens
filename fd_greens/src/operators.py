"""Operator utility functions"""

from typing import Callable, Tuple, Dict, Union, Sequence, Optional
import numpy as np
from qiskit import *
from qiskit.quantum_info import PauliTable, SparsePauliOp
from qiskit.opflow import PauliSumOp

PauliOperator = Union[PauliSumOp, SparsePauliOp]

class SecondQuantizedOperators:
    """A class to store the X and Y parts of the creation and annihilation operators."""

    def __init__(self, n_qubits: int, factor: Union[int, float] = -1) -> None:
        """Initializes a SecondQuantizedOperators object.
        
        Args:
            n_qubits: The number of qubits in the creation and annihilation operators.
            factor: A multiplication factor for simpler gate implementation.
        """
        self.n_qubits = n_qubits
        labels = ['I' * (n_qubits - i - 1) + 'X' + 'Z' * i for i in range(n_qubits)]
        labels += ['I' * (n_qubits - i - 1) + 'Y' + 'Z' * i for i in range(n_qubits)]
        pauli_table = PauliTable.from_labels(labels)
        coeffs = [1.] * n_qubits + [1j] * n_qubits
        coeffs = factor * np.array(coeffs)
        self.sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=coeffs)

    def transform(self, transform_func: Callable[[PauliOperator], PauliOperator]) -> None:
        """Transforms the set of second quantized operators by Z2 symmetries."""
        self.sparse_pauli_op = transform_func(self.sparse_pauli_op)
        # print(self.sparse_pauli_op.table.to_labels(), self.sparse_pauli_op.coeffs)

    def get_pauli_dict(self) -> Dict[int, Tuple[SparsePauliOp, SparsePauliOp]]:
        """Returns a dictionary of the second quantized operators."""
        dic = {}
        for i in range(self.n_qubits):
            if i % 2 == 0:
                dic[(i // 2, 'u')] = self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
            else:
                dic[(i // 2, 'd')] = self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
        return dic

class ChargeOperators:
    """A class to store U01 and U10 for calculating charge-charge response functions."""

    def __init__(self, n_qubits: int) -> None:
        """Initializes a ChargeOperators object.
        
        Args:
            The number of qubits in the charge operators.
        """
        self.n_qubits = n_qubits

        labels = ['I' * n_qubits for _ in range(n_qubits)]
        labels += ['I' * (n_qubits - i - 1) + 'Z' + 'I' * i for i in range(n_qubits)]
        pauli_table = PauliTable.from_labels(labels)
        coeffs = [1.] * 2 * n_qubits
        self.sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=coeffs)

    def transform(self, transform_func: Callable[[PauliOperator], PauliOperator]) -> None:
        """Transforms the set of second quantized operators by Z2 symmetries."""
        self.sparse_pauli_op = transform_func(self.sparse_pauli_op)

    def get_pauli_dict(self) -> Dict[int, SparsePauliOp]:
        """Returns a dictionary of the charge U operators."""
        # for i in range(self.n_qubits):
        #     if i % 2 == 0:
        #         dic[(i // 2, 'u')] = self.sparse_pauli_op[i]
        #     else:
        #         dic[(i // 2, 'd')] = self.sparse_pauli_op[i]

        dic = {}
        for i in range(self.n_qubits):
            if i % 2 == 0:
                dic[(i // 2, 'u')] = self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
            else:
                dic[(i // 2, 'd')] = self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
        return dic