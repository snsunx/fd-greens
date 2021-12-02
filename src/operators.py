"""Operator utility functions"""

from typing import Callable, Tuple, Dict
import numpy as np
from qiskit.quantum_info import PauliTable, SparsePauliOp
from z2_symmetries import transform_4q_hamiltonian

def get_a_operator(n_qubits: int, ind: int) -> SparsePauliOp:
    """Returns the creation/annihilation operator.

    Args:
        n_qubits: Number of qubits.
        ind: The index of the creation/annihilation operator.

    Returns:
        The X and Y part of the creation/annihilation operator as a
            SparsePauliOp.
    """
    label_x = 'I' * (n_qubits - ind - 1) + 'X' + 'Z' * ind
    label_y = 'I' * (n_qubits - ind - 1) + 'Y' + 'Z' * ind
    pauli_table = PauliTable.from_labels([label_x, label_y])
    sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=[1.0, 1.0j])
    return sparse_pauli_op

# XXX: Treating the spin this way is not correct. See Markdown document.
def get_operator_dictionary(spin='up'):
    """Returns the operator dictionary for restricted orbital calculations."""
    labels_x = ['IIIX', 'IIXZ', 'IXZZ', 'XZZZ']
    labels_y = ['IIIY', 'IIYZ', 'IYZZ', 'YZZZ']
    pauli_table = PauliTable.from_labels(labels_x + labels_y)
    sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=[1.] * 4 + [1j] * 4)

    sparse_pauli_op = transform_4q_hamiltonian(sparse_pauli_op, init_state=[1, 1])
    dic = {}
    if spin == 'up':
        for i in range(2):
            dic.update({i: [sparse_pauli_op[2 * i], sparse_pauli_op[2 * i + 4]]})
    else:
        for i in range(2):
            dic.update({i: [sparse_pauli_op[2 * i + 1], sparse_pauli_op[2 * i + 1 + 4]]})
    return dic

class SecondQuantizedOperators:
    """A class to store the X and Y parts of the second quantized operators."""

    def __init__(self, n_qubits):
        """Creates an object containing second quantized operators."""
        self.n_qubits = n_qubits

        labels_x = ['I' * (n_qubits - i - 1) + 'X' + 'Z' * i for i in range(n_qubits)]
        labels_y = ['I' * (n_qubits - i - 1) + 'Y' + 'Z' * i for i in range(n_qubits)]
        labels = labels_x + labels_y
        pauli_table = PauliTable.from_labels(labels)
        # print(pauli_table.to_labels())
        coeffs = [1.] * n_qubits + [1j] * n_qubits
        coeffs = -1 * np.array(coeffs) # THIS IS HARDCODED FOR THE SPECIAL CASE
        self.sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=coeffs)

    def transform(self, transform_func: Callable) -> None:
        """Transforms the set of second quantized operators by Z2 symmetries."""
        self.sparse_pauli_op = transform_func(self.sparse_pauli_op)
        # print(self.sparse_pauli_op.table.to_labels(), self.sparse_pauli_op.coeffs)

    # TODO: Deprecate this function.
    def get_op_dict(self, spin: str = 'up') -> Dict[int, Tuple[SparsePauliOp, SparsePauliOp]]:
        """Returns the second quantized operators in a certain spin state."""
        assert spin in ['up', 'down']
        dic = {}
        for i in range(self.n_qubits // 2):
            if spin == 'up':
                x_op = self.sparse_pauli_op[2 * i]
                y_op = self.sparse_pauli_op[2 * i + self.n_qubits]
            elif spin == 'down':
                x_op = self.sparse_pauli_op[2 * i + 1]
                y_op = self.sparse_pauli_op[2 * i + 1 + self.n_qubits]
            dic.update({i: [x_op, y_op]})
        return dic

    def get_op_dict_all(self) -> Dict[int, Tuple[SparsePauliOp, SparsePauliOp]]:
        """Returns the second quantized operators in a certain spin state."""
        dic = {}
        for i in range(self.n_qubits):
            if i % 2 == 0:
                dic[(i // 2, 'u')] = [self.sparse_pauli_op[i], self.sparse_pauli_op[i + self.n_qubits]]
            else:
                dic[(i // 2, 'd')] = [self.sparse_pauli_op[i], self.sparse_pauli_op[i + self.n_qubits]]
        return dic
