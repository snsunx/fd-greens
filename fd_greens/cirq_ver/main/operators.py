"""
===========================================
Operators (:mod:`fd_greens.main.operators`)
===========================================
"""

from typing import Callable, Tuple, Dict, Union, Sequence, Optional, List
import numpy as np
import cirq
from qiskit import *
from qiskit.quantum_info import PauliTable, SparsePauliOp
from qiskit.opflow import PauliSumOp

PauliOperator = Union[PauliSumOp, SparsePauliOp]

class SecondQuantizedOperators:
    """A class to store the X and Y parts of the creation and annihilation operators."""

    def __init__(self, n_qubits: int, factor: Union[int, float] = -1) -> None:
        """Initializes a SecondQuantizedOperators object.
        
        Args:
            n_qubits: The number of qubits in the second quantized operators.
            factor: A multiplication factor for simpler gate implementation.
        """
        self.n_qubits = n_qubits
        labels = ["I" * (n_qubits - i - 1) + "X" + "Z" * i for i in range(n_qubits)]
        labels += ["I" * (n_qubits - i - 1) + "Y" + "Z" * i for i in range(n_qubits)]
        pauli_table = PauliTable.from_labels(labels)
        coeffs = [1.0] * n_qubits + [1j] * n_qubits
        coeffs = factor * np.array(coeffs)
        self.sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=coeffs)

    def transform(
        self, transform_func: Callable[[PauliOperator], PauliOperator]
    ) -> None:
        """Transforms the set of second quantized operators by Z2 symmetries."""
        self.sparse_pauli_op = transform_func(self.sparse_pauli_op)
        # print(self.sparse_pauli_op.table.to_labels(), self.sparse_pauli_op.coeffs)

    def get_pauli_dict(self) -> Dict[int, Tuple[SparsePauliOp, SparsePauliOp]]:
        """Returns a dictionary of the second quantized operators."""
        operators_dict = {}
        for i in range(self.n_qubits):
            if i % 2 == 0:
                operators_dict[(i // 2, "u")] = self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
            else:
                operators_dict[(i // 2, "d")] = self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
        return operators_dict

    get_operators_dict = get_pauli_dict

class OperatorsBase:
    """Operators base."""

    def __init__(self) -> None:
        pass

    def cnot(self, control: int, target: int) -> None:
        for i, pauli_string in enumerate(self.pauli_strings):
            self.pauli_strings[i] = pauli_string.conjugated_by(cirq.CNOT(self.qubits[control], self.qubits[target]))

    def swap(self, index1: int, index2: int) -> None:
        for i, pauli_string in enumerate(self.pauli_strings):
            self.pauli_strings[i] = pauli_string.conjugated_by(cirq.SWAP(self.qubits[index1], self.qubits[index2]))

    def taper(self, tapered_state, *tapered_indices: Sequence[int]) -> None:
        """Tapers qubits off the first two qubits, assuming the symmetry operators are both Z."""
        for i, pauli_string in enumerate(self.pauli_strings):
            coefficient = pauli_string.coefficient
            pauli_mask = pauli_string.dense(self.qubits).pauli_mask
            # print(f'{coefficient = }')
            # print(f'{pauli_string.dense(self.qubits) = }')
            # print(f'{tapered_state = }')
            for state, index in zip(tapered_state, tapered_indices):
                if pauli_mask[index] == 2:
                    coefficient *= (-1) ** state * 1j
                elif pauli_mask[index] == 3:
                    coefficient *= (-1) ** state
            # print(f'coefficient_new = {coefficient}')
            # print('-' * 80)
            pauli_mask_new = [pauli_mask[i] for i in range(self.n_qubits) if i not in tapered_indices]
            self.pauli_strings[i] = cirq.DensePauliString(pauli_mask_new, coefficient=coefficient).sparse()

    def transform(self, method_indices_pairs, tapered_state=None):
        method_dict = {"cnot": self.cnot, "swap": self.swap, "taper": self.taper}
        for method_key, indices in method_indices_pairs:
            if method_key != 'taper':
                method_dict[method_key](*indices)
            else:
                method_dict[method_key](tapered_state, *indices)


class SecondQuantizedOperators1(OperatorsBase):
    """Second quantized operators."""

    def __init__(self, qubits: Sequence[cirq.Qid], spin: str, factor: float = -1.0) -> None:
        self.qubits = qubits
        self.n_qubits = len(qubits)

        self.pauli_strings = []
        for i in range(self.n_qubits // 2):
            z_chain = [cirq.Z(qubits[j]) for j in range(2 * i + (spin == 'd'))]
            pauli_string_x = cirq.PauliString(z_chain + [cirq.X(qubits[2 * i + (spin == 'd')])])
            pauli_string_y = cirq.PauliString(z_chain + [cirq.Y(qubits[2 * i + (spin == 'd')])])
            # print('pauli_string_x =', pauli_string_x.dense(self.qubits))
            # print('pauli_string_y =', pauli_string_y.dense(self.qubits))
            self.pauli_strings.append(factor * pauli_string_x)
            self.pauli_strings.append(factor * 1j * pauli_string_y)

    def transform(self, method_indices_pairs):
        '''if 'taper' in [t[0] for t in method_indices_pairs]:
            indices = [1] * (self.n_qubits // 2) + [0] * (self.n_qubits // 2)
            indices[self.n_qubits // 2 + (self.spin == 'd')] = 1
            qubit_indices = QubitIndices([indices])
            qubit_indices.transform(method_indices_pairs)
            tapered_state = qubit_indices.system_indices[0]
            # TODO: Should keep track of the symmetry operators as well.
        else:
            tapered_state = None
        '''
        tapered_state = [1] * (self.n_qubits // 2)
        OperatorsBase.transform(self, method_indices_pairs, tapered_state=tapered_state)

    def __len__(self):
        return len(self.pauli_strings)
        
    def __getitem__(self, m):
        pauli_string = self.pauli_strings[m]
        # max_index = max(pauli_string._qubit_pauli_map.keys()).x
        dense_pauli_string = pauli_string.dense(self.qubits[:2]) # XXX: 2 is hardcoded
        return dense_pauli_string


class ChargeOperators:
    """A class to store U01 and U10 for calculating charge-charge response functions."""

    def __init__(self, n_qubits: int) -> None:
        """Initializes a ChargeOperators object.
        
        Args:
            The number of qubits in the charge operators.
        """
        self.n_qubits = n_qubits

        labels = ["I" * n_qubits for _ in range(n_qubits)]
        labels += ["I" * (n_qubits - i - 1) + "Z" + "I" * i for i in range(n_qubits)]
        pauli_table = PauliTable.from_labels(labels)
        coeffs = [1.0] * 2 * n_qubits
        self.sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=coeffs)

    def transform(
        self, transform_func: Callable[[PauliOperator], PauliOperator]
    ) -> None:
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
                dic[(i // 2, "u")] = (
                    self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
                )
            else:
                dic[(i // 2, "d")] = (
                    self.sparse_pauli_op[i] + self.sparse_pauli_op[i + self.n_qubits]
                )
        return dic

class ChargeOperators1(OperatorsBase):
    def __init__(self, qubits):
        self.qubits = qubits
        self.n_qubits = len(qubits)

        self.pauli_strings = []
        for i in range(self.n_qubits):
            pauli_string_i = cirq.PauliString()
            pauli_string_z = cirq.PauliString(cirq.Z(qubits[i]))
            self.pauli_strings.append(pauli_string_i)
            self.pauli_strings.append(pauli_string_z)

    def transform(self, method_indices_pairs):
        '''if 'taper' in method_indices_pairs:
            indices = [1] * (self.n_qubits // 2) + [0] * (self.n_qubits // 2)
            indices[self.n_qubits // 2 + (self.spin == 'd')] = 1
            qubit_indices = QubitIndices([indices])
            qubit_indices.transform(method_indices_pairs)
            tapered_state = qubit_indices.system_indices[0]
            # TODO: Should keep track of the symmetry operators as well.
        '''

        tapered_state = [1] * (self.n_qubits // 2)
        OperatorsBase.transform(self, method_indices_pairs, tapered_state=tapered_state)

    @property
    def operators(self):
        return