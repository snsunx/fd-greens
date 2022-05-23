"""
======================================
Operators (:mod:`fd_greens.operators`)
======================================
"""

from typing import Sequence, Mapping, Optional, List

import cirq

from .parameters import MethodIndicesPairs

class OperatorsBase:
    """Operators base."""

    def __init__(self) -> None:
        pass

    def _cnot(self, control: int, target: int) -> None:
        """Conjugates Pauli strings by a CNOT operation."""
        for i, pauli_string in enumerate(self.pauli_strings):
            self.pauli_strings[i] = pauli_string.conjugated_by(cirq.CNOT(self.qubits[control], self.qubits[target]))

    def _swap(self, index1: int, index2: int) -> None:
        """Conjugates Pauli strings by a SWAP operation."""
        for i, pauli_string in enumerate(self.pauli_strings):
            self.pauli_strings[i] = pauli_string.conjugated_by(cirq.SWAP(self.qubits[index1], self.qubits[index2]))

    def _taper(self, tapered_state: Sequence[int], *tapered_indices: Sequence[int]) -> None:
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

    def transform(
        self, 
        method_indices_pairs: MethodIndicesPairs, 
        tapered_state: Optional[Sequence[int]] = None
    ) -> None:
        """Transforms Pauli strings with Z2 symmetries and qubit tapering."""
        method_dict = {"cnot": self._cnot, "swap": self._swap, "taper": self._taper}
        for method_key, indices in method_indices_pairs:
            if method_key != 'taper':
                method_dict[method_key](*indices)
            else:
                method_dict[method_key](tapered_state, *indices)

    def __len__(self) -> int:
        return len(self.pauli_strings)

    def __getitem__(self, m) -> cirq.DensePauliString:
        pauli_string = self.pauli_strings[m]
        max_index = self.n_qubits - self.n_tapered
        dense_pauli_string = pauli_string.dense(self.qubits[:max_index])
        return dense_pauli_string


class SecondQuantizedOperators(OperatorsBase):
    """Second quantized operators."""

    def __init__(self, qubits: Sequence[cirq.Qid], spin: str, factor: float = -1.0) -> None:
        """Initializes a ``SecondQuantizedOperators`` object.
        
        Args:
            qubits: Qubits on which the second quantized operators act.
            spin: The spin of the second quantized operators. Either ``'u'`` or ``'d'``.
            factor: Multiplication factor of the operators for easy gate construction.
        """
        self.qubits = qubits
        self.n_qubits = len(qubits)
        self.n_tapered = 0

        self.pauli_strings = []
        for i in range(self.n_qubits // 2):
            z_chain = [cirq.Z(qubits[j]) for j in range(2 * i + (spin == 'd'))]
            pauli_string_x = cirq.PauliString(z_chain + [cirq.X(qubits[2 * i + (spin == 'd')])])
            pauli_string_y = cirq.PauliString(z_chain + [cirq.Y(qubits[2 * i + (spin == 'd')])])
            self.pauli_strings.append(factor * pauli_string_x)
            self.pauli_strings.append(factor * 1j * pauli_string_y)

    def transform(self, method_indices_pairs: MethodIndicesPairs, tapered_state: Optional[List[int]] = None) -> None:
        """Transforms the second quantized operators with Z2 symmetries and qubit tapering."""
        self.n_tapered = method_indices_pairs.n_tapered
        if tapered_state is None:
            tapered_state = [1] * self.n_tapered
        OperatorsBase.transform(self, method_indices_pairs, tapered_state=tapered_state)

class ChargeOperators(OperatorsBase):
    """Charge operators."""

    def __init__(self, qubits: Sequence[cirq.Qid]) -> None:
        """Initializes a ``ChargeOperators`` object.
        
        Args:
            qubits: Qubits on which the charge operators act.
        """
        self.qubits = qubits
        self.n_qubits = len(qubits)
        self.n_tapered = 0

        self.pauli_strings = []
        for i in range(self.n_qubits):
            pauli_string_i = cirq.PauliString()
            pauli_string_z = cirq.PauliString(cirq.Z(qubits[i]))
            self.pauli_strings.append(pauli_string_i)
            self.pauli_strings.append(pauli_string_z)

    def transform(self, method_indices_pairs: MethodIndicesPairs, tapered_state: Optional[List[int]] = None) -> None:
        """Transforms the charge operators with Z2 symmetries and qubit tapering."""
        self.n_tapered = method_indices_pairs.n_tapered
        if tapered_state is None:
            tapered_state = [1] * self.n_tapered
        OperatorsBase.transform(self, method_indices_pairs, tapered_state=tapered_state)