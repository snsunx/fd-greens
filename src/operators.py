"""Operator utility functions"""

from typing import Callable, Tuple, Dict, Union, Sequence, Optional, List
import numpy as np
from qiskit import *
from qiskit.quantum_info import PauliTable, SparsePauliOp
from qiskit.opflow import PauliSumOp

class SecondQuantizedOperators:
    """A class to store the X and Y parts of the second quantized operators."""

    def __init__(self, n_qubits, factor=-1):
        """Creates an object containing second quantized operators."""
        self.n_qubits = n_qubits

        labels_x = ['I' * (n_qubits - i - 1) + 'X' + 'Z' * i for i in range(n_qubits)]
        labels_y = ['I' * (n_qubits - i - 1) + 'Y' + 'Z' * i for i in range(n_qubits)]
        labels = labels_x + labels_y
        pauli_table = PauliTable.from_labels(labels)
        # print(pauli_table.to_labels())
        coeffs = [1.] * n_qubits + [1j] * n_qubits
        coeffs = factor * np.array(coeffs) # THIS IS HARDCODED TO -1 FOR SPECIAL CASE
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
        """Returns a dictionary of the second quantized operators."""
        dic = {}
        for i in range(self.n_qubits):
            if i % 2 == 0:
                dic[(i // 2, 'u')] = [self.sparse_pauli_op[i], self.sparse_pauli_op[i + self.n_qubits]]
            else:
                dic[(i // 2, 'd')] = [self.sparse_pauli_op[i], self.sparse_pauli_op[i + self.n_qubits]]
        return dic

class ChargeOperators:
    """A class to store U01 and U10 for calculating charge-charge response functions."""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

        labels = ['I' * (n_qubits - i - 1) + 'Z' + 'I' * i for i in range(n_qubits)]
        pauli_table = PauliTable.from_labels(labels)
        self.sparse_pauli_op = SparsePauliOp(pauli_table)

    def transform(self, transform_func: Callable) -> None:
        """Transforms the set of second quantized operators by Z2 symmetries."""
        self.sparse_pauli_op = transform_func(self.sparse_pauli_op)

    def get_op_dict_all(self) -> Dict[int, SparsePauliOp]:
        """Returns a dictionary of the charge U operators."""
        dic = {}
        for i in range(self.n_qubits):
            if i % 2 == 0:
                dic[(i // 2, 'u')] = self.sparse_pauli_op[i]
            else:
                dic[(i // 2, 'd')] = self.sparse_pauli_op[i]
        return dic


def apply_cnot_z2(op: Union[PauliSumOp, SparsePauliOp],
                  ctrl: int,
                  targ: int) -> Union[PauliSumOp, SparsePauliOp]:
    """Applies a CNOT gate to the Z2 representation of a Pauli string.

    Note that the operator in string form follows Qiskit qubit order,
    but in vector (PauliTable) form follows normal qubit order.

    Args:
        pauli_sum_op: An operator on which a CNOT is to be applied.
        ctrl: An integer indicating the index of the control qubit.
        targ: An integer indicating the index of the target qubit.

    Returns:
        The new operator after CNOT is applied.
    """
    if isinstance(op, PauliSumOp):
        sparse_pauli_op = op.primitive.copy()
    elif isinstance(op, SparsePauliOp):
        sparse_pauli_op = op.copy()
    else:
        raise TypeError("Operator must be a PauliSumOp or SparsePauliOp")

    coeffs = sparse_pauli_op.coeffs.copy()
    x_arr = sparse_pauli_op.table.X.copy()
    z_arr = sparse_pauli_op.table.Z.copy()
    n_paulis, n_qubits = x_arr.shape

    # Flip sign when the pauli is X_cZ_t or Y_cY_t
    for i in range(n_paulis):
        x = x_arr[i]
        z = z_arr[i]
        if z[targ] == x[ctrl] == True and z[ctrl] == x[targ]:
            coeffs[i] *= -1

    # Apply binary addition forward on the X part, backward on the Z part
    x_arr[:, targ] ^= x_arr[:, ctrl]
    z_arr[:, ctrl] ^= z_arr[:, targ]

    # Stack the new X and Z arrays and create the new operator
    op_new = SparsePauliOp(np.hstack((x_arr, z_arr)), coeffs=coeffs)
    if isinstance(op, PauliSumOp):
        op_new = PauliSumOp(op_new)
    return op_new

cnot_pauli = apply_cnot_z2

def swap_z2(op: Union[PauliSumOp, SparsePauliOp], q1: int, q2: int
            ) -> Union[PauliSumOp, SparsePauliOp]:
    if isinstance(op, PauliSumOp):
        sparse_pauli_op = op.primitive.copy()
    elif isinstance(op, SparsePauliOp):
        sparse_pauli_op = op.copy()
    else:
        raise TypeError("Operator must be a PauliSumOp or SparsePauliOp")

    coeffs = sparse_pauli_op.coeffs.copy()
    x_arr = sparse_pauli_op.table.X.copy()
    z_arr = sparse_pauli_op.table.Z.copy()
    n_paulis, n_qubits = x_arr.shape

    x_tmp = x_arr[:, q1].copy()
    z_tmp = z_arr[:, q1].copy()
    x_arr[:, q1] = x_arr[:, q2].copy()
    z_arr[:, q1] = z_arr[:, q2].copy()
    x_arr[:, q2] = x_tmp
    z_arr[:, q2] = z_tmp

    # Stack the new X and Z arrays and create the new operator
    op_new = SparsePauliOp(np.hstack((x_arr, z_arr)), coeffs=coeffs)
    if isinstance(op, PauliSumOp):
        op_new = PauliSumOp(op_new)
    return op_new

swap_pauli = swap_z2

def taper(op: Union[PauliSumOp, SparsePauliOp],
          inds_tapered: Sequence[int],
          init_state: Optional[Sequence[int]] = None
          ) -> Union[PauliSumOp, SparsePauliOp]:
    """Tapers certain qubits off an operator.

    Note that the operator in string form follows Qiskit qubit order,
    but in vector (PauliTable) form follows normal qubit order.

    Args:
        pauli_sum_op: An operator on which qubits are to be tapered.
        inds_tapered: Indices of the tapered qubits.
        init_state: A sequence of 0 and 1 indicating the initial state on the
            tapered qubits. Default to 0 on all the tapered qubits.

    Returns:
        The new operator after certain qubits are tapered.
    """
    # TODO: Checking statements for inds_tapered and init_state
    if isinstance(op, PauliSumOp):
        sparse_pauli_op = op.primitive.copy()
    elif isinstance(op, SparsePauliOp):
        sparse_pauli_op = op.copy()
    else:
        raise TypeError("Operator must be a PauliSumOp or SparsePauliOp")

    coeffs = sparse_pauli_op.coeffs.copy()
    x_arr = sparse_pauli_op.table.X.copy()
    z_arr = sparse_pauli_op.table.Z.copy()
    n_paulis, n_qubits = x_arr.shape
    n_tapered = len(inds_tapered)
    if init_state is None:
        init_state = [0] * n_tapered

    for i in range(n_paulis):
        x = x_arr[i]
        z = z_arr[i]
        for j in range(n_tapered):
            # Z acts on |1>
            if z[j] == True and x[j] == False and init_state[j] == 1:
                coeffs[i] *= -1.0
            # Y acting on either |0> or |1>
            if z[j] == True and x[j] == True:
                if init_state[j] == 0:
                    coeffs[i] *= 1j
                elif init_state[j] == 1:
                    coeffs[i] *= -1j

    inds_kept = sorted(set(range(n_qubits)) - set(inds_tapered))
    x_arr = x_arr[:, inds_kept]
    z_arr = z_arr[:, inds_kept]

    op_new = SparsePauliOp(np.hstack((x_arr, z_arr)), coeffs=coeffs)
    if isinstance(op, PauliSumOp):
        op_new = PauliSumOp(op_new)
    return op_new

taper_pauli = taper

def transform_4q_pauli(
        op: Union[PauliSumOp, SparsePauliOp],
        init_state: List[int],
        tapered: bool = True
    ) -> Union[PauliSumOp, SparsePauliOp]:
    """Converts a four-qubit Hamiltonian to a two-qubit Hamiltonian,
    assuming the symmetry operators are ZIZI and IZIZ.

    Args:
        op: An four-qubit operator in PauliSumOp or SparsePauliOp form.
        spin: A string indicating the spin subspace of the transformed Hamiltonian.

    Returns:
        A two-qubit operator in the original form after symmetry reduction.
    """
    op_new = cnot_pauli(op, 2, 0)
    op_new = cnot_pauli(op_new, 3, 1)
    op_new = swap_pauli(op_new, 2, 3)
    if tapered:
        op_new = taper_pauli(op_new, [0, 1], init_state=init_state)
    if isinstance(op_new, PauliSumOp):
        op_new = op_new.reduce()
    return op_new

transform_4q = transform_4q_hamiltonian = transform_4q_pauli
