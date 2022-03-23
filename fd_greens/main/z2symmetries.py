"""Z2 symmetry transform functions."""

from typing import Union, Sequence, Optional
import numpy as np

# from qiskit import *
from qiskit.quantum_info import PauliTable, SparsePauliOp
from qiskit.opflow import PauliSumOp

from .qubit_indices import QubitIndices

PauliOperator = Union[PauliSumOp, SparsePauliOp]


def cnot_pauli(pauli_op: PauliOperator, ctrl: int, targ: int) -> PauliOperator:
    """Applies a CNOT operation to the Z2 representation of a Pauli operator.

    Note that the operator in string form follows Qiskit qubit order,
    but in vector (PauliTable) form follows normal qubit order. The returned 
    type is the same as the input type, which can be either a PauliSumOp or
    a SparsePauliOp.

    Args:
        pauli_op: A Pauli operator on which a CNOT is to be applied.
        ctrl: An integer indicating the index of the control qubit.
        targ: An integer indicating the index of the target qubit.

    Returns:
        The new Pauli operator after a CNOT is applied.
    """
    if isinstance(pauli_op, PauliSumOp):
        sparse_pauli_op = pauli_op.primitive.copy()
    elif isinstance(pauli_op, SparsePauliOp):
        sparse_pauli_op = pauli_op.copy()
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

    # Apply binary addition forward on the X part, backward on the Z part.
    x_arr[:, targ] ^= x_arr[:, ctrl]
    z_arr[:, ctrl] ^= z_arr[:, targ]

    # Stack the new X and Z arrays and create the new operator.
    pauli_table = PauliTable(np.hstack((x_arr, z_arr)))
    pauli_op_new = SparsePauliOp(pauli_table, coeffs=coeffs)
    if isinstance(pauli_op, PauliSumOp):
        pauli_op_new = PauliSumOp(pauli_op_new)
    return pauli_op_new


def swap_pauli(pauli_op: PauliOperator, q1: int, q2: int) -> PauliOperator:
    """Applies a SWAP operation to the Z2 representation of a Pauli operator.

    Note that the operator in string form follows Qiskit qubit order,
    but in vector (PauliTable) form follows normal qubit order. The returned type
    is the same as the input type, which can be either a PauliSumOp or
    a SparsePauliOp.

    Args:
        pauli_op: A Pauli operator on which a SWAP is to be applied.
        q1: An integer indicating the first qubit index.
        q2: An integer indicating the second qubit index.

    Returns:
        The new Pauli operator after a SWAP is applied.
    """
    if isinstance(pauli_op, PauliSumOp):
        sparse_pauli_op = pauli_op.primitive.copy()
    elif isinstance(pauli_op, SparsePauliOp):
        sparse_pauli_op = pauli_op.copy()
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

    # Stack the new X and Z arrays and create the new operator.
    pauli_table = PauliTable(np.hstack((x_arr, z_arr)))
    pauli_op_new = SparsePauliOp(pauli_table, coeffs=coeffs)
    if isinstance(pauli_op, PauliSumOp):
        pauli_op_new = PauliSumOp(pauli_op_new)
    return pauli_op_new


def taper_pauli(
    pauli_op: PauliOperator,
    inds_tapered: Sequence[int],
    init_state: Optional[Sequence[int]] = None,
) -> PauliOperator:
    """Tapers certain qubits off a Pauli operator.

    Note that the operator in string form follows Qiskit qubit order,
    but in vector (PauliTable) form follows normal qubit order. THe returned type
    is the same as the input type, which can be either a PauliSumOp or
    a SparsePauliOp.

    Args:
        pauli_op: A Pauli operator on which certain qubits are to be tapered.
        inds_tapered: Indices of the tapered qubits.
        init_state: A sequence of 0 and 1 indicating the initial state on the
            tapered qubits. Default to 0 on all the tapered qubits.

    Returns:
        The new operator after certain qubits are tapered.
    """
    # TODO: Checking statements for inds_tapered and init_state
    if isinstance(pauli_op, PauliSumOp):
        sparse_pauli_op = pauli_op.primitive.copy()
    elif isinstance(pauli_op, SparsePauliOp):
        sparse_pauli_op = pauli_op.copy()
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

    pauli_table = PauliTable(np.hstack((x_arr, z_arr)))
    pauli_op_new = SparsePauliOp(pauli_table, coeffs=coeffs)
    if isinstance(pauli_op, PauliSumOp):
        pauli_op_new = PauliSumOp(pauli_op_new)
    return pauli_op_new


def transform_4q_pauli(
    pauli_op: PauliOperator,
    swap: bool = True,
    taper: bool = True,
    init_state: Optional[Sequence[int]] = None,
) -> PauliOperator:
    """Transforms a 4q pauli operator (or sum of them) to a 2q pauli operators (or sum of them).

    THe symmetries are assumed to be ZIZI and IZIZ. The operations applied are 
    CNOT(2, 0), CNOT(3, 1) and SWAP(2, 3), followed by optionally tapering off q0 and q1.

    Args:
        pauli_op: A four-qubit Pauli operator to be transformed.
        swap: Whether to swap q2 and q3.
        taper: Whether to taper q0 and q1.
        init_state: The initial state on the first two qubits if tapered.

    Returns:
        A two-qubit Pauli operator after transformation if tapered, otherwise a four-qubit
        Pauli operator.
    """
    n_qubits = 4  # 4 is hardcoded
    pauli_op_new = pauli_op.copy()
    if taper:
        assert init_state is not None and len(init_state) == 2

    for i in range(n_qubits // 2):
        pauli_op_new = cnot_pauli(pauli_op_new, i + n_qubits // 2, i)
    if swap:
        pauli_op_new = swap_pauli(pauli_op_new, 2, 3)
    if taper:
        pauli_op_new = taper_pauli(pauli_op_new, [0, 1], init_state=init_state)

    if isinstance(pauli_op_new, PauliSumOp):
        pauli_op_new = pauli_op_new.reduce()
    return pauli_op_new


transform_dict = {"cnot": cnot_pauli, "swap": swap_pauli, "taper": taper_pauli}


def cnot_indices(qubit_inds: QubitIndices, ctrl: int, targ: int) -> QubitIndices:
    """Applies a CNOT operation to qubit indices.
    
    Args:
        qubit_inds: The input QubitIndices object.
        ctrl: Index of the control qubit.
        targ: Index of the target qubit.

    Returns:
        The QubitIndices object after CNOT is applied.
    """
    data_list = qubit_inds.list_form
    data_list_new = []
    for q_ind in data_list:
        q_ind_new = q_ind.copy()
        q_ind_new[targ] = (q_ind[ctrl] + q_ind[targ]) % 2
        data_list_new.append(q_ind_new)
    qubit_inds_new = QubitIndices(data_list_new)
    return qubit_inds_new


def swap_indices(qubit_inds: QubitIndices, q1: int, q2: int) -> QubitIndices:
    """Applies a SWAP operation to qubit indices.
    
    Args:
        qubit_inds: The input QubitIndices object.
        q1: Index of the first qubit.
        q2: Index of the second qubit.

    Returns:
        The QubitIndices object after SWAP is applied.
    """
    data_list = qubit_inds.list_form
    data_list_new = []
    for q_ind in data_list:
        q_ind_new = q_ind.copy()
        q_ind_new[q2] = q_ind[q1]
        q_ind_new[q1] = q_ind[q2]
        data_list_new.append(q_ind_new)
    qubit_inds_new = QubitIndices(data_list_new)
    return qubit_inds_new


def taper_indices(
    qubit_inds: QubitIndices, inds_tapered: Sequence[int]
) -> QubitIndices:
    """Tapers certain qubits off in a QubitIndices object.
    
    Args:
        qubit_inds: The input QubitIndices object.
        inds_tapered: The tapered qubit indices.
    
    Returns:
        The QubitIndices object after tapering.
    """
    qubit_inds_list = qubit_inds.list_form
    qubit_inds_list_new = []
    for q_ind in qubit_inds_list:
        q_ind_new = [q for i, q in enumerate(q_ind) if i not in inds_tapered]
        qubit_inds_list_new.append(q_ind_new)

    qubit_inds_new = QubitIndices(qubit_inds_list_new)
    return qubit_inds_new


def transform_4q_indices(
    qubit_inds: QubitIndices, swap: bool = True, tapered: bool = True
) -> QubitIndices:
    """Transforms 4q qubit indices to 2q qubit indices.
    
    Args:
        qubit_inds: The qubit indices.
        swap: Whether to swap the last two qubits.
        tapered: Whether to taper off the first two qubits.

    Returns:
        The qubit indices after the transformation.
    """
    qubit_inds_new = cnot_indices(qubit_inds, 2, 0)
    qubit_inds_new = cnot_indices(qubit_inds_new, 3, 1)
    if swap:
        qubit_inds_new = swap_indices(qubit_inds_new, 2, 3)
    if tapered:
        qubit_inds_new = taper_indices(qubit_inds_new, [0, 1])
    return qubit_inds_new
