"""Z2 symmetry utility functions"""

from typing import Union, Sequence, Optional, List
import numpy as np

from qiskit import *
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

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

def transform_4q_hamiltonian(
        op: Union[PauliSumOp, SparsePauliOp],
        init_state: List[int],
        tapered: bool = True
    ) -> Union[PauliSumOp, SparsePauliOp]:
    """Converts a four-qubit Hamiltonian to a two-qubit Hamiltonian,
    assuming the symmetry operators are ZIZI and IZIZ.

    Args:
        pauli_sum_op: A Hamiltonian in PauliSumOp form.
        spin: A string indicating the spin subspace of the transformed Hamiltonian.

    Returns:
        A two-qubit Hamiltonian after symmetry reduction.
    """
    op_new = apply_cnot_z2(apply_cnot_z2(op, 2, 0), 3, 1)
    # print(op_new.table.to_labels(), op_new.coeffs)
    op_new = swap_z2(op_new, 2, 3)
    if tapered:
        op_new = taper(op_new, [0, 1], init_state=init_state)
    # op_new = swap_z2(op_new, 0, 1)
    return op_new

transform_4q = transform_4q_hamiltonian