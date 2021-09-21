from typing import Union, Tuple, Sequence, Optional
from hamiltonians import MolecularHamiltonian
import numpy as np

from qiskit import *
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp


def apply_cnot_z2(pauli_sum_op: Union[PauliSumOp, SparsePauliOp], 
                  ctrl: int, 
                  targ: int) -> PauliSumOp:
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
    if isinstance(pauli_sum_op, PauliSumOp):
        sparse_pauli_op = pauli_sum_op.primitive.copy()
    elif isinstance(pauli_sum_op, SparsePauliOp):
        sparse_pauli_op = pauli_sum_op.copy()
    else:
        raise TypeError("Operator must be a PauliSumOp or a SparsePauliOp")

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
    sparse_pauli_op_new = SparsePauliOp(
        np.hstack((x_arr, z_arr)), coeffs=coeffs)
    pauli_sum_op_new = PauliSumOp(sparse_pauli_op_new)
    return pauli_sum_op_new

def taper(pauli_sum_op: Union[PauliSumOp, SparsePauliOp], 
          inds_tapered: Sequence[int],
          init_state: Optional[Sequence[int]] = None) -> PauliSumOp:
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
    if isinstance(pauli_sum_op, PauliSumOp):
        sparse_pauli_op = pauli_sum_op.primitive.copy()
    elif isinstance(pauli_sum_op, SparsePauliOp):
        sparse_pauli_op = pauli_sum_op.copy()
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
        z = z_arr[i]
        for j in range(n_tapered):
            if z[j] == True and init_state[j] == 1:
                coeffs[i] *= -1

    inds_kept = sorted(set(range(n_qubits)) - set(inds_tapered))
    x_arr = x_arr[:, inds_kept]
    z_arr = z_arr[:, inds_kept]
    
    sparse_pauli_op_new = SparsePauliOp(
        np.hstack((x_arr, z_arr)), coeffs=coeffs)
    pauli_sum_op_new = PauliSumOp(sparse_pauli_op_new)
    return pauli_sum_op_new

def transform_4q_hamiltonian(
        pauli_sum_op: PauliSumOp, spin: str = ''
    ) -> PauliSumOp:
    """Converts a four-qubit Hamiltonian to a two-qubit Hamiltonian, 
    assuming the symmetry operators are ZIZI and IZIZ.
    
    Args:
        pauli_sum_op: A Hamiltonian in PauliSumOp form.
        spin: A string indicating the spin subspace of the transformed Hamiltonian.

    Returns:
        A two-qubit Hamiltonian after symmetry reduction.
    """
    assert spin in ['', 'up', 'down']

    if spin == '':
        init_state = [1, 1]
    elif spin == 'up':
        init_state = [0, 1]
    elif spin == 'down':
        init_state = [1, 0]
    else:
        raise ValueError("Spin must be one of '', 'up' or 'down'")
    
    pauli_sum_op_new = apply_cnot_z2(apply_cnot_z2(pauli_sum_op, 2, 0), 3, 1)
    # print("PauliSumOp after CNOT is applied\n", pauli_sum_op_new)
    pauli_sum_op_new = taper(pauli_sum_op_new, [0, 1], init_state=init_state)
    # print("PauliSumOp after qubits are tapered\n", pauli_sum_op_new)
    return pauli_sum_op_new

