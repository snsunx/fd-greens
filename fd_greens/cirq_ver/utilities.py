"""
======================================
Utilities (:mod:`fd_greens.utilities`)
======================================
"""

from typing import Optional, Callable
from copy import deepcopy
from collections import Counter
import math

import numpy as np
import cirq


def reverse_qubit_order(arr: np.ndarray) -> np.ndarray:
    """Reverses qubit order in a 1D or 2D array.

    This is because Qiskit uses big endian order and most other packages use small endian order.

    Args:
        arr: The 1D or 2D array on which the qubit order is to be reversed.

    Returns:
       arr: The 1D or 2D array after qubit order is reversed.
    """
    if len(arr.shape) == 1:
        # Extract the dimension and number of qubits.
        dim = arr.shape[0]
        n_qubits = int(np.log2(dim))

        # Reshape to a length 2^N array, transpose the indices and then reshape back.
        arr = arr.reshape(*([2] * n_qubits))
        arr = arr.transpose(*range(n_qubits)[::-1])
        arr = arr.reshape(dim)
    elif len(arr.shape) == 2:
        # Extract the dimension and number of qubits.
        assert arr.shape[0] == arr.shape[1]
        dim = arr.shape[0]
        n_qubits = int(np.log2(dim))

        # Reshape to a dim 2^N by 2^N array, transpose the indices and then reshape back.
        from itertools import chain

        arr = arr.reshape(*([2] * 2 * n_qubits))
        arr = arr.transpose(
            *chain(range(n_qubits)[::-1], range(n_qubits, 2 * n_qubits)[::-1])
        )
        arr = arr.reshape(dim, dim)
    else:
        raise NotImplementedError(
            "Reversing qubit order of array with more"
            "than two dimensions is not implemented."
        )
    return arr

def circuit_equal(
    circuit1: cirq.Circuit, circuit2: cirq.Circuit, init_state_0: bool = True
) -> bool:
    """Checks if two circuits are equivalent.

    The two circuits are equivalent either when the unitaries are equal up to a phase or 
    when the statevectors with the all 0 initial state are equal up to a phase.
    
    Args:
        circuit1: The first cicuit.
        circuit2: The second circuit.
        init_state_0: Whether to assume the initial state is the all 0 state. Default to True.
        
    Returns:
        is_equal: Whether the two circuits are equivalent.
    """
    # If the circuits are in instruction tuple form.
    unitary1 = cirq.unitary(circuit1)
    unitary2 = cirq.unitary(circuit2)
    assert unitary1.shape == unitary2.shape
    is_equal = unitary_equal(unitary1, unitary2, init_state_0=init_state_0)
    return is_equal


def unitary_equal(
    uni1: np.ndarray, uni2: np.ndarray, init_state_0: bool = True
) -> bool:
    """Checks if two unitaries are equal up to a phase factor.
    
    Args:
        uni1: The first unitary.
        uni2: The second unitary.
        init_state_0: If set to True, only consider equality on the first column 
            corresponding to the all 0 state.

    Returns:
        is_equal: Whether the two unitaries are equal up to a phase.
    """
    if init_state_0:
        # Obtain the phase from the first column, since only comparing
        # actions on the all 0 state.
        vec1 = deepcopy(uni1[:, 0])
        vec2 = deepcopy(uni2[:, 0])
        ind = np.argmax(np.abs(vec1))
        if abs(vec2[ind]) == 0:
            return False
        phase1 = vec1[ind] / abs(vec1[ind])
        phase2 = vec2[ind] / abs(vec2[ind])

        # Compare the two vectors after dividing by their respective phases.
        vec1 /= phase1
        vec2 /= phase2
        is_equal = np.allclose(vec1, vec2)
        if not is_equal:
            print("The difference is ", np.linalg.norm(vec1 - vec2))
    else:
        # Obtain the phase from the first row.
        ind = np.argmax(np.abs(uni1[0]))
        if abs(uni2[0, ind]) == 0:
            return False
        phase1 = uni1[0, ind] / abs(uni1[0, ind])
        phase2 = uni2[0, ind] / abs(uni2[0, ind])

        # Compare the two unitaries after dividing by their respective phases.
        uni1_copy = deepcopy(uni1)
        uni2_copy = deepcopy(uni2)
        uni1_copy /= phase1
        uni2_copy /= phase2
        is_equal = np.allclose(uni1_copy, uni2_copy)

    return is_equal

def histogram_to_array(histogram: Counter, n_qubits: Optional[int] = None) -> np.ndarray:
    """Converts a Cirq histogram to a numpy array."""
    if n_qubits is None:
        keys = histogram.keys()
        inds = []
        for key in keys:
            ind = int(''.join([str(i) for i in key]), 2)
            inds.append(ind + 1)
        n_qubits = math.ceil(math.log2(max(inds)))
            
    arr = np.zeros((2 ** n_qubits,))
    for key, val in histogram.items():
        ind = int(''.join([str(i) for i in key]), 2)
        arr[ind] = val
    return arr

def get_gate_counts(circuit: cirq.Circuit, criterion: Callable[[cirq.OP_TREE], bool] = lambda op: True) -> int:
    """Returns the count of gates satisfying a certain criterion."""
    count = 0
    for op in circuit.all_operations():
        if criterion(op):
            count += 1
    return count