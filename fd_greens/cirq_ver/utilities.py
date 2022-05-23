"""
======================================
Utilities (:mod:`fd_greens.utilities`)
======================================
"""

from collections import Counter
from itertools import product
import math
from typing import Optional, Callable

import numpy as np
import cirq

def reverse_qubit_order(array: np.ndarray) -> np.ndarray:
    """Reverses qubit order in a 1D or 2D array.
    
    Args:
        array: The array on which qubit order is to be reversed.
        
    Returns:
        array_new: The new array with qubit order reversed.
    """
    assert len(array.shape) in [1, 2]
    n_qubits = int(np.log2(array.shape[0]))
    indices = [''.join(x) for x in product('01', repeat=n_qubits)]
    indices = [int(x[::-1], 2) for x in indices]

    if len(array.shape) == 1:
        array_new = array[indices]
    elif len(array.shape) == 2:
        array_new = array[indices][:, indices]
    return array_new

# TODO: Maybe can deprecate this function and combine it with unitary_equal.
def circuit_equal(circuit1: cirq.Circuit, circuit2: cirq.Circuit, initial_state_0: bool = True) -> bool:
    """Checks if two circuits are equivalent.

    The two circuits are equivalent either when the unitaries are equal up to a phase or 
    when the statevectors with all 0 initial state are equal up to a phase.
    
    Args:
        circuit1: The first cicuit.
        circuit2: The second circuit.
        initial_state_0: Whether to assume the initial state is the all 0 state.
        
    Returns:
        is_equal: Whether the two circuits are equivalent.
    """
    unitary1 = cirq.unitary(circuit1)
    unitary2 = cirq.unitary(circuit2)
    assert unitary1.shape == unitary2.shape
    is_equal = unitary_equal(unitary1, unitary2, initial_state_0=initial_state_0)
    return is_equal


def unitary_equal(unitary1: np.ndarray, unitary2: np.ndarray, initial_state_0: bool = True) -> bool:
    """Checks if two unitaries are equal up to a phase factor.
    
    Args:
        unitary1: The first unitary.
        unitary2: The second unitary.
        initial_state_0: Whether to assume the initial state to be the all 0 state.

    Returns:
        is_equal: Whether the two unitaries are equal up to a phase.
    """

    # Find the index for the phase factor in the first column, since the (0, 0) element might be 0.
    index = np.argmax(np.abs(unitary1[:, 0]))
    if abs(unitary2[index, 0]) == 0:
        return False

    phase1 = unitary1[index, 0] / abs(unitary1[index, 0])
    phase2 = unitary2[index, 0] / abs(unitary2[index, 0])

    if initial_state_0:
        is_equal = np.allclose(unitary1[:, 0] / phase1, unitary2[:, 0] / phase2)
    else:
        is_equal = np.allclose(unitary1 / phase1, unitary2 / phase2)

    return is_equal

def histogram_to_array(histogram: Counter, n_qubits: Optional[int] = None) -> np.ndarray:
    """Converts a Cirq histogram to a numpy array.
    
    Args:
        histogram: The histogram from Cirq simulator runs.
        n_qubits: Number of qubits.
        
    Returns:
        array: The array form of the histogram.
    """
    if n_qubits is None:
        indices = []
        for key in histogram.keys():
            index = int(''.join([str(i) for i in key]), 2)
            indices.append(index + 1)
        n_qubits = math.ceil(math.log2(max(indices)))
    
    array = np.zeros((2 ** n_qubits,))
    for key, value in histogram.items():
        index = int(''.join([str(i) for i in key]), 2)
        array[index] = value
    return array

def get_gate_counts(circuit: cirq.Circuit, criterion: Callable[[cirq.OP_TREE], bool] = lambda op: True) -> int:
    """Returns the count of gates satisfying a certain criterion.
    
    Args:
        circuit: The circuit on which to return gate counts.
        criterion: The criterion of gates to be counted.
        
    Returns:
        count: Number of gates satisfying a certain criterion.
    """
    count = 0
    for op in circuit.all_operations():
        if criterion(op):
            count += 1
    return count