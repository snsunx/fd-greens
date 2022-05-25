"""
======================================
Utilities (:mod:`fd_greens.utilities`)
======================================
"""

from collections import Counter
from itertools import product
import math
from typing import Optional, Callable, Union

import numpy as np
import cirq

from .parameters import REVERSE_QUBIT_ORDER

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

def unitary_equal(unitary1: Union[cirq.Circuit, np.ndarray], 
                  unitary2: Union[cirq.Circuit, np.ndarray], 
                  initial_state_0: bool = True
                  ) -> bool:
    """Checks if two unitaries are equal up to a phase factor.
    
    Args:
        unitary1: The first unitary in circuit or array form.
        unitary2: The second unitary in circuit or array form.
        initial_state_0: Whether to assume the initial state to be the all 0 state.

    Returns:
        is_equal: Whether the two unitaries are equal up to a phase.
    """
    # It the unitaries are in circuit form, convert them to unitaries.
    if isinstance(unitary1, cirq.Circuit):
        unitary1 = cirq.unitary(unitary1)
    if isinstance(unitary2, cirq.Circuit):
        unitary2 = cirq.unitary(unitary2)
    
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

def histogram_to_array(
    histogram: dict,
    n_qubits: Optional[int] = None,
    base: int = 2,
    normalize: bool = True
) -> np.ndarray:
    """Converts a bitstring histogram to a numpy array.

    Keys of the histogram can either be strings, such as '01', or tuples of integers, such as (0, 1).
    
    Args:
        histogram: The histogram from simulators or experiments.
        n_qubits: Number of qubits.
        base: The base in integer conversion. Defaults to 2 for qubits.
        
    Returns:
        array: The bitstring array from the histogram.
    """
    if n_qubits is None:
        n_qubits = len(list(histogram.keys())[0])
        # indices = []
        # for key in histogram.keys():
        #     index = int(''.join([str(i) for i in key]), base)
        #     indices.append(index + 1)
        # n_qubits = math.ceil(math.log2(max(indices)))
    
    array = np.zeros((base ** n_qubits,))
    for key, value in histogram.items():
        if isinstance(key, str):
            index = int(key, base)
        else:
            index = int(''.join([str(i) for i in key]), base)
        array[index] = value

    if normalize:
        array = array / np.sum(array)
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

def two_qubit_state_tomography(result_array: np.ndarray) -> np.ndarray:
    """Two-qubit quantum state tomography.
    
    Args:
        result_array: The array that contains normalized bitstring counts.
        
    Returns:
        density_matrix: The density matrix obtained from state tomography.
    """
    basis_matrix = []

    if REVERSE_QUBIT_ORDER:
        bases = [(f'{x[1]}{x[2]}', f'{x[0]}{x[3]}') for x in product('xyz', 'xyz', '01', '01')]
    else:
        bases = [(f'{x[0]}{x[2]}', f'{x[1]}{x[3]}') for x in product('xyz', 'xyz', '01', '01')]
    
    states = {
        "x0": np.array([1.0, 1.0]) / np.sqrt(2),
        "x1": np.array([1.0, -1.0]) / np.sqrt(2),
        "y0": np.array([1.0, 1.0j]) / np.sqrt(2),
        "y1": np.array([1.0, -1.0j]) / np.sqrt(2),
        "z0": np.array([1.0, 0.0]),
        "z1": np.array([0.0, 1.0]),
    }

    for basis in bases:
        basis_state = np.kron(states[basis[0]], states[basis[1]])
        basis_vectorized = np.outer(basis_state, basis_state.conj()).reshape(-1)
        basis_matrix.append(basis_vectorized)
    basis_matrix = np.array(basis_matrix)

    density_matrix = np.linalg.lstsq(basis_matrix, result_array)[0]
    dim = int(np.sqrt(density_matrix.shape[0]))
    density_matrix = density_matrix.reshape(dim, dim, order='F')
    return density_matrix