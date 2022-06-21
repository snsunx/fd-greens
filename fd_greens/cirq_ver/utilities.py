"""
======================================
Utilities (:mod:`fd_greens.utilities`)
======================================
"""

import re
from itertools import product
from typing import Optional, Callable, Union, List

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
    
    Args:
        histogram: The histogram from simulators or experiments. Keys of the histogram 
            can either be strings, such as ``'01'``, or tuples of integers, such as ``(0, 1)``.
        n_qubits: Number of qubits.
        base: The base in integer conversion. Defaults to 2 for qubits.
        normalize: Whether to normalize the sum of the bitstring counts to 1.
        
    Returns:
        array: The bitstring array from the histogram.
    """
    if n_qubits is None:
        n_qubits = len(list(histogram.keys())[0])
    
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

def get_gate_counts(circuit: cirq.Circuit, *,
					criterion: Callable[[cirq.OP_TREE], bool] = lambda op: True,
                    num_qubits: Optional[int] = None
				   ) -> int:
    """Returns the count of gates satisfying a certain criterion.

    Only one of ``num_qubits`` and ``criterion`` should be given. If ``num_qubits`` is given,
    count the number of gates with ``num_qubits`` number of qubits. Otherwise counting is performed
    with ``criterion``. Defaults to counting all non-Z-rotation gates.
    
    Args:
        circuit: The circuit on which to return gate counts.
        criterion: The criterion of gates to be counted.
        num_qubits: Number of qubits in gates to be counted.
        
    Returns:
        count: Number of gates satisfying a certain criterion.
    """
    if num_qubits is not None:
        # Counting gates with number of qubits.
        criterion = lambda op: op.gate.num_qubits() == num_qubits

    if criterion is None:
        # Only counting non-Z-rotation gates.
        criterion = lambda op: not isinstance(op.gate, cirq.ZPowGate)

    count = 0
    for op in circuit.all_operations():
        if criterion(op):
            count += 1
    return count

def split_circuit_across_moment(circuit: cirq.Circuit, moment_split: cirq.Moment) -> List[cirq.Circuit]:
    """Splits a circuit across a certain moment.
    
    Args:
        circuit: The circuit to be split.
        moment_split: A moment used to split the circuit.
        
    Returns:
        circuits: A list of circuits split at moments.
    """
    unitary_split = cirq.unitary(moment_split)

    circuits = []
    moments = []
    for moment in circuit:
        unitary = cirq.unitary(moment)
        # TODO: The criterion should be more specific than comparing unitaries.
        if unitary.shape == unitary_split.shape and np.allclose(unitary, unitary_split):
            circuits.append(cirq.Circuit(moments))
            moments = []
        else:
            moments.append(moment)

    # Append the circuit built from the last moments to circuits.
    circuits.append(cirq.Circuit(moments))
    return circuits

def get_non_z_locations(circuit: Union[cirq.Circuit, List[List[str]]]) -> List[int]:
    """Returns the non-Z locations of a circuit.
    
    Args:
        circuit: A circuit in either Cirq circuit form or Qtrl strings form.
        
    Returns:
        locations: A list of integers indicating the non-Z locations.
    """
    locations = []

    for i, moment in enumerate(circuit):
        if isinstance(moment, cirq.Moment):
            is_z_gate = [isinstance(op.gate, cirq.ZPowGate) for op in moment]
        else:
            is_z_gate = [re.findall('Q\d/Z.+', s) != [] or re.findall('CP.+', s) != [] for s in moment]
         
        if not all(is_z_gate):
            locations.append(i)
    
    return locations

def get_circuit_depth(circuit: Union[cirq.Circuit, List[List[str]]]) -> int:
    """Returns the circuit depth by excluding Z and CP gates.
    
    Args:
        circuit: A circuit in either Cirq circuit form or Qtrl strings form.
    
    Returns:
        depth: The depth of the circuit.
    """
    depth = len(get_non_z_locations(circuit)) 
    return depth

def get_n_simultaneous_cz(qtrl_strings: List[List[str]]) -> int:
    """Returns the number of simultaneous CZ gates."""
    n_simul_cz = 0
    for i in range(len(qtrl_strings) - 1):
        if qtrl_strings[i] == ['CZ/C7T6'] and qtrl_strings[i + 1] == ['CZ/C5T4']:
            n_simul_cz += 1
        if qtrl_strings[i] == ['CZ/C5T4'] and qtrl_strings[i + 1] == ['CZ/C7T6']:
            n_simul_cz += 1
    return n_simul_cz

def split_simultaneous_cz(circuit: Union[cirq.Circuit, List[List[str]]]) -> None:
    """Splits simultaneous CZ/CS/CSDs onto different moments.
    
    Args:
        circuit: The circuit in either Cirq circuit form or Qtrl strings form.
    """
    circuit_new = []
    for moment in circuit:
        if isinstance(moment, cirq.Moment):
            is_cz_gate = [isinstance(op.gate, cirq.CZPowGate) for op in moment]
        else:
            is_cz_gate = [re.findall('C(Z|S|SD)/.+', s) for s in moment]

        if all(is_cz_gate) and len(is_cz_gate) > 1:
            for op in moment:
                circuit_new.append([op])
        else:
            circuit_new.append(moment)
        
    if isinstance(circuit, cirq.Circuit):
        circuit_new = cirq.Circuit([cirq.Moment(l) for l in circuit_new])

    return circuit_new

def combine_simultaneous_cz(circuit: Union[cirq.Circuit, List[List[str]]]) -> None:
    """Combines simultaneous CZ/CS/CSDs into the same moment.
    
    Args:
        circuit: The circuit in either Cirq circuit form or Qtrl strings form.
    """
    for i in range(len(circuit) - 1):
        if isinstance(circuit[i], cirq.Moment):
            is_simul_cz = (
                len(circuit[i]) == 1 and list(circuit[i])[0].gate == cirq.CZ and
                len(circuit[i + 1]) == 1 and list(circuit[i + 1])[0].gate == cirq.CZ and
                len(circuit[i].qubits.intersection(circuit[i + 1].qubits)) == 0
            )
            if is_simul_cz:
                circuit[i] = circuit[i] + circuit[i + 1]
                circuit[i + 1] = cirq.Moment()
        else:
            is_simul_cz = ((circuit[i], circuit[i + 1]) == (['CZ/C7T6'], ['CZ/C5T4']) or
                (circuit[i], circuit[i + 1]) ==  (['CZ/C5T4'], ['CZ/C7T6']))

            if is_simul_cz:
                circuit[i] = ['CZ/C7T6', 'CZ/C5T4']
                circuit[i + 1] = None

    if isinstance(circuit, cirq.Circuit):
        cirq.DropEmptyMoments().optimize_circuit(circuit)
    else:
        circuit = [moment for moment in circuit if moment != None]
    return circuit

def print_circuit_statistics(circuit: cirq.Circuit) -> None:
    """Prints out circuit statistics.
    
    Args:
        circuit: The circuit on which statistics are to be printed.
    """
    depth = get_circuit_depth(circuit)
    n_2q = get_gate_counts(circuit, num_qubits=2)
    n_3q = get_gate_counts(circuit, num_qubits=3)
    print(f"Circuit depth = {depth}")
    print(f"Number of two-qubit gates = {n_2q}")
    print(f"Number of three-qubit gates = {n_3q}")

    # Only print out multi-qubit gate counts on 4-qubit circuits.
    qubits = sorted(circuit.all_qubits())
    if len(qubits) == 4:
        for i in range(len(qubits) - 1):
            qubit_pair = (qubits[i], qubits[i + 1])
            n_cs = get_gate_counts(
                circuit, 
                criterion=lambda op: op.gate == cirq.CZPowGate(exponent=0.5) and set(op.qubits) == set(qubit_pair))
            # t_cs = 150 + 200 * (i % 2 == 0)
            n_csd = get_gate_counts(
                circuit,
                criterion=lambda op: op.gate == cirq.CZPowGate(exponent=-0.5) and set(op.qubits) == set(qubit_pair))
            # t_csd = 150 + 200 * (i % 2 == 1)
            n_cz = get_gate_counts(
                circuit, 
                criterion=lambda op: op.gate == cirq.CZPowGate(exponent=1.0) and set(op.qubits) == set(qubit_pair))
            # t_cz = 200
            print(f"Number of CS, CSD, CZ gates on qubits ({i}, {i + 1}) = {n_cs}, {n_csd}, {n_cz}")

# TODO: Write a general implementation of state tomography.
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
