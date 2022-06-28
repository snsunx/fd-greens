"""
==========================================
Utilities (:mod:`fd_greens.general_utils`)
==========================================
"""

from functools import reduce
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

def get_bloch_vector(state: np.ndarray) -> np.ndarray:
    """Returns the Bloch vector of a state."""
    if len(state.shape) == 1:
        state = np.outer(state, state.conj())
    bloch_vector = np.zeros((3,))
    bloch_vector[0] = 2 * np.real(state[0][1])
    bloch_vector[1] = 2 * np.imag(state[1][0])
    bloch_vector[2] = np.real(state[0][0] - state[1][1])
    return bloch_vector

def get_fidelity(state1: np.ndarray, state2: np.ndarray, purify: bool = False) -> float:
    """Returns the fidelity between two states.
    
    Args:
        state1: The first state.
        state2: The second state.
        purify: Whether to purify density matrices.
    
    Returns:
        fidelity: The fidelity between two states.
    """
    def modify_state(state):
        if len(state.shape) == 1:
            if np.linalg.norm(state) != 1.0:
                state = state / np.linalg.norm(state)
        elif len(state.shape) == 2:
            state = project_density_matrix(state)
            if purify:
                state = purify_density_matrix(state)
        return state

    state1 = modify_state(state1)
    state2 = modify_state(state2)

    # TODO: Cirq fidelity function requires the density matrices to be PSD.
    # Need to implement a function without this constraint.
    fidelity = cirq.fidelity(state1, state2)
    return fidelity

def project_density_matrix(density_matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Projects a density matrix to its closest positive semi-definite equilvalent."""
    if normalize:
        density_matrix = density_matrix / np.trace(density_matrix)

    eigvals, eigvecs = np.linalg.eigh(density_matrix)
    if np.min(eigvals) >= 0:
        return density_matrix

    # Flip the eigenvalues to decreasing order.
    eigvals = np.flip(eigvals)
    eigvals_new = np.zeros_like(eigvals)

    final_index = len(eigvals)
    accumulator = 0.0

    while eigvals[final_index - 1] + accumulator / float(final_index) < 0:
        accumulator += eigvals[final_index - 1]
        final_index -= 1
        # print("acc / final idx =", accumulator / float(final_index))

    for i in range(final_index):
        eigvals_new[i] = eigvals[i] + accumulator / float(final_index)
    eigvals_new = np.flip(eigvals_new)

    density_matrix_new = eigvecs @ np.diag(eigvals_new) @ eigvecs.conj().T
    return density_matrix_new

def purify_density_matrix(density_matrix: np.ndarray, niter: int = 10) -> np.ndarray:
    """Purifies a density matrix using McWeeney purification.
    
    Args:
        density_matrix: The density matrix to be purified.
        niter: Number of iterations used in the purification algorithm.
        
    Returns:
        density_matrix: The density matrix after purification.
    """
    dim = density_matrix.shape[0]
    for _ in range(niter):
        density_matrix = density_matrix @ density_matrix @ (3 * np.eye(dim) - 2 * density_matrix)
        # print("<<< Density matrix eigvals = ", np.linalg.eigh(density_matrix)[0])
        density_matrix = density_matrix / np.trace(density_matrix)
        # print(">>> Density matrix eigvals = ", np.linalg.eigh(density_matrix)[0])
        # print("Trace = ", np.trace(density_matrix))
        # print("Purity =", np.trace(density_matrix @ density_matrix))
    return density_matrix

def state_tomography(result_array: np.ndarray) -> np.ndarray:
    """Quantum state tomography.
    
    Args:
        result_array: The array that contains normalized bitstring counts.
        
    Returns:
        density_matrix: The density matrix obtained from quantum state tomography.
    """
    # Create the basis and bistring labels.
    n_qubits = int(np.log(len(result_array)) / np.log(6))
    basis_labels = list(product('xyz', repeat=n_qubits))
    bitstring_labels = list(product('01', repeat=n_qubits))
    if REVERSE_QUBIT_ORDER:
        basis_labels = [x[::-1] for x in basis_labels]
    
    # Create a dictionary of all basis states.
    states = {
        ('x', '0'): np.array([1.0, 1.0]) / np.sqrt(2),
        ('x', '1'): np.array([1.0, -1.0]) / np.sqrt(2),
        ('y', '0'): np.array([1.0, 1.0j]) / np.sqrt(2),
        ('y', '1'): np.array([1.0, -1.0j]) / np.sqrt(2),
        ('z', '0'): np.array([1.0, 0.0]),
        ('z', '1'): np.array([0.0, 1.0]),
    }

    # Construct the basis matrix.
    basis_matrix = []
    for basis_label in basis_labels:
        for bitstring_label in bitstring_labels:
            basis_state = reduce(np.kron, [states[(basis_label[i], bitstring_label[i])] for i in range(n_qubits)])
            basis_vectorized = np.outer(basis_state, basis_state.conj()).reshape(-1)
            basis_matrix.append(basis_vectorized)
    basis_matrix = np.array(basis_matrix)

    density_matrix = np.linalg.lstsq(basis_matrix, result_array)[0]
    dim = int(np.sqrt(density_matrix.shape[0]))
    density_matrix = density_matrix.reshape(dim, dim, order='F')
    return density_matrix

two_qubit_state_tomography = state_tomography
