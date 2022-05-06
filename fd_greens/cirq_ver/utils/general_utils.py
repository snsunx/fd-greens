"""
========================================================
General Utilities (:mod:`fd_greens.utils.general_utils`)
========================================================
"""

import os
import h5py
from typing import (
    Optional,
    Union,
    Iterable,
    List,
    Tuple,
    Sequence,
    Mapping,
    Any,
    Callable,
)
from itertools import product
from copy import deepcopy

import numpy as np
import cirq
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Aer,
    execute,
    IBMQ,
)
from qiskit.utils import QuantumInstance
from qiskit.circuit import Instruction, Qubit, Clbit
from qiskit.opflow import PauliSumOp
from qiskit.result import Result, Counts
from qiskit.quantum_info import OneQubitEulerDecomposer

from .circuit_utils import remove_instructions, create_circuit_from_inst_tups

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]
QuantumCircuitLike = Union[QuantumCircuit, Iterable[InstructionTuple]]
AnsatzFunction = Callable[[Sequence[float]], QuantumCircuit]


def get_tomography_labels(n_qubits: int) -> List[str]:
    """Returns the tomography labels on a certain number of qubits.
    
    Args:
        n_qubits: The number of qubits to be tomographed.
        
    Returns:
        tomo_labels: The tomography labels.
    """
    tomo_labels = ["".join(x) for x in product("xyz", repeat=n_qubits)]
    return tomo_labels


def get_statevector(circ_like: QuantumCircuitLike, reverse: bool = False) -> np.ndarray:
    """Returns the statevector of a circuit.

    Args:
        circ_like: The circuit or instruction tuples on which the state is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        statevector: The statevector array of the circuit.
    """
    if isinstance(circ_like, QuantumCircuit):
        circ = circ_like
    else:  # instruction tuples
        circ = create_circuit_from_inst_tups(circ_like)

    backend = Aer.get_backend("statevector_simulator")
    job = execute(circ, backend)
    result = job.result()
    statevector = result.get_statevector()
    if reverse:
        statevector = reverse_qubit_order(statevector)
    return statevector


def get_unitary(circ_like: QuantumCircuitLike, reverse: bool = False) -> np.ndarray:
    """Returns the unitary of a circuit.

    Args:
        circ_like: The circuit or instruction tuples on which the unitary is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        unitary: The unitary array of the circuit.
    """
    circ_like = remove_instructions(circ_like, ["barrier", "measure"])
    if isinstance(circ_like, QuantumCircuit):
        circ = circ_like
    else:  # instruction tuples
        circ = create_circuit_from_inst_tups(circ_like)

    backend = Aer.get_backend("unitary_simulator")
    result = execute(circ, backend).result()
    unitary = result.get_unitary()
    if reverse:
        unitary = reverse_qubit_order(unitary)
    return unitary


def get_overlap(state1: np.ndarray, state2: np.ndarray) -> float:
    """Returns the overlap of two states in either statevector or density matrix form.
    
    Args:
        state1: A 1D or 2D numpy array corresponding to the first state.
        state2: A 1D or 2D numpy array corresponding to the second state.

    Returns:
        overlap: The overlap between the two states.
    """
    # If both state1 and state2 are statevectors, return |<state1|state2>|^2.
    if len(state1.shape) == 1 and len(state2.shape) == 1:
        overlap = abs(state1.conj() @ state2) ** 2

    # If state1 is a statevector and state2 is a density matrix,
    # return Re(<state1|state2|state1>).
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        overlap = (state1.conj() @ state2 @ state1).real

    # If state1 is a density matrix and state2 is a statevector,
    # return Re(<state2|state1|state2>).
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        overlap = (state2.conj() @ state1 @ state2).real

    # If both state1 and state2 are density matries, return Re(<state1|state2>).
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        overlap = np.trace(state1.conj().T @ state2).real

    return overlap


def reverse_qubit_order(arr: np.ndarray) -> np.ndarray:
    """Reverses qubit order in a 1D or 2D array.

    This is because Qiskit uses big endian order and 

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


def counts_dict_to_arr(
    counts: Union[Counts, dict], n_qubits: Optional[int] = None
) -> np.ndarray:
    """Converts bitstring counts from ``qiskit.result.Counts`` form to ``np.ndarray`` form.
    
    Args:
        counts: Bitstring counts either in ``Counts`` form or in ``dict`` form.
        n_qubits: The number of qubits. If not passed in, will be extracted 
            from the bitstring counts.

    Returns:
        arr: A numpy array corresponding to the bitstring counts.
    """
    # Convert counts from Counts form to dict form and extract the number of qubits.
    if isinstance(counts, Counts):
        if n_qubits is None:
            n_qubits = len(list(counts.keys())[0])
        counts = counts.int_raw
    elif isinstance(counts, dict):
        if n_qubits is None:
            from math import ceil

            n_qubits = ceil(np.log2(max(counts.keys())))
    else:
        raise TypeError("counts must be of type Counts or dict.")

    arr = np.zeros((2 ** n_qubits,))
    for key, val in counts.items():
        arr[key] = val
    return arr


def circuit_equal(
    circuit1: cirq.Circuit, circuit2: cirq.Circuit, init_state_0: bool = True
) -> bool:
    """Checks if two circuits are equivalent.

    The two circuits are equivalent either when the unitaries are equal up to a phase or 
    when the statevectors with the all 0 initial state are equal up to a phase.
    
    Args:
        circ1: The first cicuit.
        circ2: The second circuit.
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


def decompose_1q_gate(U: np.ndarray, qubit: cirq.Qid) -> List[cirq.Operation]:
    """ZXZXZ decomposition of a single-qubit gate."""
    decomposer = OneQubitEulerDecomposer("U3")
    U_decomposed = decomposer(U)
    theta, phi, lam = U_decomposed[0][0].params

    operations = [
        cirq.rz(lam - np.pi)(qubit),
        cirq.rx(np.pi / 2)(qubit),
        cirq.rz(np.pi - theta)(qubit),
        cirq.rx(np.pi / 2)(qubit),
        cirq.rz(phi)(qubit),
    ]

    assert unitary_equal(cirq.Circuit(operations).unitary(), U)
    return operations
