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
import numpy as np

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

from .circuit_utils import remove_instructions, create_circuit_from_inst_tups

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]
QuantumCircuitLike = Union[QuantumCircuit, Iterable[InstructionTuple]]
AnsatzFunction = Callable[[Sequence[float]], QuantumCircuit]


def get_tomography_labels(n_qubits: int) -> List[str]:
    """Returns the tomography labels on a certain number of qubits."""
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
    """Converts bitstring counts from qiskit.result.Counts form to np.ndarray form.
    
    Args:
        counts: Bitstring counts either in Counts form or in dict form.
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
    circ1: QuantumCircuitLike, circ2: QuantumCircuitLike, init_state_0: bool = True
) -> bool:
    """Checks if two circuits are equivalent.

    The two circuits are equivalent either when the unitaries are equal up to a phase or 
    when the statevectors with initial state all 0 are equal up to a phase.
    
    Args:
        circ1: The first cicuit.
        circ2: The second circuit.
        init_state_0: Whether to assume the initial state is the all 0 state.
        
    Returns:
        is_equal: Whether the two circuits are equivalent.
    """
    # If the circuits are in instruction tuple form.
    if not isinstance(circ1, QuantumCircuit):
        circ1 = create_circuit_from_inst_tups(circ1)
    if not isinstance(circ2, QuantumCircuit):
        circ2 = create_circuit_from_inst_tups(circ2)
    uni1 = get_unitary(circ1)
    uni2 = get_unitary(circ2)
    assert uni1.shape == uni2.shape
    # print('uni1\n', uni1)
    # print('uni2\n', uni2)
    is_equal = unitary_equal(uni1, uni2, init_state_0=init_state_0)
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
        vec1 = uni1[:, 0]
        vec2 = uni2[:, 0]
        ind = np.argmax(np.abs(vec1))
        if abs(vec2[ind]) == 0:
            return False
        phase1 = vec1[ind] / abs(vec1[ind])
        phase2 = vec2[ind] / abs(vec2[ind])
        vec1 /= phase1
        vec2 /= phase2
        is_equal = np.allclose(vec1, vec2)
    else:
        ind = np.argmax(np.abs(uni1[0]))
        if abs(uni2[0, ind]) == 0:
            return False
        phase1 = uni1[0, ind] / abs(uni1[0, ind])
        phase2 = uni2[0, ind] / abs(uni2[0, ind])
        uni1 /= phase1
        uni2 /= phase2
        is_equal = np.allclose(uni1, uni2)
    return is_equal


def vqe_minimize(
    h_op: PauliSumOp,
    ansatz_func: AnsatzFunction,
    init_params: Sequence[float],
    q_instance: QuantumInstance,
) -> Tuple[float, QuantumCircuit]:
    """Minimizes the energy of a Hamiltonian using VQE.
    
    Args:
        h_op: The Hamiltonian operator.
        ansatz_func: The ansatz function for VQE.
        init_params: Initial guess parameters.
        q_instance: The quantum instance for executing the circuits.
    
    Returns:
        energy: The VQE ground state energy.
        ansatz: The VQE ground state ansatz.
    """

    def obj_func_sv(params):
        """VQE objective function for statevector simulator."""
        ansatz = ansatz_func(params).copy()
        result = q_instance.execute(ansatz)
        psi = result.get_statevector()
        energy = psi.conj().T @ h_op.to_matrix() @ psi
        return energy.real

    def obj_func_qasm(params):
        """VQE objective function for QASM simulator and hardware."""
        # This function assumes the Hamiltonian only has the terms
        # II, IZ, ZI, ZZ, IX, ZX, XI, XZ, YY.
        shots = q_instance.run_config.shots

        label_coeff_list = h_op.primitive.to_list()
        label_coeff_dict = dict(label_coeff_list)
        for key in ["II", "IZ", "ZI", "ZZ", "IX", "ZX", "XI", "XZ", "YY"]:
            if key not in label_coeff_dict.keys():
                label_coeff_dict[key] = 0.0

        energy = label_coeff_dict["II"]

        # Measure in ZZ basis
        ansatz = ansatz_func(params).copy()
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ["00", "01", "10", "11"]:
            if key not in counts.keys():
                counts[key] = 0
        energy += (
            label_coeff_dict["IZ"]
            * (counts["00"] + counts["10"] - counts["01"] - counts["11"])
            / shots
        )
        energy += (
            label_coeff_dict["ZI"]
            * (counts["00"] + counts["01"] - counts["10"] - counts["11"])
            / shots
        )
        energy += (
            label_coeff_dict["ZZ"]
            * (counts["00"] - counts["01"] - counts["10"] + counts["11"])
            / shots
        )

        # Measure in ZX basis
        ansatz = ansatz_func(params).copy()
        ansatz.h(0)
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ["00", "01", "10", "11"]:
            if key not in counts.keys():
                counts[key] = 0
        energy += (
            label_coeff_dict["IX"]
            * (counts["00"] + counts["10"] - counts["01"] - counts["11"])
            / shots
        )
        energy += (
            label_coeff_dict["ZX"]
            * (counts["00"] - counts["01"] - counts["10"] + counts["11"])
            / shots
        )

        # Measure in XZ basis
        ansatz = ansatz_func(params).copy()
        ansatz.h(1)
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ["00", "01", "10", "11"]:
            if key not in counts.keys():
                counts[key] = 0
        energy += (
            label_coeff_dict["XI"]
            * (counts["00"] + counts["01"] - counts["10"] - counts["11"])
            / shots
        )
        energy += (
            label_coeff_dict["XZ"]
            * (counts["00"] - counts["01"] - counts["10"] + counts["11"])
            / shots
        )

        # Measure in YY basis
        ansatz = ansatz_func(params).copy()
        ansatz.rx(np.pi / 2, 0)
        ansatz.rx(np.pi / 2, 1)
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ["00", "01", "10", "11"]:
            if key not in counts.keys():
                counts[key] = 0
        energy += (
            label_coeff_dict["YY"]
            * (counts["00"] - counts["01"] - counts["10"] + counts["11"])
            / shots
        )

        energy = energy.real
        return energy

    if q_instance.backend.name() == "statevector_simulator":
        print("STATEVECTOR")
        obj_func = obj_func_sv
    else:
        obj_func = obj_func_qasm
    res = minimize(obj_func, x0=init_params, method="L-BFGS-B")
    energy = res.fun
    ansatz = ansatz_func(res.x)
    print("-" * 80)
    print("in vqe_minimize")
    print("energy =", energy)
    print("ansatz =", ansatz)
    for inst, qargs, cargs in ansatz.data:
        print(inst.name, inst.params, qargs)
    with np.printoptions(precision=15):
        psi = get_statevector(ansatz)
        print(psi)

    h = get_lih_hamiltonian(5.0)
    h_arr = transform_4q_pauli(h.qiskit_op, init_state=[1, 1]).to_matrix()
    print(psi.conj().T @ h_arr @ psi)

    e, v = np.linalg.eigh(h_arr)
    print(e[0])
    print(v[:, 0])
    print("-" * 80)

    return energy, ansatz
