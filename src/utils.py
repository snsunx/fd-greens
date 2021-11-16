"""Utility functions"""

from typing import ClassVar, Optional, Union, Iterable, List, Tuple, Sequence
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, IBMQ, execute
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.ignis.verification.tomography import (state_tomography_circuits,
                                                  StateTomographyFitter)
from qiskit.circuit import Instruction
from qiskit.opflow import PauliSumOp

CircuitData = Iterable[Tuple[Instruction, List[int], Optional[List[int]]]]


def get_quantum_instance(backend,
                         noise_model_name=None,
                         optimization_level=0,
                         initial_layout=None,
                         shots=1):
    # TODO: Write the part for IBMQ backends
    if isinstance(backend, AerBackend):
        if noise_model_name is None:
            q_instance = QuantumInstance(backend=backend, shots=shots)
        else:
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q-research', group='caltech-1', project='main')
            device = provider.get_backend(noise_model_name)
            q_instance = QuantumInstance(
                backend=backend, shots=shots,
                noise_model=NoiseModel.from_backend(device.properties()),
                coupling_map=device.configuration().coupling_map,
                optimization_level=optimization_level,
                initial_layout=initial_layout)
    return q_instance

def get_statevector(circ: QuantumCircuit,
                    reverse: bool = False) -> np.ndarray:
    """Returns the statevector of a quantum circuit.

    Args:
        circ: The circuit on which the state is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        The statevector array of the circuit.
    """
    if isinstance(circ, list): # CircuitData
        circ = data_to_circuit(circ)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    statevector = result.get_statevector()
    if reverse:
        statevector = reverse_qubit_order(statevector)
    return statevector

def get_unitary(circ: Union[QuantumCircuit, CircuitData],
                reverse: bool = False, n_qubits = None) -> np.ndarray:
    """Returns the unitary of a quantum circuit.

    Args:
        circ: The circuit on which the unitary is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        The unitary array of the circuit.
    """

    if isinstance(circ, list): # CircuitData
        circ = data_to_circuit(circ, n_qubits=n_qubits)
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circ, backend)
    result = job.result()
    unitary = result.get_unitary()
    if reverse:
        unitary = reverse_qubit_order(unitary)
    return unitary

def remove_barriers(circ_data):
    """Removes barriers in circuit data."""
    circ_data_new = []
    for inst_tup in circ_data:
        if inst_tup[0].name != 'barrier':
            circ_data_new.append(inst_tup)
    return circ_data_new

def data_to_circuit(data, n_qubits=None, remove_barr=True):
    """Converts circuit data to circuit."""
    if remove_barr:
        data = remove_barriers(data)
    if n_qubits is None:
        try:
            n_qubits = max([max(x[1]) for x in data]) + 1
        except:
            n_qubits = max([max([y.index for y in x[1]]) for x in data]) + 1
    circ_new = QuantumCircuit(n_qubits)
    for inst_tup in data:
        inst, qargs = inst_tup[:2]
        try:
            circ_new.append(inst, qargs)
        except:
            qargs = [q.index for q in qargs]
            circ_new.append(inst, qargs)
    return circ_new

def reverse_qubit_order(arr: np.ndarray) -> np.ndarray:
    """Reverses qubit order in a 1D or 2D array.

    Args:
        The array on which the qubit order is to be reversed.

    Returns:
        The array after qubit order is reversed.
    """
    if len(arr.shape) == 1:
        dim = arr.shape[0]
        num = int(np.log2(dim))
        shape_expanded = (2,) * num
        inds_transpose = np.flip(np.arange(num))
        arr = arr.reshape(*shape_expanded)
        arr = arr.transpose(*inds_transpose)
        arr = arr.reshape(dim)
    elif len(arr.shape) == 2:
        shape_original = arr.shape
        dim = shape_original[0]
        num = int(np.log2(dim))
        shape_expanded = (2,) * 2 * num
        inds_transpose = list(reversed(range(num))) + list(reversed(range(num, 2 * num)))
        arr = arr.reshape(*shape_expanded)
        arr = arr.transpose(*inds_transpose)
        arr = arr.reshape(*shape_original)
    else:
        raise NotImplementedError("Reversing qubit order of array with more"
                                  "than two dimensions is not implemented.")
    return arr

def state_tomography(circ: QuantumCircuit,
                     q_instance: Optional[QuantumInstance] = None
                     ) -> np.ndarray:
    """Performs state tomography on a quantum circuit.

    Args:
        circ: The quantum circuit to perform tomography on.
        q_instance: The QuantumInstance to execute the circuit.

    Returns:
        The density matrix obtained from state tomography.
    """
    if q_instance is None:
        backend = Aer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend, shots=8192)

    qreg = circ.qregs[0]
    qst_circs = state_tomography_circuits(circ, qreg)
    if True:
        fig = qst_circs[5].draw(output='mpl')
        fig.savefig('qst_circ.png')
    result = q_instance.execute(qst_circs)
    qst_fitter = StateTomographyFitter(result, qst_circs)
    rho_fit = qst_fitter.fit(method='lstsq')
    return rho_fit

def save_circuit(circ, 
                 fname,
                 savetxt: bool = True,
                 savefig: bool = True) -> None:
    """Saves a circuit to disk in QASM string form and/or figure form.
    
    Args:
        fname: The file name.
        savetxt: Whether to save the QASM string of the circuit as a text file.
        savefig: Whether to save the figure of the circuit.
    """
        
    if savefig:
        fig = circ.draw(output='mpl')
        fig.savefig(fname + '.png')
    
    if savetxt:
        circ_data = []
        for inst_tup in circ.data:
            if inst_tup[0].name != 'c-unitary':
                circ_data.append(inst_tup)
        circ = data_to_circuit(circ_data, remove_barr=False)
        # for inst_tup in circ_data:
        #    print(inst_tup[0].name)
        # exit()
        f = open(fname + '.txt', 'w')
        qasm_str = circ.qasm()
        f.write(qasm_str)
        f.close()

def solve_energy_probabilities(a, b):
    A = np.array([[1, 1], [a[0], a[1]]])
    x = np.linalg.inv(A) @ np.array([1.0, b])
    return x

def measure_operator(circ: QuantumCircuit,
                     op: PauliSumOp,
                     q_instance: QuantumInstance,
                     anc_state: Sequence[int] = [],
                     qreg: Optional[QuantumRegister] = None) -> float:
    """Measures an operator on a circuit.

    Args:
        circ: The quantum circuit to be measured.
        op: The operator to be measured.
        q_instance: The QuantumInstance on which to execute the circuit.
        anc_state: A sequence of integers indicating the ancilla states.
        qreg: The quantum register on which the operator is measured.
    
    Returns:
        The value of the operator on the circuit.
    """
    if qreg is None:
        qreg = circ.qregs[0]
    n_qubits = len(qreg)
    n_anc = len(anc_state)
    n_sys = n_qubits - n_anc

    circ_list = []
    for term in op:
        label = term.primitive.table.to_labels()[0]
        
        # Create circuit for measuring this Pauli string
        circ_meas = circ.copy()
        creg = ClassicalRegister(n_qubits)
        circ_meas.add_register(creg)

        for i in range(n_anc):
            circ_meas.measure(i, i)
        
        for i in range(n_sys):
            if label[n_sys - 1 - i] == 'X':
                circ_meas.h(qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == 'Y':
                circ_meas.rx(np.pi / 2, qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == 'Z':
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])

        circ_list.append(circ_meas)
    
    result = q_instance.execute(circ_list)
    counts_list = result.get_counts()

    value = 0
    for i, counts in enumerate(counts_list):
        # print('counts =', counts)
        coeff = op.primitive.coeffs[i]
        counts_new = counts.copy()

        for key, val in counts.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if key_anc != anc_state:
                del counts_new[key]
        
        shots = sum(counts_new.values())
        for key, val in counts_new.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if sum(key_sys) % 2 == 0:
                value += coeff * val / shots
            else:
                value -= coeff * val / shots
    return value

