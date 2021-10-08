"""Utility functions"""

from typing import ClassVar, Optional, Union, Iterable, List, Tuple
import numpy as np

from qiskit import QuantumCircuit, Aer, IBMQ, execute
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.ignis.verification.tomography import (state_tomography_circuits,
                                                  StateTomographyFitter)
from qiskit.circuit import Instruction

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
    circ_data_new = []
    for inst_tup in circ_data:
        if inst_tup[0].name != 'barrier':
            circ_data_new.append(inst_tup)
    return circ_data_new

def data_to_circuit(data, n_qubits=None):
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
