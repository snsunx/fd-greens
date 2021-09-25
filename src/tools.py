from typing import Union, Tuple, List, Iterable, Optional, Sequence
import json

import numpy as np

from qiskit import QuantumCircuit, Aer, IBMQ, execute
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.quantum_info import PauliTable, SparsePauliOp
# from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.algorithms import VQEResult

from openfermion.ops import PolynomialTensor
from openfermion.transforms import get_fermion_operator, jordan_wigner

from z2_symmetries import apply_cnot_z2, transform_4q_hamiltonian
from constants import HARTREE_TO_EV

PauliTuple = Tuple[Tuple[str, int]]

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

def reverse_qubit_order(arr):
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
    return arr

def get_statevector(circ):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    statevector = result.get_statevector()
    return statevector

def get_unitary(circ):
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circ, backend)
    result = job.result()
    unitary = result.get_unitary()
    return unitary

# FIXME: The load feature is not working due to job ID retrieval problem
def load_vqe_result(ansatz: QuantumCircuit, prefix: str = None) -> Tuple[float, QuantumCircuit]:
    """Loads the VQE energy and optimal parameters from files."""
    if prefix is None:
        prefix = 'vqe'
    with open(prefix + '_energy.txt', 'r') as f:
        energy_gs = json.loads(f.read())
    with open(prefix + '_ansatz.txt', 'r') as f:
        params_dict = json.loads(f.read())
        params_dict_new = {}
        for key, val in params_dict.items():
            for param in ansatz.parameters:
                if param.name == key:
                    params_dict_new.update({param: val})
        ansatz_new = ansatz.assign_parameters(params_dict_new)
    return energy_gs, ansatz_new

def save_vqe_result(vqe_result: VQEResult, prefix: str = None) -> None:
    """Saves VQE energy and optimal parameters to files."""
    if prefix is None:
        prefix = 'vqe'
    with open(prefix + '_energy.txt', 'w') as f:
        energy_gs = vqe_result.optimal_value * HARTREE_TO_EV
        f.write(json.dumps(energy_gs))
    with open(prefix + '_ansatz.txt', 'w') as f:
        params_dict = vqe_result.optimal_parameters
        params_dict_new = {}
        for key, val in params_dict.items():
            params_dict_new.update({str(key): val})
        f.write(json.dumps(params_dict_new))

# TODO: Deprecate this function
def get_pauli_tuple(n_qubits: int, ind: int
                    ) -> Tuple[PauliTuple]:
    """Obtains the tuple form of a Pauli string from number of qubits and 
    creation/annihilation operator index.
    
    Args:
        n_qubits: The number of qubits.
        ind: The index of the qubit on which the creation/annihilation
            operator acts on.
    
    Return:
        The Pauli string in tuple form.
    """
    arr = np.zeros((n_qubits,))
    arr[ind] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    tup_xy = tuple(qubit_op.terms)
    return tup_xy

def get_a_operator(n_qubits: int, ind: int) -> SparsePauliOp:
    """Returns the creation/annihilation operator."""
    label_x = 'I' * (n_qubits - ind - 1) + 'X' + 'Z' * ind
    label_y = 'I' * (n_qubits - ind - 1) + 'Y' + 'Z' * ind
    pauli_table = PauliTable.from_labels([label_x, label_y])
    sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=[1.0, 1.0j])
    return sparse_pauli_op

def pauli_label_to_tuple(label: str) -> PauliTuple:
    """Converts a Pauli string from label form to tuple form.
    
    Args:
        The Pauli string in label form.
        
    Returns:
        The Pauli string in tuple form.
    """
    lst = []
    for i, c in enumerate(label[::-1]):
        if c != 'I':
            lst.append((i, c))
    tup = tuple(lst)
    return tup

def get_indices_with_ancilla(inds: Iterable[str], anc: str):
    """Returns indices with ancillas positions."""
    print('inds =', inds)
    inds_new = [ind + anc for ind in inds]
    inds_new = [int(ind, 2) for ind in inds_new]
    return inds_new


# XXX: The following function is hardcoded
# XXX: Treating the spin this way is not correct. See Markdown document.
def get_pauli_tuple_dictionary(spin='up'):
    labels_x = ['IIIX', 'IIXZ', 'IXZZ', 'XZZZ']
    labels_y = ['IIIY', 'IIYZ', 'IYZZ', 'YZZZ']
    pauli_table = PauliTable.from_labels(labels_x + labels_y)
    sparse_pauli_op = SparsePauliOp(pauli_table, coeffs=[1.] * 4 + [1j] * 4)
    for i in range(8):
        print(sparse_pauli_op[i].coeffs, sparse_pauli_op[i].table)

    sparse_pauli_op = transform_4q_hamiltonian(sparse_pauli_op, init_state=[1, 1])
    dic = {}
    if spin == 'up':
        for i in range(2):
            dic.update({i: [sparse_pauli_op[2 * i], sparse_pauli_op[2 * i + 4]]})
    else:
        for i in range(2):
            dic.update({i: [sparse_pauli_op[2 * i + 1].primitive, sparse_pauli_op[2 * i + 1 + 4]]})
    return dic