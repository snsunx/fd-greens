from hamiltonians import MolecularHamiltonian
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp
from itertools import combinations
import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion.ops.operators.qubit_operator import QubitOperator
from scipy.sparse.data import _data_matrix
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel

from qiskit import *
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend

def get_quantum_instance(backend, noise_model_name=None, 
                         optimization_level=0, initial_layout=None, shots=1):
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

def get_number_state_indices(n_orb, n_elec, anc='', return_type='decimal'):
    assert return_type in ['binary', 'decimal']
    inds = []
    for tup in combinations(range(n_orb), n_elec):
        bin_list = ['1' if (n_orb - 1 - i) in tup else '0' for i in range(n_orb)]
        bin_str = anc + ''.join(bin_list)
        inds.append(bin_str)
    inds.sort()
    if return_type == 'decimal':
        inds = [int(s, 2) for s in inds]
    return inds

def number_state_eigensolver(hamiltonian, n_elec):
    """Exact eigensolver for the Hamiltonian in the subspace of 
    a certain number of electrons."""
    if isinstance(hamiltonian, MolecularHamiltonian):
        hamiltonian_arr = hamiltonian.to_array(array_type='sparse')
    elif isinstance(hamiltonian, QubitOperator):
        hamiltonian_arr = get_sparse_operator(hamiltonian)
    elif (isinstance(hamiltonian, _data_matrix) or 
          isinstance(hamiltonian, np.ndarray)):
        hamiltonian_arr = hamiltonian
    else:
        raise TypeError("""hamiltonian must be one of MolecularHamiltonian, 
                        QubitOperator, sparse array or ndarray""")

    n_orb = int(np.log2(hamiltonian_arr.shape[0]))
    inds = get_number_state_indices(n_orb, n_elec)
    hamiltonian_subspace = hamiltonian_arr[inds][:, inds]
    if isinstance(hamiltonian_subspace, _data_matrix):
        hamiltonian_subspace = hamiltonian_subspace.toarray()
    
    eigvals, eigvecs = np.linalg.eigh(hamiltonian_subspace)
    return eigvals, eigvecs

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



