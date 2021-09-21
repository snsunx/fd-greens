from typing import Union, Tuple, List, Iterable, Optional
import json

from hamiltonians import MolecularHamiltonian
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp
from itertools import combinations
from constants import HARTREE_TO_EV
from qiskit.algorithms import VQEResult

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

def get_number_state_indices(n_orb: int,
                             n_elec: int,
                             anc: Iterable[str] = '',
                             return_type: str = 'decimal',
                             reverse: bool = True) -> List[int]:
    """Obtains the indices corresponding to a certain number of electrons.
    
    Args:
        n_orb: An integer indicating the number of orbitals.
        n_elec: An integer indicating the number of electrons.
        anc: An iterable of '0' and '1' indicating the state of the
            ancilla qubit(s).
        return_type: Type of the indices returned. Must be 'decimal'
            or 'binary'. Default to 'decimal'.
        reverse: Whether the qubit indices are reversed because of 
            Qiskit qubit order. Default to True.
    """
    assert return_type in ['binary', 'decimal']
    inds = []
    for tup in combinations(range(n_orb), n_elec):
        bin_list = ['1' if (n_orb - 1 - i) in tup else '0' 
                    for i in range(n_orb)]
        if reverse:
            bin_str = ''.join(bin_list) + anc[::-1]
        else:
            bin_str = anc + ''.join(bin_list)
        inds.append(bin_str)
    if reverse:
        inds = sorted(inds, reverse=True)
    if len(anc) <= 1:
        print(anc)
        print(inds)
    if return_type == 'decimal':
        inds = [int(s, 2) for s in inds]
    return inds

def number_state_eigensolver(
        hamiltonian: Union[MolecularHamiltonian, QubitOperator, np.ndarray], 
        n_elec: int,
        reverse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact eigensolver for the Hamiltonian in the subspace of 
    a certain number of electrons.
    
    Args:
        hamiltonian: The Hamiltonian of the molecule.
        n_elec: An integer indicating the number of electrons.
    `
    Returns:
        eigvals: The eigenenergies in the number state subspace.
        eigvecs: The eigenstates in the number state subspace.
    """
    if isinstance(hamiltonian, MolecularHamiltonian):
        hamiltonian_arr = hamiltonian.to_array(array_type='sparse')
    elif isinstance(hamiltonian, QubitOperator):
        hamiltonian_arr = get_sparse_operator(hamiltonian)
    elif (isinstance(hamiltonian, _data_matrix) or 
          isinstance(hamiltonian, np.ndarray)):
        hamiltonian_arr = hamiltonian
    else:
        raise TypeError("Hamiltonian must be one of MolecularHamiltonian,"
                        "QubitOperator, sparse array or ndarray")

    n_orb = int(np.log2(hamiltonian_arr.shape[0]))
    inds = get_number_state_indices(n_orb, n_elec, reverse=reverse)
    hamiltonian_subspace = hamiltonian_arr[inds][:, inds]
    if isinstance(hamiltonian_subspace, _data_matrix):
        hamiltonian_subspace = hamiltonian_subspace.toarray()
    
    eigvals, eigvecs = np.linalg.eigh(hamiltonian_subspace)

    # TODO: Note that the minus sign below depends on the `reverse` variable. 
    # Might need to take care of this
    sort_arr = [(eigvals[i], -np.argmax(np.abs(eigvecs[:, i]))) 
                for i in range(len(eigvals))]
    #print(sort_arr)
    #print(sorted(sort_arr))
    inds_new = sorted(range(len(sort_arr)), key=sort_arr.__getitem__)
    #print(inds_new)
    eigvals = eigvals[inds_new]
    eigvecs = eigvecs[:, inds_new]
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