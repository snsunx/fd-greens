"""I/O utility functions"""
# TODO: This script need to be rewritten and renamed to io_utils.py

from typing import Union, Tuple
import json
import os

from qiskit import QuantumCircuit
from qiskit.algorithms import VQEResult

from params import HARTREE_TO_EV
from hamiltonians import MolecularHamiltonian

class CacheRecompilation:

    folder = 'cache'

    @classmethod
    def __root(cls) -> str:
        """Obtains absolute path to cache storage directory."""
        # We need to backtrack one level and then into `cache` folder.
        # TODO: This implies cache has to exist. Implement an exists and
        # mkdir statement.
        path = os.path.abspath(os.path.dirname(__file__))
        path = path[:-3] + 'cache/'
        return path


    @classmethod
    def __filepath(cls,
                   hamiltonian: MolecularHamiltonian,
                   index: str,
                   states: str) -> str:
        """Constructs filename for caching circuits."""
        return cls.__root() + f'{hamiltonian.name}_{index}_{states}.txt'


    @classmethod
    def save_recompiled_circuit(cls,
                                hamiltonian: MolecularHamiltonian,
                                index: str,
                                states: str,
                                circuit: str,
                                verbose: bool = True) -> None:
        """Saves given circuit into cache file."""
        with open(cls.__filepath(hamiltonian, index, states), 'w') as f:
            f.write(json.dumps(circuit))
        if verbose:
            print('[Cache] Wrote circuit to '
                  f'{cls.__filepath(hamiltonian, index, states)}')


    @classmethod
    def load_recompiled_circuit(cls,
                                hamiltonian: MolecularHamiltonian,
                                index: str,
                                states: str,
                                verbose: bool = True) -> Union[str, None]:
        """Loads circuit for given Hamiltonian, index, and states. If circuit
        is not found, returns None."""
        if not os.path.isfile(cls.__filepath(hamiltonian, index, states)):
            return None   # File does not exist.
        with open(cls.__filepath(hamiltonian, index, states), 'r') as f:
            circuit = json.loads(f.read())
        if verbose:
            print('[Cache] Read circuit from'
                  f'{cls.__filepath(hamiltonian, index, states)}')
        return circuit

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