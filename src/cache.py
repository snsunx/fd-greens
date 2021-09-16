from typing import Union
import json
import os

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

    