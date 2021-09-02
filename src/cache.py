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
        path = os.path.abspath(os.path.dirname(__file__))
        path = path[:-3] + 'cache/'
        return path


    @classmethod
    def __filepath(cls, hamiltonian: MolecularHamiltonian, type: str) -> str:
        """Constructs filename for caching circuits."""
        return cls.__root() + f'{hamiltonian.name}-{type}.txt'
        

    @classmethod
    def save_recompiled_circuit(cls,
                                hamiltonian: MolecularHamiltonian,
                                type: str, circuit: str,
                                verbose: bool = True) -> None:
        """Saves given circuit into cache file."""
        with open(cls.__filepath(hamiltonian, type), 'w') as f:
            f.write(json.dumps(circuit))
        if verbose:
            print(f'[Cache] Wrote circuit to  {cls.__filepath(hamiltonian, type)}')


    @classmethod
    def load_recompiled_circuit(cls, 
                                hamiltonian: MolecularHamiltonian,
                                type: str,
                                verbose: bool = True) -> Union[str, None]:
        """Loads circuit for given Hamiltonian and `type`. If circuit is not
        found, returns None."""
        if not os.path.isfile(cls.__filepath(hamiltonian, type)):
            return None   # File does not exist.
        with open(cls.__filepath(hamiltonian, type), 'r') as f:
            circuit = json.loads(f.read())
        if verbose:
            print(f'[Cache] Read circuit from {cls.__filepath(hamiltonian, type)}')
        return circuit

    