from typing import Sequence, Optional

import numpy as np
import h5py

from .molecular_hamiltonian import MolecularHamiltonian, get_nah_hamiltonian
from .parameters import HARTREE_TO_EV
from .greens_function import GreensFunction
from .response_function import ResponseFunction
from .general_utils import get_fidelity

__all__ = [
    "generate_greens_function", 
    "generate_response_function",
    "generate_fidelity_vs_depth",
    "generate_fidelity_and_trace"
]

def generate_greens_function():
    return

def generate_response_function(
    h5fnames: Sequence[str],
    hamiltonian: Optional[MolecularHamiltonian] = None,
    omegas: Optional[Sequence[float]] = None,
    eta: float = 0.02 * HARTREE_TO_EV
) -> None:
    if hamiltonian is None:
        hamiltonian = get_nah_hamiltonian(3.7)
    if omegas is None:
        omegas = np.arange(-32, 32, 0.1)

    for fname in h5fnames:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method="exact")
        else:
            resp = ResponseFunction(hamiltonian, fname=fname, method="tomo", fname_exact='lih_resp_exact')
        resp.process()
        resp.response_function(omegas, eta)

def generate_fidelity_vs_depth(h5fname0: str, h5fname1: str, dirname: str = "data", datfname: str = "fidelity_vs_depth") -> None:
    letter_to_int = {'s': 1, 'd': 2, 't': 3}

    h5file0 = h5py.File(h5fname0 + ".h5", "r")
    h5file1 = h5py.File(h5fname1 + ".h5", "r")

    group0 = h5file0["psi"] if "psi" in h5file0 else h5file0["rho"]
    group1 = h5file1["psi"] if "psi" in h5file1 else h5file1["rho"]

    nq_gates_dict = dict()
    fidelity_dict = dict()
    for key in group0.keys():
        state0 = group0[key]
        state1 = group1[key]

        nq_gates = letter_to_int[key[-1]]
        nq_gates_dict[int(key[:-1])] = nq_gates

        fidelity = get_fidelity(state0, state1)
        fidelity_dict[int(key[:-1])] = fidelity

    with open(f"{dirname}/{datfname}.dat", "w") as f:
        for key in sorted(fidelity_dict.keys()):
            f.write(f"{key:2d} {nq_gates_dict[key]:2d} {fidelity_dict[key]:.8f}\n")

    h5file0.close()
    h5file1.close()

def generate_fidelity_and_trace():
    return