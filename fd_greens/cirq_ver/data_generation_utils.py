"""
==================================================================
Data Generation Utilities (:mod:`fd_greens.data_generation_utils`)
==================================================================
"""

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
    "generate_fidelity_matrix",
    "generate_trace_matrix"
]

def generate_greens_function(
    h5fnames: Sequence[str],
    hamiltonian: Optional[MolecularHamiltonian] = None,
    omegas: Optional[Sequence[float]] = None,
    eta: float = 0.02 * HARTREE_TO_EV
) -> None:
    """Generates Green's function data files."""
    if hamiltonian is None:
        hamiltonian = get_nah_hamiltonian(3.7)
    if omegas is None:
        omegas = np.arange(-32, 32, 0.1)
    
    for h5fname in h5fnames:
        if 'exact' in h5fname:
            greens = GreensFunction(hamiltonian, fname=h5fname, method='exact', spin='u')
        else:
            greens = GreensFunction(hamiltonian, fname=h5fname, method='tomo', spin='u')
        greens.process()
        greens.spectral_function(omegas, eta)
        greens.self_energy(omegas, eta)


def generate_response_function(
    h5fnames: Sequence[str],
    hamiltonian: Optional[MolecularHamiltonian] = None,
    omegas: Optional[Sequence[float]] = None,
    eta: float = 1.5 # 0.05 * HARTREE_TO_EV
) -> None:
    """Generates response function data files."""
    if hamiltonian is None:
        hamiltonian = get_nah_hamiltonian(3.7) # XXX
    if omegas is None:
        omegas = np.arange(-32, 32, 0.1)

    for fname in h5fnames:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method="exact")
        else:
            resp = ResponseFunction(hamiltonian, fname=fname, method="tomo", fname_exact='nah_resp_exact') # XXX
        resp.process()
        resp.response_function(omegas, eta)


def generate_fidelity_vs_depth(
    h5fname0: str,
    h5fname1: str,
    dirname: str = "data/traj", 
    datfname: Optional[str] = None
) -> None:
    """Generates fidelity vs depth on a single circuit."""
    if datfname is None:
        datfname = f"fid_vs_depth_{h5fname1}"
    
    print("> Generating fidelity vs depth")
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


def generate_fidelity_matrix(
    h5fname_exact: str,
    h5fname_expt: str,
    subscript: Optional[str] = None,
    spin: Optional[str] = None,
    dirname: str = "data/mat",
    datfname: Optional[str] = None
) -> None:
    """Generates the fidelity matrix."""
    print(f"> Generating fidelity matrix of {h5fname_expt}")
    if subscript is None:
        subscript = "e" if "greens" in h5fname_expt else "n"
    if spin is None:
        spin = "u" if "greens" in h5fname_expt else ""
    if datfname is None:
        datfname = "fid_mat_" + '_'.join(h5fname_expt.split('_')[1:])
    
    h5file_exact = h5py.File(h5fname_exact + ".h5", "r")
    h5file_expt = h5py.File(h5fname_expt + ".h5", "r")

    dim = 2 if spin in ['u', 'd'] else 4
    fidelity_matrix = np.zeros((dim, dim))

    for i in range(dim):
        state_exact = h5file_exact[f"psi/{subscript}{i}{spin}"][:]
        state_expt = h5file_expt[f"rho/{subscript}{i}{spin}"][:]
        
        fidelity_matrix[i, i] = get_fidelity(state_exact, state_expt)

        for j in range(i + 1, dim):
            # p is filled on the upper triangle.
            state_exact = h5file_exact[f"psi/{subscript}p{i}{j}{spin}"][:]
            state_expt = h5file_expt[f"rho/{subscript}p{i}{j}{spin}"][:]
            fidelity_matrix[i, j] = get_fidelity(state_exact, state_expt)

            # m is filled on the lower triangle.
            state_exact = h5file_exact[f"psi/{subscript}m{i}{j}{spin}"][:]
            state_expt = h5file_expt[f"rho/{subscript}m{i}{j}{spin}"][:]
            fidelity_matrix[j, i] = get_fidelity(state_exact, state_expt)

    np.savetxt(f"{dirname}/{datfname}.dat", fidelity_matrix)

    h5file_exact.close()
    h5file_expt.close()


def generate_trace_matrix(
    h5fname: str,
    subscript: Optional[str] = None,
    spin: Optional[str] = None,
    dirname: str = "data/mat",
    datfname: Optional[str] = None,
) -> None:
    """Genreates the ancilla bitstring probabilities."""
    print(f"> Generating trace matrix of {h5fname}")
    if subscript is None:
        subscript = "e" if "greens" in h5fname else "n"
    if spin is None:
        spin = "u" if "greens" in h5fname else ""
    if datfname is None:
        datfname = "trace_mat_" + '_'.join(h5fname.split('_')[1:])

    h5file = h5py.File(h5fname + ".h5", "r")

    dim = 2 if spin in ['u', 'd'] else 4

    trace_matrix = np.zeros((dim, dim))
    for i in range(dim):
        trace_matrix[i, i] = h5file[f"trace/{subscript}{i}{spin}"][()]
        for j in range(i + 1, dim):
            trace_matrix[i, j] = h5file[f"trace/{subscript}p{i}{j}{spin}"][()]
            trace_matrix[j, i] = h5file[f"trace/{subscript}m{i}{j}{spin}"][()]
    
    np.savetxt(f"{dirname}/{datfname}.dat", trace_matrix)

    h5file.close()
