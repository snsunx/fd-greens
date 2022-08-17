"""
==================================================================
Data Generation Utilities (:mod:`fd_greens.data_generation_utils`)
==================================================================
"""

import os
from typing import Sequence, Optional

import numpy as np
import h5py

from .molecular_hamiltonian import get_alkali_hydride_hamiltonian
from .greens_function import GreensFunction
from .response_function import ResponseFunction
from .general_utils import get_fidelity, quantum_state_tomography

__all__ = [
    "generate_greens_function", 
    "generate_response_function",
    "generate_fidelity_vs_depth",
    "generate_fidelity_matrix",
    "generate_trace_matrix"
]

def generate_greens_function(
    h5fnames: Sequence[str],
    omegas: Optional[Sequence[float]] = None,
    eta: float = 1.5
) -> None:
    """Generates Green's function data files.

    Args:
        h5fnames: The HDF5 file names to generate response function from.
        omegas: The frequencies at which the response function is evaluated.
        eta: The broadening factor.
    """
    if "nah" in h5fnames[0]:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.0)
    elif "kh" in h5fnames[0]:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
    else:
        raise ValueError("The input HDF5 file name must contain \"nah\" or \"kh\"")
    if omegas is None:
        np.arange(-32, 32, 0.1)
    
    for h5fname in h5fnames:
        if "exact" in h5fname:
            greens = GreensFunction(hamiltonian, fname=h5fname, method="exact", spin='u')
        else:
            greens = GreensFunction(hamiltonian, fname=h5fname, method="tomo", spin='u')
        greens.process()
        greens.spectral_function(omegas, eta)
        greens.self_energy(omegas, eta)


def generate_response_function(
    h5fnames: Sequence[str],
    omegas: Optional[Sequence[float]] = None,
    eta: float = 1.5
) -> None:
    """Generates response function data files.
    
    Args:
        h5fnames: The HDF5 file names to generate response function from.
        omegas: The frequencies at which the response function is evaluated.
        eta: The broadening factor.
    """
    if "nah" in h5fnames[0]:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 1.7)
        fname_exact = "nah_resp_exact"
    elif "kh" in h5fnames[0]:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
        fname_exact = "kh_resp_exact"
    else:
        raise ValueError("The input HDF5 file names must contain \"nah\" or \"kh\".")
    if omegas is None:
        omegas = np.arange(-32, 32, 0.1)

    for fname in h5fnames:
        if "exact" in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method="exact")
        else:
            resp = ResponseFunction(hamiltonian, fname=fname, method="tomo", fname_exact=fname_exact)
        resp.process()
        resp.response_function(omegas, eta)


def generate_fidelity_vs_depth(h5fname: str, dirname: str = "data/traj", datfname: Optional[str] = None) -> None:
    """Generates fidelity vs depth on a single circuit.
    
    Args:
        h5fname: The HDF5 file name on which to generate the fidelty vs depth data.
        dirname: The directory to save the fidelity vs depth data.
        datfname: The data file name.
    """
    print("> Generating fidelity vs depth")
    if datfname is None:
        datfname = f"fid_vs_depth_{h5fname}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    letter_to_int = {'s': 1, 'd': 2, 't': 3}

    h5file = h5py.File(h5fname + ".h5", "r")

    nq_gates_dict = dict()
    fidelity_dict = dict()
    for key in h5file["psi"].keys():
        state_vector = h5file["psi"][key]
        density_matrix = quantum_state_tomography(h5file, 4, circuit_label=f"circ{int(key[:-1])}", suffix='')

        nq_gates = letter_to_int[key[-1]]
        nq_gates_dict[int(key[:-1])] = nq_gates

        fidelity = get_fidelity(state_vector, density_matrix)
        fidelity_dict[int(key[:-1])] = fidelity
        print(f"key = {key}, fidelity = {fidelity}")

    with open(f"{dirname}/{datfname}.dat", "w") as f:
        for key in sorted(fidelity_dict.keys()):
            f.write(f"{key:3d} {nq_gates_dict[key]:3d} {fidelity_dict[key]:.8f}\n")

    h5file.close()


def generate_fidelity_matrix(
    h5fname_exact: str,
    h5fname_expt: str,
    subscript: Optional[str] = None,
    spin: Optional[str] = None,
    dirname: str = "data/mat",
    datfname: Optional[str] = None
) -> None:
    """Generates the fidelity matrix data file.
    
    Args:
        h5fname_exact: HDF5 file name of the exact data.
        h5fname_expt: HDF5 file name of the experimental data.
        subscript: The subscript of the quantities, ``'e'``, ``'h'`` for Green's function
            and ``'n'`` for response function.
        spin: The spin state of the quantity, ``'u'`` or ``'d'`` for Green's function
            and ``''`` for response function.
        dirname: Directory name of the data file.
        datfname: The data file name.
    """
    print(f"> Generating fidelity matrix of {h5fname_expt}")
    if subscript is None:
        subscript = "e" if "greens" in h5fname_expt else "n"
    if spin is None:
        spin = "u" if "greens" in h5fname_expt else ""
    if datfname is None:
        datfname = "fid_mat_" + '_'.join(h5fname_expt.split('_')[1:])
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
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
    """Genreates the ancilla bitstring probability matrix data file.
    
    Args:
        h5fname: HDF5 file name of the data.
        subscript: The subscript of the quantities, ``'e'``, ``'h'`` for Green's function
            and ``'n'`` for response function.
        spin: The spin state of the quantity, ``'u'`` or ``'d'`` for Green's function
            and ``''`` for response function.
        dirname: Directory name of the data file.
        datfname: The data file name.
    """
    print(f"> Generating trace matrix of {h5fname}")
    if subscript is None:
        subscript = "e" if "greens" in h5fname else "n"
    if spin is None:
        spin = "u" if "greens" in h5fname else ""
    if datfname is None:
        datfname = "trace_mat_" + '_'.join(h5fname.split('_')[1:])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

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
