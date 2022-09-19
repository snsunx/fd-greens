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
from .parameters import HARTREE_TO_EV

__all__ = [
    "generate_greens_function", 
    "generate_response_function",
    "generate_fidelity_vs_depth",
    "generate_fidelity_matrix",
    "generate_trace_matrix"
]

def generate_greens_function(
    h5fname: str,
    omegas: Optional[Sequence[float]] = None,
    eta: float = 0.75 # 0.02 * HARTREE_TO_EV
) -> None:
    """Generates Green's function data files.

    Args:
        h5fnames: The HDF5 file names to generate response function from.
        omegas: The frequencies at which the response function is evaluated.
        eta: The broadening factor.
    """
    if "lih" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Li", 3.0)
    elif "nah" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
    elif "kh" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
    
    if omegas is None:
        omegas = np.arange(-32, 32, 0.1)
    
    if "exact" in h5fname:
        greens = GreensFunction(hamiltonian, h5fname=h5fname, method="exact")
    else:
        greens = GreensFunction(hamiltonian, h5fname=h5fname, method="tomo")
    greens.process()
    greens.spectral_function(omegas, eta)
    # greens.self_energy(omegas, eta)


def generate_response_function(
    h5fname: str,
    omegas: Optional[Sequence[float]] = None,
    eta: float = 1.5
) -> None:
    """Generates response function data files.
    
    Args:
        h5fnames: The HDF5 file names to generate response function from.
        omegas: The frequencies at which the response function is evaluated.
        eta: The broadening factor.
    """
    if "nah" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
        h5fname_exact = "nah_resp_exact"
    elif "kh" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
        h5fname_exact = "kh_resp_exact"
    else:
        raise ValueError("The input HDF5 file names must contain \"nah\" or \"kh\".")
    if omegas is None:
        omegas = np.arange(-32, 32, 0.1)

    if "exact" in h5fname:
        resp = ResponseFunction(hamiltonian, h5fname=h5fname, method="exact")
    else:
        resp = ResponseFunction(hamiltonian, h5fname=h5fname, method="tomo", h5fname_exact=h5fname_exact)
    resp.process()
    resp.response_function(omegas, eta)


def generate_fidelity_vs_depth(
    h5fname: str,
    dirname: str = "data/traj",
    datfname: Optional[str] = None
) -> None:
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
        key_int = int(key[:-1])
        # if key[:-1] == "last":
        #     key_int = 999
        # else:
        #     key_int = int(key[:-1])
        
        if True: # key[:-1] != "1" and key[:-1] != "3":
            # print(f"key = {key}")
            state_vector = h5file["psi"][key]
            # print(f"{state_vector.shape =}")
            density_matrix = quantum_state_tomography(h5file, 4, circuit_label=f"circ{key[:-1]}", suffix='')
            # if key == "1s":
            #     print(key, '\n', density_matrix)

            nq_gates = letter_to_int[key[-1]]
            nq_gates_dict[key_int] = nq_gates

            fidelity = get_fidelity(state_vector, density_matrix)
            fidelity_dict[key_int] = fidelity
            print(f"key = {key}, fidelity = {fidelity}")

    with open(f"{dirname}/{datfname}.dat", "w") as f:
        for key in sorted(fidelity_dict.keys()):
            f.write(f"{key:5d} {nq_gates_dict[key]:5d} {fidelity_dict[key]:.8f}\n")

    h5file.close()


def generate_fidelity_matrix(
    h5fname_exact: str,
    h5fname_expt: str,
    dirname: str = "data/mat",
    datfname: Optional[str] = None,
    calculation_mode: Optional[str] = None
) -> None:
    """Generates the fidelity matrix data file.
    
    Args:
        h5fname_exact: HDF5 file name of the exact data.
        h5fname_expt: HDF5 file name of the experimental data.
        dirname: Name of the data file directory.
        datfname: The data file name.
        calculation_mode: Mode of the calculation.
    """
    print(f"> Generating fidelity matrix of {h5fname_expt}")
    if calculation_mode is None:
        # Infer calculation mode from the file name.
        calculation_mode = h5fname_expt.split("_")[1]
    assert calculation_mode in ["greens", "resp"]
    if datfname is None:
        datfname = "fid_mat_" + h5fname_expt
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    h5file_exact = h5py.File(h5fname_exact + ".h5", "r")
    h5file_expt = h5py.File(h5fname_expt + ".h5", "r")

    subscripts  = "eh" if calculation_mode == "greens" else "n"
    spins = "u" if calculation_mode == "greens" else " " # XXX: Should be changed to "ud"
    dim = 2 if calculation_mode == "greens" else 4

    for s in subscripts:
        for spin in spins:
            fidelity_matrix = np.zeros((dim, dim))

            for i in range(dim):
                state_exact = h5file_exact[f"psi{spin.strip()}/{s}{i}"][:]
                state_expt = h5file_expt[f"rho{spin.strip()}/{s}{i}"][:]
                fidelity_matrix[i, i] = get_fidelity(state_exact, state_expt)

                for j in range(i + 1, dim):
                    # p values is filled on the upper triangle.
                    state_exact = h5file_exact[f"psi{spin.strip()}/{s}p{i}{j}"][:]
                    state_expt = h5file_expt[f"rho{spin.strip()}/{s}p{i}{j}"][:]
                    fidelity_matrix[i, j] = get_fidelity(state_exact, state_expt)

                    # m values is filled on the lower triangle.
                    state_exact = h5file_exact[f"psi{spin.strip()}/{s}m{i}{j}"][:]
                    state_expt = h5file_expt[f"rho{spin.strip()}/{s}m{i}{j}"][:]
                    fidelity_matrix[j, i] = get_fidelity(state_exact, state_expt)
            
            s_ = '' if s == 'n' else s
            np.savetxt(f"{dirname}/{datfname}{s_}{spin.strip()}.dat", fidelity_matrix)

    h5file_exact.close()
    h5file_expt.close()


def generate_trace_matrix(
    h5fname: str,
    dirname: str = "data/mat",
    datfname: Optional[str] = None,
    calculation_mode: Optional[str] = None
) -> None:
    """Genreates the ancilla bitstring probability matrix data file.
    
    Args:
        h5fname: HDF5 file name of the data.
        dirname: Name of the data file directory.
        datfname: The data file name.
        calculation_mode: The calculation mode.
    """
    print(f"> Generating trace matrix of {h5fname}")
    if calculation_mode is None:
        # Infer calculation mode from the file name.
        calculation_mode = h5fname.split("_")[1]
    assert calculation_mode in ["greens", "resp"]
    if datfname is None:
        datfname = "trace_mat_" + h5fname
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    subscripts = "eh" if calculation_mode == "greens" else "n"
    spins = "u" if calculation_mode == "greens" else " " # XXX: Should be changed to "ud"
    dim = 2 if calculation_mode == "greens" else 4

    h5file = h5py.File(h5fname + ".h5", "r")

    for s in subscripts:
        for spin in spins:
            trace_matrix = np.zeros((dim, dim))
            for i in range(dim):
                trace_matrix[i, i] = h5file[f"trace{spin.strip()}/{s}{i}"][()]
                for j in range(i + 1, dim):
                    trace_matrix[i, j] = h5file[f"trace{spin.strip()}/{s}p{i}{j}"][()]
                    trace_matrix[j, i] = h5file[f"trace{spin.strip()}/{s}m{i}{j}"][()]
        
            s_ = '' if s == 'n' else s
            np.savetxt(f"{dirname}/{datfname}{s_}{spin.strip()}.dat", trace_matrix)

    h5file.close()
