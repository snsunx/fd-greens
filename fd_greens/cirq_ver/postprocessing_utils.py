"""
================================================================
Postprocessing Utilities (:mod:`fd_greens.postprocessing_utils`)
================================================================
"""

import os
import pickle
from typing import Optional, Tuple, List
from itertools import product
from deprecated import deprecated

import json
import h5py
import numpy as np

from .general_utils import reverse_qubit_order, histogram_to_array
from .helpers import initialize_hdf5

__all__ = ["process_all_bitstring_counts", "process_all_bitstring_counts_by_depth"]

@deprecated
def restrict_to_qubit_subspace(
    arr: np.ndarray, circ_label: Optional[str] = None, reverse: bool = True
) -> np.ndarray:
    """Restricts the bitstrings counts to the qubit subspace.
    
    Specifically, this functions does the following:
    
    1. Discard the bitstrings with 2 in ternary representation.

    2. Reverses the qubit order due to difference in Qiskit and Berkeley qubit ordering.
    
    Args:
        arr: The bitstring counts array.
        circ_label: The circuit label, which determines what qubit order is reversed
        reverse: Whether the qubit order is reversed. Default to True.
    
    Returns:
        arr_new: The processed bitstring counts array.
    """
    # Extract the number of qubits and construct the bitstrings. For certain circuit labels,
    # permute the bitstring elements according to the SWAP gates at the end.
    n_qubits = int(np.log(arr.shape[0]) / np.log(3))
    bitstrings = ["".join(x) for x in product("012", repeat=n_qubits)]
    if circ_label in ["0u", "1u"]:
        # print("bitstring swapped")
        bitstrings = ["".join([x[i] for i in [0, 2, 1]]) for x in bitstrings]
    if circ_label in ["0u1u", "0d1u"]:
        # print("bitstring swapped")
        bitstrings = ["".join([x[i] for i in [0, 1, 3, 2]]) for x in bitstrings]

    # Construct the new array by taking only the bitstrings that don't contain 2.
    arr_new = []
    for i, x in enumerate(bitstrings):
        if "2" not in x:
            arr_new.append(arr[i])
    arr_new = np.array(arr_new)
    if reverse:
        arr_new = reverse_qubit_order(arr_new)
    return arr_new

@deprecated
def process_berkeley_results(
    h5fname: str, circ_label: str, tomo_label: str, counts_name: str
) -> None:
    """Processes the Berkeley hardware results and saves the processed bitstring counts 
    in the HDF5 file.
    
    Args:
        h5fname: Name of the HDF5 file.
        circ_label: Label of the quantum circuit, e.g. '0u', '0d', '01d'.
        tomo_label: The tomography label, e.g. 'xx', 'xy', 'xz'.
        counts_name: Name of the bitstring counts.
    """
    h5file = h5py.File(h5fname + ".h5", "r+")

    # Obtain the experimental bitstring counts with suffix _exp.
    dset = h5file[f"circ{circ_label}/{tomo_label}"]
    counts_exp = dset.attrs[f"{counts_name}_exp"]

    # Process the results and saves to a new bitstring counts attributes with suffix _exp_proc.
    counts_exp_proc = restrict_to_qubit_subspace(counts_exp, circ_label=circ_label)
    dset.attrs[f"{counts_name}_exp_proc"] = counts_exp_proc

@deprecated
def compute_tvd(h5fname: str, circ_label: str, counts_name: str) -> float:
    """Calculates the average total variational distance (TVD) of a circuit.
    
    Args:
        h5fname: The HDF5 file name.
        circ_label: The circuit label.
        counts_name: Name of the bitstring counts.
    """
    h5file = h5py.File(h5fname + ".h5", "r")
    tomo_labels = ["".join(x) for x in product("xyz", repeat=2)]

    tvds = []
    for tomo_label in tomo_labels:
        dset = h5file[f"circ{circ_label}/{tomo_label}"]
        counts = dset.attrs[counts_name]
        counts_norm = counts / np.sum(counts)
        counts_exp = dset.attrs[f"{counts_name}_exp_proc"]
        counts_exp_norm = counts_exp / np.sum(counts_exp)

        tvd = np.sum(np.abs(counts_norm - counts_exp_norm)) / 2
        tvds.append(tvd)

    tvd_avg = np.average(tvds)
    return tvd_avg

@deprecated
def process_bitstring_counts(
    histogram: dict,
    confusion_matrix: Optional[np.ndarray] = None,
    pad_zero: str = '',
    fname: Optional[str] = None,
    dset_name: Optional[str] = None,
    counts_name: Optional[str] = None,
    mitigation_before_restriction: bool = True
) -> Optional[np.ndarray]:
    """Processes bitstring counts.
    
    Args:
        histogram: The histogram of the bitstring counts.
        confusion_matrix: The confusion matrix for readout error mitigation.
        pad_zero: Where to pad zeros in the bitstrings. ``'front'`` or ``'end'``
            for three qubits, ``''`` for four qubits.
        fname: The HDF5 file name to save the processed bitstring counts to.
        dset_name: The dataset name.
        counts_name: The bitstring counts name.

    Returns:
        array (Optional): The bitstring counts array if ``fname`` is not given.
    """
    assert pad_zero in ['', 'front', 'end']
    
    n_qubits = len(list(histogram.keys())[0])
    array = histogram_to_array(histogram, n_qubits=n_qubits, base=3)

    # Qubit subspace indices.
    if pad_zero == 'front':
        indices = [int('0' + ''.join(x), 3) for x in product('01', repeat=n_qubits - 1)]
    elif pad_zero == 'end':
        indices = [int(''.join(x) + '0', 3) for x in product('01', repeat=n_qubits - 1)]
    else:
        indices = [int(''.join(x), 3) for x in product('01', repeat=n_qubits)]

    # Perform readout error mitigation if confusion matrix is given.
    if confusion_matrix is not None and mitigation_before_restriction:
        array = np.linalg.lstsq(confusion_matrix, array)[0]

    # Restrict the bitstring array to qubit subspace.
    array = array[indices]
    array = array / np.sum(array)

    # Perform readout error mitigation if confusion matrix is given.
    if confusion_matrix is not None and not mitigation_before_restriction:
        confusion_matrix = confusion_matrix[indices][:, indices]
        array = np.linalg.lstsq(confusion_matrix, array)[0]

    if fname is not None and dset_name is not None and counts_name is not None:
        with h5py.File(fname + '.h5', 'r+') as h5file:
            h5file[dset_name].attrs[counts_name] = array
    else:
        return array

def copy_simulation_data(fname_expt: str, fname_exact: str, calculation_type: str = "greens") -> None:
    """Copy ground- and excited-states simulation data to experimental HDF5 file.
    
    Args:
        fname_expt: The experimental HDF5 file name.
        fname_exact: The simulation HDF5 file name.
        calculation_type: The calculation type. Either ``'greens'`` or ``'resp'``.
    """
    assert calculation_type in ['greens', 'resp']

    h5file_exact = h5py.File(fname_exact + '.h5', 'r')
    h5file_expt = h5py.File(fname_expt + '.h5', 'r+')

    # Copy ground state energy over.
    if 'gs' in h5file_expt:
        del h5file_expt['gs']
    h5file_expt['gs/energy'] = h5file_exact['gs/energy'][()]
    
    # Copy excited state energy and states over.
    if 'es' in h5file_expt:
        del h5file_expt['es']    
    if calculation_type== 'greens':
        h5file_expt['es/energies_e'] = h5file_exact['es/energies_e'][:]
        h5file_expt['es/energies_h'] = h5file_exact['es/energies_h'][:]
        h5file_expt['es/states_e'] = h5file_exact['es/states_e'][:]
        h5file_expt['es/states_h'] = h5file_exact['es/states_h'][:]
    else:
        h5file_expt['es/energies'] = h5file_exact['es/energies'][:]
        h5file_expt['es/states'] = h5file_exact['es/states'][:]

    h5file_expt.close()
    h5file_exact.close()


def get_subspace_indices(
    is_diagonal: bool,
    zero_padding_location: str = "front",
    n_qubits: int = 4,
    base: int = 2
) -> List[int]:
    """Returns the subspace indices with or without zero padding.
    
    Args:
        is_diagonal: Whether the circuit is a diagonal circuit and needs zero padding.
        pad_zero: Whether to pad zeros in front or end.
        n_qubits: Total number of qubits. Defaults to 4.
        base: The base to convert the subspace indices.

    Returns:
        subspace_indices: The subspace indices.
    """
    assert zero_padding_location in ["front", "end"]
    
    if is_diagonal:
        subspace_indices = [''.join(x) for x in product("01", repeat=n_qubits - 1)]
        if zero_padding_location == 'front':
            subspace_indices = [int('0' + x, base) for x in subspace_indices]
        elif zero_padding_location == 'end':
            subspace_indices = [int(x + '0', base) for x in subspace_indices]
        else:
            raise ValueError("zero_padding_location must be specified when the circuit is diagonal")
    else:
        subspace_indices = [int(''.join(x), base) for x in product("01", repeat=n_qubits)]
    return subspace_indices

def readout_error_mitigation(
    counts_array: np.ndarray,
    confusion_matrix: np.ndarray,
    mitigation_first: bool = True,
    normalize: bool = True
) -> np.ndarray:
    return

def process_all_bitstring_counts(
    h5fname_expt: str,
    h5fname_exact: str,
    pklfname: str,
    # pkldsetname: str = 'full',
    npyfname: Optional[str] = None,
    calculation_type: str = "greens",
    counts_name: str = "counts",
    counts_miti_name: str = "counts_miti",
    zero_padding_location: str = "end", 
    # mitigation_first: bool = False,
    # normalize_counts: bool = True
) -> None:
    """Processes all bitstring counts.
    
    Args:
        h5fname_expt: The experimental HDF5 file name.
        h5fname_exact: The exact HDF5 file name.
        pklfname: Name of the pkl file that contains the experimental data.
        pkldsetname: Name of the dataset in the pkl file.
        mode: The mode of computation, either ``'greens'`` or ``'resp'``.
        npyfname: Name of the npy file that contains the confusion matrix.
        counts_name: Name of the bitstring counts.
        counts_name_miti: Name of the bitstring counts after error mitigation.
        pad_zero: Whether to pad zeros when not all qubits are used.
        mitigation_first: Whether to perform error mitigation first.
        normalize_counts: Whether to normalize bitstring counts.
    """
    assert zero_padding_location in ["front", "end"]
    assert calculation_type in ["greens", "resp"]

    print("> Start processing results.")

    if not os.path.exists(h5fname_expt + ".h5"):
        if calculation_type == "greens":
            initialize_hdf5(h5fname_expt, spin="ud", mode=calculation_type)
        else:
            initialize_hdf5(h5fname_expt, mode=calculation_type)

    # Load circuits, circuit labels and bitstring counts from files.
    with open(pklfname + ".pkl", "rb") as f:
        pkl_data = pickle.load(f)
    # circuits = pkl_data[pkldsetname]['circs']
    labels = pkl_data["labels"]
    results = pkl_data["results"]

    # Load the confusion matrix if npyfname is given.
    if npyfname is not None:
        confusion_matrix = np.load(npyfname + ".npy")
        if confusion_matrix.shape == (16, 16):
            base = 2
        else:
            base = 3
    else:
        confusion_matrix = None

    # Initialize HDF5 files to store processed bitstring counts.
    copy_simulation_data(h5fname_expt, h5fname_exact, calculation_type=calculation_type)
    h5file = h5py.File(h5fname_expt + '.h5', 'r+')

    for dsetname, histogram in zip(labels, results):
        print(f"> Processing circuit {dsetname}.")

        if dsetname in h5file:
            del h5file[dsetname]
        dset = h5file.create_dataset(dsetname, shape=())
        n_qubits = len(list(histogram.keys())[0])
        counts_array = histogram_to_array(histogram, n_qubits=n_qubits, base=2)
        dset.attrs[counts_name] = counts_array

        # subspace_indices = get_subspace_indices(
        #     len(dsetname) == 9, # XXX
        #     zero_padding_location=zero_padding_location,
        #     n_qubits=n_qubits,
        #     base=base
        # )

        # # If confusion_matrix is given, calculate the mitigated bitstring array counts_array_miti. 
        # if confusion_matrix is not None:
        #     counts_array_miti = readout_error_mitigation(counts_array, confusion_matrix)

        #     if mitigation_first:
        #         counts_array_miti = np.linalg.lstsq(confusion_matrix, counts_array)[0]
        #         counts_array_miti = counts_array_miti[subspace_indices]
        #     else:
        #         confusion_matrix_subspace = confusion_matrix[subspace_indices][:, subspace_indices]
        #         counts_array_miti = counts_array[subspace_indices]
        #         counts_array_miti = np.linalg.lstsq(confusion_matrix_subspace, counts_array_miti)[0]
            
        #     if normalize_counts:
        #         counts_array_miti /= np.sum(counts_array_miti)
        
        # Slice and normalize (unmitigated) bitstring counts.
        # counts_array = counts_array[subspace_indices]
        
        # Save the processed bitstring counts to HDF5 file.
        if confusion_matrix is not None:
            counts_array_miti = np.linalg.lstsq(confusion_matrix, counts_array)[0]
            dset.attrs[counts_miti_name] = counts_array_miti

    h5file.close()

    print("> Finish processing results.")

def process_all_bitstring_counts_by_depth(
    h5fname_expt: str,
    pklfname: str,
    npyfname: Optional[str] = None,
    counts_name: str = "counts",
    counts_miti_name: str = "counts_miti",
) -> None:

    if not os.path.exists(h5fname_expt + ".h5"):
        initialize_hdf5(h5fname_expt, mode="resp")

    # Load circuits, circuit labels and bitstring counts from .pkl file.
    with open(pklfname + ".pkl", "rb") as f:
        pkl_data = pickle.load(f)
    labels = pkl_data["labels"]
    results = pkl_data["results"]

    # Load the confusion matrix if npyfname is given.
    if npyfname is not None:
        confusion_matrix = np.load(npyfname + ".npy")
        if confusion_matrix.shape == (16, 16):
            base = 2
        else:
            base = 3
    else:
        confusion_matrix = None

    # Initialize HDF5 file to store processed bitstring counts.
    h5file_expt = h5py.File(h5fname_expt + ".h5", "r+")

    # # Copy exact statevectors over.
    # if "psi" in h5file_expt:
    #     del h5file_expt["psi"]
    # with h5py.File(h5fname_exact + ".h5", "r") as h5file_exact:
    #     for key in h5file_exact["psi"].keys():
    #         h5file_expt[f"psi/{key}"] = h5file_exact[f"psi/{key}"][:]
    #     # h5file_expt["psi"] = h5file_exact["psi"]

    for dsetname, histogram in zip(labels, results):
        print(f"> Processing circuit {dsetname}")
        if dsetname not in h5file_expt:
            dset = h5file_expt.create_dataset(dsetname, shape=())
        else:
            dset = h5file_expt[dsetname]
        
        n_qubits = len(list(histogram.keys())[0])        
        counts_array = histogram_to_array(histogram, n_qubits=n_qubits, base=base)
        dset.attrs[counts_name] = counts_array

        if confusion_matrix is not None:
            print("confusion_matrix.shape = ", confusion_matrix.shape)
            print("counts_array.shape = ", counts_array.shape)
            if counts_array.shape[0] == 16: # XXX: circ3 is not of shape 16 by 16
                counts_array_miti = np.linalg.lstsq(confusion_matrix, counts_array)[0]
            dset.attrs[counts_miti_name] = counts_array_miti

    h5file_expt.close()
