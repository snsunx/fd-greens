"""
================================================================
Postprocessing Utilities (:mod:`fd_greens.postprocessing_utils`)
================================================================
"""

import os
import pickle

import h5py
from typing import Optional
from itertools import product

import numpy as np
from deprecated import deprecated

from .general_utils import reverse_qubit_order, histogram_to_array
from .helpers import initialize_hdf5

__all__ = ["process_all_bitstring_counts"]

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

def copy_simulation_data(fname_expt: str, fname_exact: str, mode: str = 'greens') -> None:
    """Copy ground- and excited-states simulation data to experimental HDF5 file.
    
    Args:
        fname_expt: The experimental HDF5 file name.
        fname_exact: The simulation HDF5 file name.
        mode: The calculation mode. Either ``'greens'`` or ``'resp'``.
    """
    assert mode in ['greens', 'resp']

    h5file_exact = h5py.File(fname_exact + '.h5', 'r')
    h5file_expt = h5py.File(fname_expt + '.h5', 'r+')

    if 'gs' in h5file_expt:
        del h5file_expt['gs']
    h5file_expt['gs/energy'] = h5file_exact['gs/energy'][()]
    
    if 'es' in h5file_expt:
        del h5file_expt['es']    
    if mode == 'greens':
        h5file_expt['es/energies_e'] = h5file_exact['es/energies_e'][:]
        h5file_expt['es/energies_h'] = h5file_exact['es/energies_h'][:]
        h5file_expt['es/states_e'] = h5file_exact['es/states_e'][:]
        h5file_expt['es/states_h'] = h5file_exact['es/states_h'][:]
    else:
        h5file_expt['es/energies'] = h5file_exact['es/energies'][:]
        h5file_expt['es/states'] = h5file_exact['es/states'][:]

    h5file_expt.close()
    h5file_exact.close()

def process_all_bitstring_counts(
    h5fname_expt: str,
    h5fname_exact: str,
    pklfname: str,
    pkldsetname: str = 'full',
    npyfname: Optional[str] = None,
    mode: str = 'greens',
    counts_name: str = 'counts',
    counts_name_miti: str = 'counts_miti',
    pad_zero: str = 'end', 
    mitigation_first: bool = False,
    normalize_counts: bool = True
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
    assert pad_zero in ['front', 'end', None]
    assert mode in ['greens', 'resp']

    print("Start processing results.")

    if not os.path.exists(h5fname_expt + '.h5'):
        if mode == 'greens':
            initialize_hdf5(h5fname_expt, spin='ud', mode=mode)
        else:
            initialize_hdf5(h5fname_expt, mode=mode)

    # Load circuits, circuit labels and bitstring counts from files.
    with open(pklfname + '.pkl', 'rb') as f:
        pkl_data = pickle.load(f)
    circuits = pkl_data[pkldsetname]['circs']
    labels = pkl_data[pkldsetname]['labels']
    results = pkl_data[pkldsetname]['results']

    # Load the confusion matrix if npyfname is given.
    confusion_matrix = np.load(npyfname + '.npy') if npyfname is None else None

    # Initialize HDF5 files to store processed bitstring counts.
    copy_simulation_data(h5fname_expt, h5fname_exact, mode=mode)
    h5file = h5py.File(h5fname_expt + '.h5', 'r+')

    for qtrl_strings, dset_name, histogram in zip(circuits, labels, results):
        print(f"> Processing circuit {dset_name}.")

        # Save circuit to HDF5 file.
        if dset_name in h5file:
            del h5file[dset_name]
        dset = h5file.create_dataset(dset_name, data=json.dumps(qtrl_strings))

        n_qubits = len(list(histogram.keys())[0])
        counts_array = histogram_to_array(histogram, n_qubits=n_qubits, base=3)
        # print(f"{n_qubits = }, {counts_array.shape = }")
        
        # Obtain the subspace indices.
        if pad_zero and len(dset_name) == 9:
            if pad_zero == 'front':
                subspace_indices = [int('0' + ''.join(x), 3) for x in product('01', repeat=n_qubits - 1)]
                subspace_indices_base2 = [int('0' + ''.join(x), 2) for x in product('01', repeat=n_qubits - 1)]
            elif pad_zero == 'end':
                subspace_indices = [int(''.join(x) + '0', 3) for x in product('01', repeat=n_qubits - 1)]
                subspace_indices_base2 = [int(''.join(x) + '0', 2) for x in product('01', repeat=n_qubits - 1)]
        else:
            subspace_indices = [int(''.join(x), 3) for x in product('01', repeat=n_qubits)]
            subspace_indices_base2 = [int(''.join(x), 2) for x in product('01', repeat=n_qubits)]


        # If confusion_matrix is given, calculate counts_array_miti first. 
        # Otherwise counts_array will be modified into the subspace.
        if confusion_matrix is not None:
            # print("Confusion matrix dimension", confusion_matrix.shape)
            if mitigation_first:
                counts_array_miti = np.linalg.lstsq(confusion_matrix, counts_array)[0]
                counts_array_miti = counts_array_miti[subspace_indices]
            else:
                counts_array_miti = counts_array[subspace_indices]
                if confusion_matrix.shape[0] == 16:
                    confusion_matrix1 = confusion_matrix[subspace_indices_base2][:, subspace_indices_base2]
                else: # 81 by 81
                    confusion_matrix1 = confusion_matrix[subspace_indices][:, subspace_indices]

                # print(f"{confusion_matrix1.shape = }")
                # print(f"{counts_array_miti.shape = }")
                counts_array_miti = np.linalg.lstsq(confusion_matrix1, counts_array_miti)[0]
            
            if normalize_counts:
                counts_array_miti /= np.sum(counts_array_miti)
        
        counts_array = counts_array[subspace_indices]
        if normalize_counts:
            counts_array /= np.sum(counts_array)
        # print(f"{counts_array.shape = }")
        
        # Save the processed bitstring counts to HDF5 file.
        dset.attrs[counts_name] = counts_array
        if confusion_matrix is not None:
            dset.attrs[counts_name_miti] = counts_array_miti

    h5file.close()

    print("Finish processing results.")
