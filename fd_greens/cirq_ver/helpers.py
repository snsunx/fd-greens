"""
==================================
Helpers (:mod:`fd_greens.helpers`)
==================================
"""

import pickle
import os
import json
from typing import List, Optional
from itertools import product

import cirq
import h5py
import numpy as np

from .general_utils import histogram_to_array


def get_circuit_labels(n_orbitals: int, mode: str = 'greens', spin: str = '') -> List[str]:
    """Returns the circuit labels of a greens or resp calculation.
    
    Args:
        n_orbitals: Number of orbitals.
        mode: Calculation mode. ``'greens'`` or ``'resp'``.
        spin: Spin states included in the calculation. If ``'greens'``, ``spin`` must be
            ``'u'`` or ``'d'``; if ``'resp'``, ``spin`` must be ``''``.
        
    Returns:
        circuit_labels: A list of strings corresponding to the circuit labels.
    """
    assert mode in ['greens', 'resp']
    if mode == 'greens':
        assert spin in ['u', 'd', 'ud']
    else:
        assert spin == ''

    if spin == 'ud':
        return get_circuit_labels(n_orbitals, spin='u') + get_circuit_labels(n_orbitals, spin='d')

    # For Green's function, orbital labels are just strings of the orbital indices.
    # For response function, orbital labels are orbital indices with 'u' or 'd' suffix.
    if mode == 'greens':
        orbital_labels = [str(i) for i in range(n_orbitals)]
    elif mode == 'resp':
        orbital_labels = list(product(range(n_orbitals), ['u', 'd']))
        orbital_labels = [f'{x[0]}{x[1]}' for x in orbital_labels]

    # Circuit labels include diagonal and off-diagonal combinations of orbital labels.
    circuit_labels = []
    for i in range(len(orbital_labels)):
        circuit_labels.append(f'circ{orbital_labels[i]}{spin}')
        for j in range(i + 1, len(orbital_labels)):
            circuit_labels.append(f'circ{orbital_labels[i]}{orbital_labels[j]}{spin}')
    
    return circuit_labels

def get_tomography_labels(n_qubits: int) -> List[str]:
    """Returns the tomography labels on a given number of qubits.
    
    Args:
        n_qubits: Number of qubits to be tomographed.
        
    Returns:
        tomography_labels: The tomography labels.
    """
    tomography_labels = [''.join(x) for x in product('xyz', repeat=n_qubits)]
    return tomography_labels

def initialize_hdf5(
    fname: str = 'lih',
    mode: str = 'greens',
    spin: str = '',
    n_orbitals: int = 2,
    n_tomography: int = 2,
    create_datasets: bool = False
) -> None:
    """Initializes an HDF5 file for Green's function or response function calculation.
    
    Args:
        fname: The HDF5 file name.
        mode: Calculation mode. Either ``'greens'`` or ``'resp'``.
        spin: Spin of the second-quantized operators.
        n_orbitals: Number of orbitals. Defaults to 2.
        n_tomography: Number of qubits to be tomographed. Defaults to 2.
        overwrite: Whether to overwrite groups if they are found in the HDF5 file.
        create_datasets: Whether to create datasets in the HDF5 file.
    """
    assert mode in ['greens', 'resp']
    if mode == 'greens':
        assert spin in ['u', 'd', 'ud']
    else:
        assert spin == ''

    if spin == 'ud':
        initialize_hdf5(
            fname=fname, 
            mode=mode,
            spin='u',
            n_orbitals=n_orbitals,
            n_tomography=n_tomography,
            create_datasets=create_datasets
        )
        initialize_hdf5(
            fname=fname, 
            mode=mode,
            spin='d',
            n_orbitals=n_orbitals,
            n_tomography=n_tomography,
            create_datasets=create_datasets
        )
        return
    
    h5fname = fname + '.h5'
    if os.path.exists(h5fname):
        h5file = h5py.File(h5fname, 'r+')
    else:
        h5file = h5py.File(h5fname, 'w')

    # Groups contain observable groups and circuit groups.
    circuit_labels = get_circuit_labels(n_orbitals, mode=mode, spin=spin)
    group_names = ['gs', 'es', 'amp', 'psi', 'rho', 'params/circ', 'params/miti'] + circuit_labels

    for group_name in group_names:
        # Create the group if it does not exist. If overwrite is set to True then overwrite the group.
        if group_name not in h5file.keys():
            h5file.create_group(group_name)

        # Create datasets if create_datasets is set to True.
        if create_datasets and group_name not in ['gs', 'es', 'amp']:
            tomography_labels = [''.join(x) for x in product('xyz', repeat=n_tomography)]
            for tomography_label in tomography_labels:
                print(f'Creating {group_name}/{tomography_label} in {fname}.h5.')
                h5file.create_dataset(f'{group_name}/{tomography_label}', data='')
    
    h5file.close()

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
    normalize_counts: bool = True, 
) -> None:
    """Processes all bitstring counts.
    
    Args:
        h5fname_expt: The experimental HDF5 file name.
        h5fname_exact: The exact HDF5 file name.
        pklfname: The pickle file name.
        npyfname: The confusion matrix file name.
        counts_name: Name of the bitstring counts.
        counts_name: Name of the bitstring counts after error mitigation.
        pad_zero: Whether to pad zeros when not all qubits are used.
        mitigation_first: Whether to perform error mitigation first.
        normalize_counts: Whether to normalize bitstring counts.
    """
    assert pad_zero in ['front', 'end']
    assert mode in ['greens', 'resp']

    print("Start processing results.")

    if not os.path.exists(h5fname_expt + '.h5'):
        if mode == 'greens':
            initialize_hdf5(h5fname_expt, spin='ud', mode=mode)
        else:
            initialize_hdf5(h5fname_expt, mode=mode)

    # Load circuit labels and bitstring counts from files.
    pkl_data = pickle.load(open(pklfname + '.pkl', 'rb'))
    circuits = pkl_data[pkldsetname]['circs']
    labels = pkl_data[pkldsetname]['labels']
    results = pkl_data[pkldsetname]['results']

    if npyfname is not None:
        confusion_matrix = np.load(npyfname + '.npy')
    else:
        confusion_matrix = None

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
        
        # Obtain the subspace indices.
        if len(dset_name) == 9:
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
            print("Confusion matrix dimension", confusion_matrix.shape)
            if mitigation_first:
                counts_array_miti = np.linalg.lstsq(confusion_matrix, counts_array)[0]
                counts_array_miti = counts_array_miti[subspace_indices]
            else:
                counts_array_miti = counts_array[subspace_indices]
                if confusion_matrix.shape[0] == 16:
                    confusion_matrix1 = confusion_matrix[subspace_indices_base2][:, subspace_indices_base2]
                else: # 81 by 81
                    confusion_matrix1 = confusion_matrix[subspace_indices][:, subspace_indices]

                print(f"{confusion_matrix1.shape = }")
                print(f"{counts_array_miti.shape = }")
                counts_array_miti = np.linalg.lstsq(confusion_matrix1, counts_array_miti)[0]
            
            if normalize_counts:
                counts_array_miti /= np.sum(counts_array_miti)
        
        counts_array = counts_array[subspace_indices]
        if normalize_counts:
            counts_array /= np.sum(counts_array)
        
        dset.attrs[counts_name] = counts_array

        if confusion_matrix is not None:
            dset.attrs[counts_name_miti] = counts_array_miti

    print("Finished processing results.")

def print_circuit(circuit: cirq.Circuit) -> None:
    """Prints out a circuit 10 elements at a time.
    
    Args:
        circuit: The circuit to be printed.
    """
    if len(circuit) < 10:
        print(circuit)
    else:
        for i in range(len(circuit) // 10 + 1):
            print(circuit[i * 10: min((i + 1) * 10, len(circuit) - 1)], '\n')
            print('-' * 120, '\n')