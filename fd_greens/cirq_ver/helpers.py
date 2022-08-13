"""
==================================
Helpers (:mod:`fd_greens.helpers`)
==================================
"""

import os
from typing import List, Any
from itertools import product

import cirq
import h5py
import numpy as np

__all__ = [
    "get_circuit_labels",
    "initialize_hdf5",
    "save_to_hdf5",
    "save_data_to_file",
    "print_circuit"
]

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

def save_to_hdf5(
    h5file: h5py.File,
    dsetname: str,
    data: Any,
    overwrite: bool = True,
    return_dataset: bool = False
) -> None:
    """Saves a dataset to an HDF5 file.
    
    Args:
        h5file: The HDF5 file.
        dsetname: The dataset name.
        data: The data to be saved.
        overwrite: Whether to overwrite the dataset.
        return_dataset: Whether to return the dataset.
    """
    if overwrite:
        if dsetname in h5file:
            del h5file[dsetname]

    if return_dataset:
        dset = h5file.create_dataset(dsetname, data=data)
        return dset
    else:
        h5file[dsetname] = data

def save_data_to_file(dirname: str, fname: str, data: np.ndarray) -> None:
    """Saves a numpy array to a text file.
    
    Args:
        dirname: THe directory name.
        fname: The file name.
        data: The data array.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savetxt(f"{dirname}/{fname}.dat", data)

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