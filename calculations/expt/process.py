import h5py
from typing import Optional
from itertools import product
import numpy as np
import sys
sys.path.append('../../src/')
from utils import reverse_qubit_order

bitstrings = [''.join(x) for x in product('012', repeat=3)]

def take_01(arr: np.ndarray, circ_label: Optional[str] = None) -> np.ndarray:
    """Takes the bitstrings with 0 and 1 in a bitstring counts and reverse the qubit order."""
    # Extract the number of qubits and construct the bitstrings. For certain circuit labels, 
    # permute the bitstring elements according to the SWAP gates at the end.
    n_qubits = int(np.log(arr.shape[0]) / np.log(3))
    bitstrings = [''.join(x) for x in product('012', repeat=n_qubits)]
    if circ_label in ['0u', '1u']:
        print('bitstring swapped')
        bitstrings = [''.join([x[i] for i in [0, 2, 1]]) for x in bitstrings]
    if circ_label in ['0u1u', '0d1u']:
        print('bitstring swapped')
        bitstrings = [''.join([x[i] for i in [0, 1, 3, 2]]) for x in bitstrings]

    # Construct the new array by taking only the bitstrings that don't contain 2.
    arr_new = []
    for i, x in enumerate(bitstrings):
        if '2' not in x:
            arr_new.append(arr[i])
    arr_new = np.array(arr_new)
    arr_new = reverse_qubit_order(arr_new)
    return arr_new

def main_process(h5fname: str, circ_label: str, tomo_label: str, counts_name: str) -> None:
    """Processes the Berkeley hardware results and saves the processed bitstring counts 
    in the HDF5 file.
    
    Args:
        h5fname: Name of the HDF5 file.
        circ_label: Label of the quantum circuit, e.g. '0u', '0d', '01d'.
        tomo_label: The tomography label, e.g. 'xx', 'xy', 'xz'.
        counts_name: Name of the bitstring counts.
    """
    h5file = h5py.File(h5fname + '.h5', 'r+')
    dset = h5file[f'circ{circ_label}/{tomo_label}'].attrs
    counts_d = np.array(dset[counts_name], dtype=int)
    counts_d_exp = dset[f'{counts_name}_exp']
    counts_d_exp_proc = take_01(counts_d_exp, circ_label=circ_label)
    dset[f'{counts_name}_exp_proc'] = counts_d_exp_proc

if __name__ == '__main__':
    h5fnames = ['lih', 'lih_run2']
    # circ_labels = ['0d', '1d', '01d']
    circ_labels = ['0u', '0d', '1u', '1d', '0u0d', '0u1u', '0u1d', '0d1u', '0d1d', '1u1d']
    tomo_labels = [''.join(x) for x in product('xyz', repeat=2)] 
    counts_name = 'counts_noisy'

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            print(circ_label)
            for tomo_label in tomo_labels:
                main_process(h5fname, circ_label, tomo_label, counts_name)
