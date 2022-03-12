import h5py
from itertools import product
import numpy as np
import sys
sys.path.append('../../src/')
from utils import reverse_qubit_order

bitstrings = [''.join(x) for x in product('012', repeat=3)]

def take_01(arr, circ_label=None):
    n_qubits = int(np.log(arr.shape[0]) / np.log(3))
    bitstrings = [''.join(x) for x in product('012', repeat=n_qubits)]
    if circ_label in ['0u', '1u']:
        print('bitstring swapped')
        bitstrings = [''.join([x[i] for i in [0, 2, 1]]) for x in bitstrings]
    if circ_label in ['0u1u', '0d1u']:
        print('bitstring swapped')
        bitstrings = [''.join([x[i] for i in [0, 1, 3, 2]]) for x in bitstrings]

    arr_new = []
    for i, x in enumerate(bitstrings):
        if '2' not in x:
            arr_new.append(arr[i])
    arr_new = np.array(arr_new)
    arr_new = reverse_qubit_order(arr_new)
    return arr_new

def main_process(h5fname, circ_label, tomo_label, counts_name):
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
