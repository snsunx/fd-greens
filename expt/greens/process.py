import h5py
from typing import Optional
from itertools import product
import numpy as np
import sys
sys.path.append('../../fd_greens')
from utils import reverse_qubit_order

bitstrings = [''.join(x) for x in product('012', repeat=3)]

def main_process(h5fname: str, circ_label: str, tomo_label: str, counts_name: str) -> None:
    """Processes the Berkeley hardware results and saves the processed bitstring counts 
    in the HDF5 file.
    """
    h5file = h5py.File(h5fname + '.h5', 'r+')
    dset = h5file[f'circ{circ_label}/{tomo_label}'].attrs
    counts_d = np.array(dset[counts_name], dtype=int)
    counts_d_exp = dset[f'{counts_name}_exp']
    counts_d_exp_proc = take_01(counts_d_exp, circ_label=circ_label)
    dset[f'{counts_name}_exp_proc'] = counts_d_exp_proc

if __name__ == '__main__':
    h5fnames = ['lih', 'lih_run2']
    tomo_labels = [''.join(x) for x in product('xyz', repeat=2)] 
    counts_name = 'counts_noisy'

    for h5fname in h5fnames:
        for circ_label in circ_labels:
            print(circ_label)
            for tomo_label in tomo_labels:
                main_process(h5fname, circ_label, tomo_label, counts_name)
