"""Process May 2022 results and save to HDF5 files."""

import sys

sys.path.append('../../..')

import pickle
import numpy as np
import cirq

from fd_greens import histogram_to_array, initialize_hdf5, CircuitStringConverter
from fd_greens.cirq_ver.postprocessing import process_bitstring_counts

import matplotlib.pyplot as plt

def process_0524(is_567):
    """Processes bitstring counts and save to file. is_567 is a flag to indicate whether the 3-qubit 
    circuits are run on qubits 4, 5, 6 or qubits 5, 6, 7."""

    if not is_567:
        results = pickle.load(open('qtrl_collections_3A_run0524_2_CZCS.pkl', 'rb'))
        fname = 'lih_3A_expt'
    else:
        results = pickle.load(open('qtrl_collections_3A_run0524_2_567_CZCS.pkl', 'rb'))
        fname = 'lih_3A_expt_567'
    initialize_hdf5(fname, spin='u', create_datasets=True)
    initialize_hdf5(fname, spin='d', create_datasets=True)
    confusion_matrix = np.load('response_greens_0524.npy')

    for label, histogram in zip(results['full']['labels'], results['full']['results']):
        if len(label) == 9: # e.g. circ0d/xx
            if is_567:
                pad_zero = 'front'
            else:
                pad_zero = 'end'
        else:
            pad_zero = ''

        process_bitstring_counts(
            histogram,
            pad_zero=pad_zero,
            fname=fname,
            dset_name=label,
            counts_name='counts'
        )

        process_bitstring_counts(
            histogram,
            pad_zero=pad_zero,
            confusion_matrix=confusion_matrix,
            fname=fname,
            dset_name=label,
            counts_name='counts_miti'
        )

def process_0527():
    """Process bitstring counts from the run on May 27 and save to HDF5 file."""

    data = pickle.load(open('qtrl_collections_3A_run0527_0_CZCS.pkl', 'rb'))
    confusion_matrix = np.load('response_greens_0527.npy')

    fname = 'lih_3A_expt_0527'
    initialize_hdf5(fname, spin='u', create_datasets=True)
    initialize_hdf5(fname, spin='d', create_datasets=True)

    for label, histogram in zip(data['full']['labels'], data['full']['results']):
        if len(label) == 9:
            pad_zero = 'end'
        else:
            pad_zero = ''
        
        process_bitstring_counts(histogram, pad_zero=pad_zero, fname=fname, dset_name=label, counts_name='counts')
        process_bitstring_counts(histogram, pad_zero=pad_zero, confusion_matrix=confusion_matrix, fname=fname, 
                                 dset_name=label, counts_name='counts_miti')




if __name__ == '__main__':
    # process_0524(True)
    # process_0524(False)
    process_0527()
