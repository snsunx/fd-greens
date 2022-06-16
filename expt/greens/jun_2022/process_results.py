"""Process Jun 2022 results and save to HDF5 files."""

import sys

sys.path.append('../../..')

import pickle
import numpy as np
import cirq

from fd_greens import histogram_to_array, initialize_hdf5, CircuitStringConverter
from fd_greens.cirq_ver.postprocessing import process_bitstring_counts

import matplotlib.pyplot as plt

def process_0613(run_id):
    """Processes bitstring counts and save to file."""

    # Load circuit labels and bitstring counts from files.
    pkl_data = pickle.load(open(f'greens_3A_run0613_{run_id}.pkl', 'rb'))
    labels = pkl_data['full']['labels']
    counts = pkl_data['full']['results']
    confusion_matrix = np.load(f'response_greens_0613_{run_id}.npy')

    # Initialize HDF5 files to store processed bitstring counts.
    fname = f'lih_3A_expt{run_id}'
    initialize_hdf5(fname, spin='u', create_datasets=True)
    initialize_hdf5(fname, spin='d', create_datasets=True)

    for label, histogram in zip(labels, counts):
        if len(label) == 9: # e.g. circ0d/xx
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

if __name__ == '__main__':
    for run_id in [1, 2, 3]:
        process_0613(run_id)
