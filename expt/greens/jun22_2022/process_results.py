import sys
sys.path.append('../../..')

import pickle
import numpy as np

from fd_greens import initialize_hdf5, copy_simulation_data, process_bitstring_counts

def main():
    print("Start processing results.")

    # Load circuit labels and bitstring counts from files.
    pkl_data = pickle.load(open(f'greens_3A_run0622_0_batched.pkl', 'rb'))
    labels = pkl_data['full']['labels']
    results = pkl_data['full']['results']
    confusion_matrix = np.load(f'response_greens_0622_0_batched.npy')
    for x in results:
        for k, v in x.items():
            x[k] = sum(v)

    # Initialize HDF5 files to store processed bitstring counts.
    fname = f'lih_3A_expt_batched'
    initialize_hdf5(fname, spin='u', create_datasets=True)
    initialize_hdf5(fname, spin='d', create_datasets=True)
    copy_simulation_data(fname, 'lih_3A_exact')

    for label, histogram in zip(labels, results):
        if len(label) == 9: # e.g. circ0d/xx
            pad_zero = 'end'
        else:
            pad_zero = ''

        process_bitstring_counts(histogram, pad_zero=pad_zero, fname=fname, dset_name=label, counts_name='counts')

        process_bitstring_counts(histogram, pad_zero=pad_zero, confusion_matrix=confusion_matrix, fname=fname, 
                                 dset_name=label, counts_name='counts_miti')

    print("Finished processing results.")

if __name__ == '__main__':
    main()
