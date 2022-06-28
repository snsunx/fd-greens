"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

import pickle
import numpy as np

from fd_greens import (
    initialize_hdf5,
    ResponseFunction,
    get_lih_hamiltonian,
    copy_simulation_data,
    plot_response_function,
    process_bitstring_counts,
    HARTREE_TO_EV)

def main_process():
    """Processes bitstring counts and save to file."""

    # Load circuit labels and bitstring counts from files.
    pkl_data = pickle.load(open(f'resp_3A_run0621_0.pkl', 'rb'))
    labels = pkl_data['full']['labels']
    counts = pkl_data['full']['results']
    confusion_matrix = np.load(f'response_greens_0621_0.npy')

    # Initialize HDF5 files to store processed bitstring counts.
    fname = f'lih_resp_expt'
    initialize_hdf5(fname, mode='resp', create_datasets=True)
    copy_simulation_data(fname, 'lih_resp_sim', 'resp')

    for label, histogram in zip(labels, counts):
        if len(label) == 9: # e.g. circ0d/xx
            pad_zero = 'end'
        else:
            pad_zero = ''

        process_bitstring_counts(histogram, pad_zero=pad_zero, fname=fname, dset_name=label, counts_name='counts')

        process_bitstring_counts(histogram, pad_zero=pad_zero, confusion_matrix=confusion_matrix, fname=fname, 
                                 dset_name=label, counts_name='counts_miti')


def main_generate():
    hamiltonian = get_lih_hamiltonian(3.0)
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV
    for fname in ['lih_resp_expt']:
        if 'exact' in fname:
            resp = ResponseFunction(hamiltonian, fname=fname, method='exact')
        else:
            resp = ResponseFunction(hamiltonian, fname=fname, method='tomo')
        resp.response_function(omegas, eta)

def main_plot():
    h5fnames = ['lih_resp_exact', 'lih_resp_expt', 'lih_resp_expt']
    suffixes = ['', '', '_miti']
    labels = ['Exact', 'Expt', 'Expt (miti)']
    plot_response_function(h5fnames, suffixes, labels=labels, text="legend", dirname=f"figs/data")

if __name__ == '__main__':
    main_generate()
    # main_plot()