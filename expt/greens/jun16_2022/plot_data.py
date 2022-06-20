import sys
sys.path.append('../../..')

import pickle
import numpy as np

from fd_greens import initialize_hdf5, GreensFunction, get_lih_hamiltonian
from fd_greens.cirq_ver.postprocessing import process_bitstring_counts
from fd_greens.cirq_ver.helpers import copy_simulation_data, plot_spectral_function, plot_trace_self_energy
from fd_greens.cirq_ver.parameters import HARTREE_TO_EV, linestyles_A, linestyles_TrSigma

def main_process():
    print("Start processing results.")

    # Load circuit labels and bitstring counts from files.
    pkl_data = pickle.load(open(f'greens_3A_run0616_0.pkl', 'rb'))
    labels = pkl_data['full']['labels']
    results = pkl_data['full']['results']
    confusion_matrix = np.load(f'response_greens_0616_0.npy')

    # Initialize HDF5 files to store processed bitstring counts.
    fname = f'lih_3A_expt'
    initialize_hdf5(fname, spin='u', create_datasets=True)
    initialize_hdf5(fname, spin='d', create_datasets=True)
    copy_simulation_data(fname, 'lih_3A_sim')

    for label, histogram in zip(labels, results):
        if len(label) == 9: # e.g. circ0d/xx
            pad_zero = 'end'
        else:
            pad_zero = ''

        process_bitstring_counts(histogram, pad_zero=pad_zero, fname=fname, dset_name=label, counts_name='counts')

        process_bitstring_counts(histogram, pad_zero=pad_zero, confusion_matrix=confusion_matrix, fname=fname, 
                                 dset_name=label, counts_name='counts_miti')

    print("Finished processing results.")



def main_generate():
    print("Start generating data.")
    
    hamiltonian = get_lih_hamiltonian(3.0)
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV
    for spin in ['u', 'd']:
        for fname in ['lih_3A_sim', 'lih_3A_expt']:
            greens = GreensFunction(hamiltonian, fname=fname, method='tomo', spin=spin)
            greens.spectral_function(omegas, eta)
            greens.self_energy(omegas, eta)

    print("Finished generating data.")

def main_plot():
    print("Start plotting data.")

    h5fnames = ['lih_3A_sim', 'lih_3A_expt']
    suffixes = ['', '']
    labels = ['Sim', 'Expt']

    plot_spectral_function(h5fnames, suffixes, linestyles=linestyles_A, labels=labels, text="legend", dirname="figs/data")
    plot_trace_self_energy(h5fnames, suffixes, linestyles=linestyles_TrSigma, labels=labels, text="legend", dirname="figs/data")

    print("Finished plotting data.")


if __name__ == '__main__':
    main_process()
    main_generate()
    main_plot()
