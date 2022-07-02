import sys
sys.path.append('../../..')

import pickle
import numpy as np

from fd_greens import plot_spectral_function, plot_trace_self_energy


def main():
    print("Start plotting data.")

    h5fnames = ['lih_3A_exact', 'lih_3A_expt', 'lih_3A_pur']
    suffixes = ['', '', '']
    labels = ['Exact', 'Expt', 'Pur']

    plot_spectral_function(h5fnames, suffixes, labels=labels, text="legend", dirname="figs/data")
    plot_trace_self_energy(h5fnames, suffixes, labels=labels, text="legend", dirname="figs/data")

    print("Finished plotting data.")


if __name__ == '__main__':
    main()
