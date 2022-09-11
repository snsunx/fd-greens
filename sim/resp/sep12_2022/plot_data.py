"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')
import argparse

from fd_greens import plot_response_function, display_fidelities, display_traces, plot_fidelity_vs_depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", nargs="+", dest="observable_fnames")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--fidelity", nargs="+", dest="fidelity_fnames")
    parser.add_argument("--figname")
    parser.add_argument("--fidelity-matrix", dest="fidelity_matrix_fname")
    parser.add_argument("--trace-matrix", nargs=2, dest="trace_matrix_fnames")
    args = parser.parse_args()

    if args.observable_fnames is not None:
        plot_response_function(args.h5fnames, labels=args.labels, text="legend", dirname=f"figs/data")

    if args.fidelity_fnames is not None:
        plot_fidelity_vs_depth(*args.fidelity_fnames)
    
    if args.fidelity_matrix_fname is not None:
        display_fidelities(args.fidelity_matrix_fname)

    if args.trace_matrix_fnames is not None:
        display_traces(*args.trace_matrix_fnames)

if __name__ == '__main__':
    main()
