"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')
import argparse

from fd_greens import plot_spectral_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", nargs="+", dest="observable_fnames")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--figname")
    # parser.add_argument("--fidelity", nargs="+", dest="fidelity_fnames")
    # parser.add_argument("--fidelity-matrix", dest="fidelity_matrix_fname")
    # parser.add_argument("--trace-matrix", nargs=2, dest="trace_matrix_fnames")
    args = parser.parse_args()

    if args.observable_fnames is not None:
        plot_spectral_function(args.observable_fnames, labels=args.labels, text="legend", figname=args.figname)

if __name__ == '__main__':
    main()
