"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append("../../..")
import argparse

from fd_greens import plot_response_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fnames", type=str, nargs="+")
    parser.add_argument("-n", "--figname", type=str)
    parser.add_argument("-s", "--suffixes", nargs="+", default=None)
    parser.add_argument("-l", "--labels", nargs="+", default=None)
    args = parser.parse_args()

    plot_response_function(
        args.h5fnames,
        suffixes=args.suffixes,
        labels=args.labels,
        text="legend",
        dirname=f"figs/data",
        figname=args.figname)

if __name__ == "__main__":
    main()
