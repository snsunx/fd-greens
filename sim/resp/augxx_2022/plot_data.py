"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')
import argparse

from fd_greens import plot_response_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fnames", nargs="+")
    parser.add_argument("-l", "--labels", nargs="+")
    args = parser.parse_args()

    plot_response_function(args.h5fnames, labels=args.labels, text="legend", dirname=f"figs/data")

if __name__ == '__main__':
    main()
