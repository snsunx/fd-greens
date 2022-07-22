"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')
import argparse

from fd_greens import plot_response_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5fnames', type=str, nargs='+')
    parser.add_argument('-n', type=str, dest='figname')
    args = parser.parse_args()

    # h5fnames = ['nah_resp_exact', 'nah_resp_tomo_pur', 'nah_resp_tomo2q_pur']
    suffixes = ['', '', '']
    labels = ['Exact', 'iToffoli', 'CZ']
    plot_response_function(args.h5fnames, suffixes, labels=labels, text="legend", dirname=f"figs/data", figname=args.figname)

if __name__ == '__main__':
    main()
