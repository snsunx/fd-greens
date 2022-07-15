"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

from fd_greens import plot_response_function

def main():
    h5fnames = ['lih_resp_exact', 'lih_resp_tomo', 'lih_resp_tomo2q']
    suffixes = ['', '', '']
    labels = ['Exact', 'Tomo', 'Tomo 2Q']
    plot_response_function(h5fnames, suffixes, labels=labels, text="legend", dirname=f"figs/data")

if __name__ == '__main__':
    main()
