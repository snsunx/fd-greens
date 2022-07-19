"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

from fd_greens import plot_response_function

def main():
    h5fnames = ['nah_resp_exact', 'nah_resp_tomo', 'nah_resp_tomo2q']
    suffixes = ['', '', '']
    labels = ['Exact', 'Tomo', 'Tomo 2Q']
    plot_response_function(h5fnames, suffixes, labels=labels, text="legend", dirname=f"figs/data", figname="nah_chi")

    h5fnames = ['kh_resp_exact', 'kh_resp_tomo', 'kh_resp_tomo2q']
    suffixes = ['', '', '']
    labels = ['Exact', 'Tomo', 'Tomo 2Q']
    plot_response_function(h5fnames, suffixes, labels=labels, text="legend", dirname=f"figs/data", figname="kh_chi")

if __name__ == '__main__':
    main()
