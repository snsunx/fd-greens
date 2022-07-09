"""Process Jun 2022 results and save to HDF5 files."""

import sys
sys.path.append('../../..')

from fd_greens import plot_response_function

def main():
    h5fnames = ['lih_resp_exact', 'lih_resp_expt', 'lih_resp_pur', 'lih_resp_trace']
    suffixes = ['', '', '', '']
    labels = ['Exact', 'Raw', 'Pur', 'Trace']
    plot_response_function(h5fnames, suffixes, labels=labels, text="legend", dirname=f"figs/data")

if __name__ == '__main__':
    main()
