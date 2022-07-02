import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts


def main():
    h5fname_expt = 'lih_resp_expt'
    h5fname_exact = 'lih_resp_exact'
    pklfname = 'resp_3A_run0622_0'
    npyfname = 'response_greens_0622_0'
    process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, npyfname, mode='resp')

if __name__ == '__main__':
    main()
