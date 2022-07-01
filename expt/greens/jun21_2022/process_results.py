import sys

sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts

def main():
    h5fname_expt = 'lih_3A_expt'
    h5fname_exact = 'lih_3A_exact'
    pklfname = 'greens_3A_run0621_0'
    npyfname = 'response_greens_0621_0'
    process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, npyfname)

if __name__ == '__main__':
    main()
