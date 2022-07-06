import sys

sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts

def main():
    h5fname_exact = 'lih_greens_exact'
    pklfname = 'greens_0701_run0'
    npyfname = 'response_greens_0701_0.npy'

    for h5fname_expt in ['lih_greens_expt', 'lih_greens_pur']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pkldsetname='base')

    for h5fname_expt in ['lih_greens_expt2q', 'lih_greens_pur2q']:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pkldsetname='2q')

if __name__ == '__main__':
    main()
