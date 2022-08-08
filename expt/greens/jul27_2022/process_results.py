import sys
import argparse

sys.path.append('../../..')

from fd_greens.cirq_ver.helpers import process_all_bitstring_counts

def main():
    pklfname = 'greens_nox_0727_run0'
    h5fname_exact = "nah_greens_exact"

    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', type=str, nargs='+')
    parser.add_argument('-d', type=str, dest='pkldsetname')
    args = parser.parse_args()

    for h5fname_expt in args.fnames:
        process_all_bitstring_counts(h5fname_expt, h5fname_exact, pklfname, pad_zero=None, pkldsetname=args.pkldsetname)

if __name__ == '__main__':
    main()
