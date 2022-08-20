import sys
sys.path.append('../../..')
import argparse

from fd_greens.cirq_ver.postprocessing_utils import process_all_bitstring_counts_by_depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname_expt", type=str)
    # parser.add_argument("h5fname_exact", type=str)
    parser.add_argument("pklfname", type=str)
    parser.add_argument("--npyfname", type=str)
    args = parser.parse_args()
    print(args)

    process_all_bitstring_counts_by_depth(
        args.h5fname_expt,
        # args.h5fname_exact,
        args.pklfname,
        npyfname=args.npyfname)

if __name__ == '__main__':
    main()
