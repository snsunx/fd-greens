import sys
sys.path.append('../../..')
import argparse

from fd_greens import process_all_bitstring_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname_expt")
    parser.add_argument("h5fname_exact")
    parser.add_argument("pklfname")
    parser.add_argument("--npyfname")
    args = parser.parse_args()
    print(args)

    process_all_bitstring_counts(
        args.h5fname_expt,
        args.h5fname_exact,
        args.pklfname,
        calculation_mode="resp",
        npyfname=args.npyfname
    )

if __name__ == '__main__':
    main()
