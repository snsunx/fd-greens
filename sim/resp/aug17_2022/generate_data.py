import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_fidelity_vs_depth, generate_response_function

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-obs", "--observable", nargs="+", type=str, dest="observable_fnames", default=None)
    parser.add_argument("-f", "--fidelity", type=str, dest="fidelity_fname", default=None)
    args = parser.parse_args()

    if args.observable_fnames is not None:
        generate_response_function(args.observable_fnames)

    if args.fidelity_fname is not None:
        generate_fidelity_vs_depth(args.fidelity_fname)

if __name__ == '__main__':
    main()
