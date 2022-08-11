import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_fidelity_vs_depth, generate_response_function

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observable", nargs="+", type=str, dest="observable_fnames", default=None)
    parser.add_argument("-f", "--fidelity", nargs=2, type=str, dest="fidelity_fnames", default=None)
    args = parser.parse_args()

    if args.observable_fnames is not None:
        generate_response_function(args.observable_fnames)

    if args.fidelity_fnames is not None:
        generate_fidelity_vs_depth(*args.fidelity_fnames)

if __name__ == '__main__':
    main()
