import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_fidelity_vs_depth, generate_response_function, generate_fidelity_matrix, generate_trace_matrix

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", dest="observable_fnames")
    parser.add_argument("--fidelity", dest="fidelity_fname")
    parser.add_argument("--fidelity-matrix", nargs=2, dest="fidelity_matrix_fnames")
    parser.add_argument("--trace-matrix", dest="trace_matrix_fname")
    args = parser.parse_args()

    if args.observable_fnames is not None:
        generate_response_function(args.observable_fnames)

    if args.fidelity_fname is not None:
        generate_fidelity_vs_depth(args.fidelity_fname)

    if args.fidelity_matrix_fnames is not None:
        generate_fidelity_matrix(*args.fidelity_matrix_fnames)

    if args.trace_matrix_fname is not None:
        generate_trace_matrix(args.trace_matrix_fname)

if __name__ == '__main__':
    main()
