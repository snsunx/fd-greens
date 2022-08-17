import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_response_function, generate_fidelity_matrix, generate_trace_matrix

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-obs", "--observable", nargs="+", type=str, dest="observable_fnames", default=None)
    parser.add_argument("-f", "--fidelity-matrix", nargs=2, type=str, dest="fidelity_matrix_fnames", default=None)
    parser.add_argument("-t", "--trace-matrix", type=str, dest="trace_matrix_fname", default=None)
    args = parser.parse_args()

    if args.observable_fnames is not None:
        print("Calling generate_response_function")
        generate_response_function(args.observable_fnames)

    if args.fidelity_matrix_fnames is not None:
        print("Calling generate_fidelity_matrix")
        generate_fidelity_matrix(*args.fidelity_matrix_fnames)

    if args.trace_matrix_fname is not None:
        print("Calling generate_trace_matrix")
        generate_trace_matrix(args.trace_matrix_fname)

if __name__ == '__main__':
    main()
