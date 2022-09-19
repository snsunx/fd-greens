import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_greens_function, generate_fidelity_matrix, generate_trace_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observable", dest="observable_fname")
    # parser.add_argument("--fidelity", nargs=2, dest="fidelity_matrix_fnames")
    # parser.add_argument("--trace", dest="trace_matrix_fname")
    args = parser.parse_args()

    if args.observable_fname is not None:
        generate_greens_function(args.observable_fname)
    
    # if args.fidelity_matrix_fnames is not None:
    #     generate_fidelity_matrix(args.fidelity_matrix_fnames[0], args.fidelity_matrix_fnames[1])

    # if args.trace_matrix_fname is not None:
    #     generate_trace_matrix(args.trace_matrix_fname)

if __name__ == '__main__':
    main()
