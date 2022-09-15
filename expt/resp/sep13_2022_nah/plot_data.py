import sys
sys.path.append("../../..")
import argparse

from fd_greens import plot_response_function, display_fidelities, display_traces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "--observable", type=str, nargs="+", dest="observable_fnames", default=None)
    parser.add_argument("-l", "--labels", nargs="+", default=None)
    parser.add_argument("-n", "--figname", type=str)
    parser.add_argument("-f", "--fidelity-matrix", type=str, dest="fidelity_matrix_fname")
    parser.add_argument("-t", "--trace_matrix", type=str, nargs=2, dest="trace_matrix_fnames")
    args = parser.parse_args()

    print(args)

    if args.observable_fnames is not None:
        plot_response_function(args.observable_fnames, labels=args.labels, figname=args.figname)

    if args.fidelity_matrix_fname is not None:
        display_fidelities(args.fidelity_matrix_fname)

    if args.trace_matrix_fnames is not None:
        display_traces(*args.trace_matrix_fnames)

if __name__ == "__main__":
    main()
