import sys
sys.path.append("../../..")
import argparse

import cirq

from fd_greens import create_greens_hdf5


def main() -> None:
    # qubits = cirq.LineQubit.range(4)
    # h5fname = "nah_greens_2"
    # method = "tomo"

    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname")
    parser.add_argument("--method", default="exact")
    parser.add_argument("--noise", dest="noise_fname")
    parser.add_argument("--circuit", dest="circuit_fname")
    parser.add_argument("--repetitions", default=10000)
    args = parser.parse_args()
    print(args)

    create_greens_hdf5(
        args.h5fname,
        method=args.method, 
        noise_fname=args.noise_fname, 
        repetitions=args.repetitions)

if __name__ == "__main__":
    main()
