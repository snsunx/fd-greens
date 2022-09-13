import sys
sys.path.append('../../..')
import argparse

from fd_greens import create_resp_hdf5, create_resp_hdf5_by_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname")
    parser.add_argument("--method", default="exact")
    parser.add_argument("--noise", dest="noise_fname")
    parser.add_argument("--circuit", dest="circuit_name")
    parser.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()

    if args.circuit_name is None:
        create_resp_hdf5(args.h5fname, method=args.method, noise_fname=args.noise_fname, repetitions=args.repetitions)
    else:
        create_resp_hdf5_by_depth(
            args.h5fname,
            args.circuit_name,
            noise_fname=args.noise_fname,
            repetitions=args.repetitions
        )


if __name__ == '__main__':
    main()
