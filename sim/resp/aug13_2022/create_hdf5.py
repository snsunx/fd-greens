import sys
sys.path.append('../../..')
import argparse

from fd_greens import create_hdf5, create_hdf5_by_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname", type=str)
    parser.add_argument("--method", type=str, default="exact")
    parser.add_argument("-n", "--noise", type=str, dest="noise_fname", default=None)
    parser.add_argument("-c", "--circuit", type=str, dest="circuit_name", default=None)
    parser.add_argument("-r", "--repetitions", type=int, default=10000)
    args = parser.parse_args()

    print(args)

    if args.circuit_name is None:
        print("Calling create_hdf5")
        create_hdf5(args.h5fname, method=args.method, noise_fname=args.noise_fname, repetitions=args.repetitions)
    else:
        print("Calling create_hdf5_by_depth")
        create_hdf5_by_depth(args.h5fname, args.circuit_name, noise_fname=args.noise_fname)

if __name__ == '__main__':
    main()
