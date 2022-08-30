import sys
sys.path.append("../../..")
import argparse

from fd_greens import plot_spectral_function

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fnames", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--figname", default="A")
    args = parser.parse_args()
    print(args)

    plot_spectral_function(args.h5fnames, labels=args.labels, figname=args.figname)

if __name__ == "__main__":
    main()