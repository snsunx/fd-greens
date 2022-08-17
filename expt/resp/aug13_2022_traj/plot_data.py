import sys
sys.path.append("../../..")
import argparse

from fd_greens import plot_fidelity_vs_depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", type=str, nargs=2)
    args = parser.parse_args()

    plot_fidelity_vs_depth(*args.fnames)

if __name__ == "__main__":
    main()
