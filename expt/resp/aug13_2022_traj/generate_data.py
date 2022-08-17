import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_fidelity_vs_depth

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    args = parser.parse_args()

    generate_fidelity_vs_depth(args.fname)


if __name__ == '__main__':
    main()
