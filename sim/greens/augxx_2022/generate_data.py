import sys
sys.path.append('../../..')
import argparse

from fd_greens import generate_greens_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname")
    args = parser.parse_args()

    generate_greens_function(args.h5fname)

if __name__ == '__main__':
    main()
