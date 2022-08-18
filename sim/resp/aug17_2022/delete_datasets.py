import argparse
import h5py

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5fname")
    args = parser.parse_args()

    h5file = h5py.File(args.h5fname + ".h5", "r+")
    for key in h5file.keys():
        if key[:4] == "circ" and key not in ["circ10", "circ0u1d"]:
            del h5file[key]
    h5file.close()

if __name__ == "__main__":
    main()