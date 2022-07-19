#!/bin/bash

python3 create_hdf5.py nah_greens_exact exact
python3 create_hdf5.py nah_greens_tomo2q tomo
python3 create_hdf5.py kh_greens_exact exact
python3 create_hdf5.py kh_greens_tomo2q tomo
