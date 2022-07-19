#!/bin/bash

python3 create_hdf5.py nah_resp_exact exact
python3 create_hdf5.py nah_resp_tomo2q tomo
python3 create_hdf5.py kh_resp_exact exact
python3 create_hdf5.py kh_resp_tomo2q tomo

