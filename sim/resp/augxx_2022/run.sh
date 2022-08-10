#!/bin/bash

create_files=true
generate_data=false

if [ $create_files = true ]
then
    # python3 create_hdf5.py nah_resp_exact
    # python3 create_hdf5.py nah_resp_exact -c circ0u1d
    python3 create_hdf5.py nah_resp_exact -c circ0u1d -n ../../../expt/params/gate_fidelities_0708
fi

if [ $generate_data = true ]
then
    python3 generate_data.py -f nah_resp_exact_circ0u1d nah_resp_exact_circ0u1dn
fi