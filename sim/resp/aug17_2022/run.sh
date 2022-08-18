#!/bin/bash

create_hdf5=false
generate_data=false
delete_datasets=true

if [ $create_hdf5 = true ]
then
    export CONVERT_CCZ_TO_ITOFFOLI=1
    # python3 create_hdf5.py nah_resp_exact  
    # python3 create_hdf5.py kh_resp_exact  
    python3 create_hdf5.py nah_resp_alltomo --method alltomo
    python3 create_hdf5.py kh_resp_alltomo  --method alltomo

    # python3 create_hdf5.py nah_resp_exact -c circ0u1d
    # python3 create_hdf5.py kh_resp_exact  -c circ0u1d
    # python3 create_hdf5.py nah_resp_exact -c circ0u1d -n ../../../expt/params/gate_fidelities_0814
    # python3 create_hdf5.py kh_resp_exact  -c circ0u1d -n ../../../expt/params/gate_fidelities_0814

    export CONVERT_CCZ_TO_ITOFFOLI=0
    # python3 create_hdf5.py nah_resp_exact2q
    # python3 create_hdf5.py kh_resp_exact2q
    python3 create_hdf5.py nah_resp_alltomo2q --method alltomo
    python3 create_hdf5.py kh_resp_alltomo2q  --method alltomo

    # python3 create_hdf5.py nah_resp_exact2q -c circ0u1d
    # python3 create_hdf5.py kh_resp_exact2q  -c circ0u1d
    # python3 create_hdf5.py nah_resp_exact2q -c circ0u1d -n ../../../expt/params/gate_fidelities_0814
    # python3 create_hdf5.py kh_resp_exact2q  -c circ0u1d -n ../../../expt/params/gate_fidelities_0814
fi

if [ $generate_data = true ]
then
    python3 generate_data.py -f nah_resp_circ0u1d
    python3 generate_data.py -f nah_resp_circ0u1d2q
    python3 generate_data.py -f kh_resp_circ0u1d
    python3 generate_data.py -f kh_resp_circ0u1d2q
    # python3 generate_data.py -f nah_resp_circ0u1d_n0814
    # python3 generate_data.py -f nah_resp_circ0u1d2q_n0814
fi

if [ $delete_datasets = true ]
then
    # python3 delete_datasets.py nah_resp_alltomo
    # python3 delete_datasets.py kh_resp_alltomo
    # python3 delete_datasets.py nah_resp_alltomo2q
    # python3 delete_datasets.py kh_resp_alltomo2q
    python3 delete_datasets.py nah_resp_circ0u1d2q
fi