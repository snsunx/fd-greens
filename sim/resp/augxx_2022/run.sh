#!/bin/bash

create_hdf5=false
generate_data=false
plot_data=true

if [ $create_hdf5 = true ]
then
    export CONVERT_CCZ_TO_ITOFFOLI=1
    python3 create_hdf5.py nah_resp_exact    
    python3 create_hdf5.py nah_resp_tomo --method tomo

    export CONVERT_CCZ_TO_ITOFFOLI=0
    # python3 create_hdf5.py nah_resp_exact2q
    # python3 create_hdf5.py nah_resp_tomo2q --method tomo

    # python3 create_hdf5.py nah_resp_exact -c circ0u1d
    # python3 create_hdf5.py nah_resp_exact -c circ0u1d -n ../../../expt/params/gate_fidelities_0720
    
    # python3 create_hdf5.py nah_resp_exact2q -c circ0u1d
    # python3 create_hdf5.py nah_resp_exact2q -c circ0u1d -n ../../../expt/params/gate_fidelities_0720
fi

if [ $generate_data = true ]
then
    python3 generate_data.py --observable nah_resp_exact nah_resp_tomo
    # python3 generate_data.py -f nah_resp_exact_circ0u1d   nah_resp_exact_circ0u1dn
    # python3 generate_data.py -f nah_resp_exact2q_circ0u1d nah_resp_exact2q_circ0u1dn
fi

if [ $plot_data = true ]
then
    python3 plot_data.py nah_resp_exact nah_resp_tomo --labels EXACT TOMO
fi
