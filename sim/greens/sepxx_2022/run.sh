#!/bin/bash

create_hdf5=true
generate_data=false
plot_data=false

if [ $create_hdf5 = true ]
then
    export CONVERT_CCZ_TO_ITOFFOLI=1
    python3 create_hdf5.py nah_greens_exact
    # python3 create_hdf5.py nah_greens_tomo --method tomo
    # python3 create_hdf5.py kh_greens_exact
    # python3 create_hdf5.py kh_greens_tomo --method tomo

    export CONVERT_CCZ_TO_ITOFFOLI=0
    # python3 create_hdf5.py nah_greens_exact2q
    # python3 create_hdf5.py nah_greens_tomo2q --method tomo
    # python3 create_hdf5.py kh_greens_exact2q
    # python3 create_hdf5.py kh_greens_tomo2q --method tomo
fi

if [ $generate_data = true ]
then
    python3 generate_data.py nah_greens_exact   
    python3 generate_data.py nah_greens_tomo
    python3 generate_data.py kh_greens_exact
    python3 generate_data.py kh_greens_tomo
    python3 generate_data.py nah_greens_exact2q
    python3 generate_data.py nah_greens_tomo2q
    python3 generate_data.py kh_greens_exact2q
    python3 generate_data.py kh_greens_tomo2q
fi

if [ $plot_data = true ]
then
    python3 plot_data.py nah_greens_exact nah_greens_tomo nah_greens_exact2q nah_greens_exact2q --labels Exact Tomo Exact2q Tomo2q --figname A_nah
    python3 plot_data.py kh_greens_exact kh_greens_tomo kh_greens_exact2q kh_greens_tomo --labels Exact Tomo Exact2q Tomo2q --figname A_kh
fi
