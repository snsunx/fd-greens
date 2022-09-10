#!/bin/bash

create_hdf5=false
generate_data=true
plot_data=false

if [ $create_hdf5 = true ]
then
    export CONVERT_CCZ_TO_ITOFFOLI=1
    python3 create_hdf5.py nah_greens_exact
    python3 create_hdf5.py nah_greens_tomo --method tomo
    python3 create_hdf5.py kh_greens_exact
    python3 create_hdf5.py kh_greens_tomo --method tomo

    export CONVERT_CCZ_TO_ITOFFOLI=0
    python3 create_hdf5.py nah_greens_exact2q
    python3 create_hdf5.py nah_greens_tomo2q --method tomo
    python3 create_hdf5.py kh_greens_exact2q
    python3 create_hdf5.py kh_greens_tomo2q --method tomo
fi

if [ $generate_data = true ]
then    
    for h5fname in nah_greens_exact kh_greens_exact
    do
        python3 generate_data.py --observable $h5fname
        python3 generate_data.py --trace $h5fname
    done

    # export PROJECT_DENSITY_MATRICES=0
    # export PURIFY_DENSITY_MATRICES=0
    # export USE_EXACT_TRACES=0

    for h5fname in nah_greens_tomo nah_greens_tomo2q
    do
        python3 generate_data.py --observable $h5fname
        python3 generate_data.py --fidelity nah_greens_exact $h5fname
        python3 generate_data.py --trace $h5fname
    done

    for h5fname in kh_greens_tomo kh_greens_tomo2q
    do
        python3 generate_data.py --observable $h5fname
        python3 generate_data.py --fidelity nah_greens_exact $h5fname
        python3 generate_data.py --trace $h5fname
    done

fi

if [ $plot_data = true ]
then
    python3 plot_data.py nah_greens_exact nah_greens_tomo nah_greens_exact2q nah_greens_exact2q --labels Exact Tomo Exact2q Tomo2q --figname A_nah
    python3 plot_data.py kh_greens_exact kh_greens_tomo kh_greens_exact2q kh_greens_tomo --labels Exact Tomo Exact2q Tomo2q --figname A_kh
fi
