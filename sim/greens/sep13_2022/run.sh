#!/bin/bash

create_hdf5=false
generate_data=false
plot_data=true

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
        # python3 generate_data.py --trace $h5fname
    done

    # export PROJECT_DENSITY_MATRICES=0
    # export PURIFY_DENSITY_MATRICES=0
    # export USE_EXACT_TRACES=0

    for h5fname in nah_greens_tomo nah_greens_tomo2q
    do
        python3 generate_data.py --observable $h5fname
        # python3 generate_data.py --fidelity nah_greens_exact $h5fname
        # python3 generate_data.py --trace $h5fname
    done

    for h5fname in kh_greens_tomo kh_greens_tomo2q
    do
        python3 generate_data.py --observable $h5fname
        # python3 generate_data.py --fidelity kh_greens_exact $h5fname
        # python3 generate_data.py --trace $h5fname
    done

fi

if [ $plot_data = true ]
then
    python3 plot_data.py --observable nah_greens_exact nah_greens_tomo nah_greens_tomo2q --labels Exact Tomo "Tomo 2Q" --figname A_nah
    python3 plot_data.py --observable kh_greens_exact  kh_greens_tomo  kh_greens_tomo2q  --labels Exact Tomo "Tomo 2Q" --figname A_kh
fi
