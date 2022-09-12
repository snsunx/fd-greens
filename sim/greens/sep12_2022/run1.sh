#!/bin/bash

process_results=false
generate_data=true
plot_data=false

if [ $process_results = true ]
then
    for h5fname_expt in nah_greens_tomo_raw nah_greens_tomo_pur nah_greens_tomo_trace
    do
        python3 process_results.py $h5fname_expt nah_greens_exact --pkldsetname base_nah
    done

    for h5fname_expt in nah_greens_tomo2q_raw nah_greens_tomo2q_pur nah_greens_tomo2q_trace
    do
        python3 process_results.py $h5fname_expt nah_greens_exact --pkldsetname 2q_nah
    done

    for h5fname_expt in kh_greens_tomo_raw kh_greens_tomo_pur kh_greens_tomo_trace
    do
        python3 process_results.py $h5fname_expt kh_greens_exact --pkldsetname base_kh
    done

    for h5fname_expt in kh_greens_tomo2q_raw kh_greens_tomo2q_pur kh_greens_tomo2q_trace
    do
        python3 process_results.py $h5fname_expt kh_greens_exact --pkldsetname 2q_kh
    done
fi

if [ $generate_data = true ]
then
    for h5fname in nah_greens_exact kh_greens_exact
    do
        python3 generate_data.py --observable $h5fname
    done

    # export PROJECT_DENSITY_MATRICES=0
    # export PURIFY_DENSITY_MATRICES=0
    # export USE_EXACT_TRACES=0
    # for h5fname in nah_greens_tomo_raw nah_greens_tomo2q_raw
    # do
    #     python3 generate_data.py --observable $h5fname
    #     python3 generate_data.py --fidelity nah_greens_exact $h5fname
    # done

    # for h5fname in kh_greens_tomo_raw kh_greens_tomo2q_raw
    # do
    #     python3 generate_data.py --observable $h5fname
    #     python3 generate_data.py --fidelity kh_greens_exact $h5fname
    # done

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=0
    for h5fname in nah_greens_tomo_pur nah_greens_tomo2q_pur
    do
        python3 generate_data.py --observable $h5fname
        python3 generate_data.py --fidelity nah_greens_exact $h5fname
    done

    for h5fname in kh_greens_tomo_pur kh_greens_tomo2q_pur
    do
        python3 generate_data.py --observable $h5fname
        python3 generate_data.py --fidelity kh_greens_exact $h5fname
    done

    # export PROJECT_DENSITY_MATRICES=1
    # export PURIFY_DENSITY_MATRICES=1
    # export USE_EXACT_TRACES=1
    # for h5fname in nah_greens_tomo_trace nah_greens_tomo2q_trace kh_greens_tomo_trace kh_greens_tomo2q_trace
    # do
    #     python3 generate_data.py ---observable $h5fname
    # done
fi

if [ $plot_data = true ]
then
    echo "Plotting data"
    python3 plot_data.py nah_greens_exact nah_greens_tomo_pur   nah_greens_tomo2q_pur   -n nah_pur
    python3 plot_data.py nah_greens_exact nah_greens_tomo_trace nah_greens_tomo2q_trace -n nah_trace
    python3 plot_data.py kh_greens_exact  kh_greens_tomo_pur    kh_greens_tomo2q_pur    -n kh_pur
    python3 plot_data.py kh_greens_exact  kh_greens_tomo_trace  kh_greens_tomo2q_trace  -n kh_trace
fi
