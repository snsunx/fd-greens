#!/bin/bash

generate_data=true
plot_data=false

if [ $generate_data = true ]
then
    python3 generate_data.py nah_greens_exact kh_greens_exact

    export PROJECT_DENSITY_MATRICES=0
    export PURIFY_DENSITY_MATRICES=0
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_greens_tomo_raw
    # python3 generate_data.py nah_greens_tomo_raw nah_greens_tomo2q_raw 
    # python3 generate_data.py kh_greens_tomo_raw kh_greens_tomo2q_raw

    # export PROJECT_DENSITY_MATRICES=1
    # export PURIFY_DENSITY_MATRICES=1
    # export USE_EXACT_TRACES=0
    # python3 generate_data.py nah_greens_tomo_pur nah_greens_tomo2q_pur
    # python3 generate_data.py kh_greens_tomo_pur kh_greens_tomo2q_pur

    # export PROJECT_DENSITY_MATRICES=1
    # export PURIFY_DENSITY_MATRICES=1
    # export USE_EXACT_TRACES=1
    # python3 generate_data.py nah_greens_tomo_trace nah_greens_tomo2q_trace
    # python3 generate_data.py kh_greens_tomo_trace kh_greens_tomo2q_trace
fi

if [ $plot_data = true ]
then
    echo "Plotting data"
    python3 plot_data.py nah_greens_exact nah_greens_tomo_pur   nah_greens_tomo2q_pur   -n nah_pur
    python3 plot_data.py nah_greens_exact nah_greens_tomo_trace nah_greens_tomo2q_trace -n nah_trace
    python3 plot_data.py kh_greens_exact  kh_greens_tomo_pur    kh_greens_tomo2q_pur    -n kh_pur
    python3 plot_data.py kh_greens_exact  kh_greens_tomo_trace  kh_greens_tomo2q_trace  -n kh_trace
fi
