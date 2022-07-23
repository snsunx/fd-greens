#!/bin/bash

generate_data=true
plot_data=false

# Generate data
if [ $generate_data = true ]
then
    echo "Generating data"
    python3 generate_data.py nah_resp_exact kh_resp_exact

    export PROJECT_DENSITY_MATRICES=0
    export PURIFY_DENSITY_MATRICES=0
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_resp_tomo_raw nah_resp_tomo2q_raw 
    python3 generate_data.py kh_resp_tomo_raw kh_resp_tomo2q_raw

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_resp_tomo_pur nah_resp_tomo2q_pur
    python3 generate_data.py kh_resp_tomo_pur kh_resp_tomo2q_pur

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=1
    python3 generate_data.py nah_resp_tomo_trace nah_resp_tomo2q_trace
    python3 generate_data.py kh_resp_tomo_trace kh_resp_tomo2q_trace
fi

if [ $plot_data = true ]
then
    echo "Plotting data"
    python3 plot_data.py nah_resp_exact nah_resp_tomo_pur   nah_resp_tomo2q_pur   -n nah_pur_chi
    python3 plot_data.py nah_resp_exact nah_resp_tomo_trace nah_resp_tomo2q_trace -n nah_trace_chi
    python3 plot_data.py kh_resp_exact  kh_resp_tomo_pur    kh_resp_tomo2q_pur    -n kh_pur_chi
    python3 plot_data.py kh_resp_exact  kh_resp_tomo_trace  kh_resp_tomo2q_trace  -n kh_trace_chi
fi
