#!/bin/bash

process_results=false
generate_data=false
plot_data=true

if [ $process_results = true ]
then
    python3 process_results.py nah_resp_tomo_noem_raw nah_resp_tomo_noem_pur nah_resp_tomo_noem_trace -d raw
    python3 process_results.py nah_resp_tomo_em_raw   nah_resp_tomo_em_pur   nah_resp_tomo_em_trace   -d em
fi

# Generate data
if [ $generate_data = true ]
then
    python3 generate_data.py nah_resp_exact kh_resp_exact

    export PROJECT_DENSITY_MATRICES=0
    export PURIFY_DENSITY_MATRICES=0
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_resp_tomo_noem_raw nah_resp_tomo_em_raw

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_resp_tomo_noem_pur nah_resp_tomo_em_pur

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=1
    python3 generate_data.py nah_resp_tomo_noem_trace nah_resp_tomo_em_trace
fi

if [ $plot_data = true ]
then
    python3 plot_data.py nah_resp_exact nah_resp_tomo_noem_pur   nah_resp_tomo_em_pur   -n nah_pur
    python3 plot_data.py nah_resp_exact nah_resp_tomo_noem_trace nah_resp_tomo_em_trace -n nah_trace
fi
