#!/bin/bash

process_results=true
generate_data=true
plot_data=true

if [ $process_results = true ]
then
    python3 process_results.py nah_resp_tomo_rc_raw       nah_resp_tomo_rc_pur       nah_resp_tomo_rc_trace       -d rc
    python3 process_results.py nah_resp_tomo_rcnoxraw_raw nah_resp_tomo_rcnoxraw_pur nah_resp_tomo_rcnoxraw_trace -d rcnoxraw
    python3 process_results.py nah_resp_tomo_rcnoxem_raw  nah_resp_tomo_rcnoxem_pur  nah_resp_tomo_rcnoxem_trace  -d rcnoxem
fi

# Generate data
if [ $generate_data = true ]
then
    python3 generate_data.py nah_resp_exact

    export PROJECT_DENSITY_MATRICES=0
    export PURIFY_DENSITY_MATRICES=0
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_resp_tomo_rc_raw nah_resp_tomo_rcnoxraw_raw nah_resp_tomo_rcnoxem_raw

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=0
    python3 generate_data.py nah_resp_tomo_rc_pur nah_resp_tomo_rcnoxraw_pur nah_resp_tomo_rcnoxem_pur

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=1
    python3 generate_data.py nah_resp_tomo_rc_trace nah_resp_tomo_rcnoxraw_trace nah_resp_tomo_rcnoxem_trace
fi

if [ $plot_data = true ]
then
    python3 plot_data.py nah_resp_exact nah_resp_tomo_rc_raw       nah_resp_tomo_rc_pur       nah_resp_tomo_rc_trace       -n nah_rc
    python3 plot_data.py nah_resp_exact nah_resp_tomo_rcnoxraw_raw nah_resp_tomo_rcnoxraw_pur nah_resp_tomo_rcnoxraw_trace -n nah_rcnoxraw
    python3 plot_data.py nah_resp_exact nah_resp_tomo_rcnoxem_raw  nah_resp_tomo_rcnoxem_pur  nah_resp_tomo_rcnoxem_trace  -n nah_rcnoxem
fi
