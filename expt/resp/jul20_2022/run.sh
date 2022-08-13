#!/bin/bash

generate_data=true
plot_data=false

# Generate data
if [ $generate_data = true ]
then
    python3 generate_data.py -obs nah_resp_exact # kh_resp_exact
    python3 generate_data.py -t nah_resp_exact

    export PROJECT_DENSITY_MATRICES=0
    export PURIFY_DENSITY_MATRICES=0
    export USE_EXACT_TRACES=0
    for h5fname in nah_resp_tomo_raw nah_resp_tomo2q_raw
    do
        python3 generate_data.py -obs $h5fname
        python3 generate_data.py -f nah_resp_exact $h5fname
        python3 generate_data.py -t $h5fname
    done
    # python3 generate_data.py kh_resp_tomo_raw kh_resp_tomo2q_raw

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=0
    for h5fname in nah_resp_tomo_pur nah_resp_tomo2q_pur
    do
        python3 generate_data.py -obs $h5fname
        python3 generate_data.py -f nah_resp_exact $h5fname
    done
    # python3 generate_data.py kh_resp_tomo_pur kh_resp_tomo2q_pur

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=1
    # python3 generate_data.py kh_resp_tomo_trace kh_resp_tomo2q_trace
    for h5fname in nah_resp_tomo_trace nah_resp_tomo2q_trace
    do
        python3 generate_data.py -obs $h5fname
    done
fi

if [ $plot_data = true ]
then
    python3 plot_data.py -obs nah_resp_exact nah_resp_tomo_pur   nah_resp_tomo2q_pur   \
        -l Exact Pur "Pur 2Q" -n nah_pur_chi
    python3 plot_data.py -obs nah_resp_exact nah_resp_tomo_trace nah_resp_tomo2q_trace \
        -l Exact Trace "Trace 2Q" -n nah_trace_chi
    # python3 plot_data.py kh_resp_exact  kh_resp_tomo_pur    kh_resp_tomo2q_pur    -n kh_pur_chi
    # python3 plot_data.py kh_resp_exact  kh_resp_tomo_trace  kh_resp_tomo2q_trace  -n kh_trace_chi

    for datfname in fid_mat_resp_tomo_raw fid_mat_resp_tomo2q_raw fid_mat_resp_tomo_pur fid_mat_resp_tomo2q_pur
    do
        python3 plot_data.py -f $datfname
    done

    for datfname in trace_mat_resp_tomo_raw # trace_mat_resp_tomo2q_raw
    do
        python3 plot_data.py -t trace_mat_resp_exact $datfname
    done
fi
