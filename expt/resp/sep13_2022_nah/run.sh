#!/bin/bash

process_results=false
generate_data=false
plot_data=true

if [ $process_results = true ]
then
    for h5fname in nah_resp_tomo_raw nah_resp_tomo_pur nah_resp_tomo_trace
    do
        python3 process_results.py $h5fname nah_resp_exact nah_resp_tomo_results
    done

    for h5fname in nah_resp_tomo2q_raw nah_resp_tomo2q_pur nah_resp_tomo2q_trace
    do
        python3 process_results.py $h5fname nah_resp_exact nah_resp_tomo2q_results
    done

    for h5fname in nah_resp_tomo_rc_raw nah_resp_tomo_rc_pur nah_resp_tomo_rc_trace
    do
        python3 process_results.py $h5fname nah_resp_exact nah_resp_tomo_results_RC
    done

    for h5fname in nah_resp_tomo2q_rc_raw nah_resp_tomo2q_rc_pur nah_resp_tomo2q_rc_trace
    do
        python3 process_results.py $h5fname nah_resp_exact nah_resp_tomo2q_results_RC
    done
fi

if [ $generate_data = true ]
then
    # python3 generate_data.py --observable  nah_resp_exact
    # python3 generate_data.py --trace-matrix nah_resp_exact
    
    # Generate raw data
    # export PROJECT_DENSITY_MATRICES=0
    # export PURIFY_DENSITY_MATRICES=0
    # export USE_EXACT_TRACES=0
    # for h5fname in nah_resp_tomo_raw nah_resp_tomo2q_raw nah_resp_tomo_rc_raw nah_resp_tomo2q_rc_raw
    # do
    #     python3 generate_data.py --observable $h5fname
    #     python3 generate_data.py --fidelity-matrix nah_resp_exact $h5fname
    #     python3 generate_data.py --trace-matrix $h5fname
    # done

    # Generate purified data
    # export PROJECT_DENSITY_MATRICES=1
    # export PURIFY_DENSITY_MATRICES=1
    # export USE_EXACT_TRACES=0
    # for h5fname in nah_resp_tomo_pur nah_resp_tomo2q_pur nah_resp_tomo_rc_pur nah_resp_tomo2q_rc_pur
    # do
    #     python3 generate_data.py --observable      $h5fname
    #     python3 generate_data.py --fidelity-matrix nah_resp_exact $h5fname
    # done

    # Generate trace corrected data
    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=1
    for h5fname in nah_resp_tomo_trace nah_resp_tomo2q_trace nah_resp_tomo_rc_trace nah_resp_tomo2q_rc_trace
    do 
        python3 generate_data.py --observable $h5fname
    done
fi

if [ $plot_data = true ]
then
    python3 plot_data.py \
        -O nah_resp_exact nah_resp_tomo_raw nah_resp_tomo_pur nah_resp_tomo_trace \
        -l Exact Raw Purified Trace \
        -n nah_resp_tomo

    python3 plot_data.py \
        -O nah_resp_exact nah_resp_tomo2q_raw nah_resp_tomo2q_pur nah_resp_tomo2q_trace \
        -l Exact Raw Purified Trace \
        -n nah_resp_tomo2q

    python3 plot_data.py \
        -O nah_resp_exact nah_resp_tomo_rc_raw nah_resp_tomo_rc_pur nah_resp_tomo_rc_trace \
        -l Exact Raw Purified Trace \
        -n nah_resp_tomo_rc

    python3 plot_data.py \
        -O nah_resp_exact nah_resp_tomo2q_rc_raw nah_resp_tomo2q_rc_pur nah_resp_tomo2q_rc_trace \
        -l Exact Raw Purified Trace \
        -n nah_resp_tomo2q_rc
    
    # for datfname in fid_mat_resp_tomo_raw fid_mat_resp_tomo_pur fid_mat_resp_tomo2q_raw fid_mat_resp_tomo2q_pur \
    #                 fid_mat_resp_tomo_rc_raw fid_mat_resp_tomo_rc_pur fid_mat_resp_tomo2q_rc_raw fid_mat_resp_tomo2q_rc_pur
    # do        
    #     python3 plot_data.py -f $datfname
    # done

    # for datfname in trace_mat_resp_exact trace_mat_resp_tomo_raw trace_mat_resp_tomo2q_raw \
    #                 trace_mat_resp_tomo_rc_raw trace_mat_resp_tomo2q_rc_raw
    # do
    #     python3 plot_data.py -t trace_mat_resp_exact $datfname
    # done
fi
