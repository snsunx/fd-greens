#!/bin/bash

process_results=false
generate_data=false
plot_data=true

if [ $process_results = true ]
then
    python3 process_results.py nah_resp_tomo_rc_raw nah_resp_tomo_rc_pur nah_resp_tomo_rc_trace \
        --dsetname rc
    python3 process_results.py nah_resp_tomo_rcnoxraw_raw nah_resp_tomo_rcnoxraw_pur nah_resp_tomo_rcnoxraw_trace \
        --dsetname rcnoxraw
    python3 process_results.py nah_resp_tomo_rcnoxem_raw nah_resp_tomo_rcnoxem_pur nah_resp_tomo_rcnoxem_trace \
        --dsetname rcnoxem
fi

if [ $generate_data = true ]
then
    python3 generate_data.py -o nah_resp_exact
    python3 generate_data.py -t nah_resp_exact

    export PROJECT_DENSITY_MATRICES=0
    export PURIFY_DENSITY_MATRICES=0
    export USE_EXACT_TRACES=0
    for h5fname in nah_resp_tomo_rc_raw nah_resp_tomo_rcnoxraw_raw nah_resp_tomo_rcnoxem_raw
    do
        python3 generate_data.py -o $h5fname
        python3 generate_data.py -f nah_resp_exact $h5fname
        python3 generate_data.py -t $h5fname
    done

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=0
    for h5fname in nah_resp_tomo_rc_pur nah_resp_tomo_rcnoxraw_pur nah_resp_tomo_rcnoxem_pur
    do
        python3 generate_data.py -o $h5fname
        python3 generate_data.py -f nah_resp_exact $h5fname
    done

    export PROJECT_DENSITY_MATRICES=1
    export PURIFY_DENSITY_MATRICES=1
    export USE_EXACT_TRACES=1
    for h5fname in nah_resp_tomo_rc_trace nah_resp_tomo_rcnoxraw_trace nah_resp_tomo_rcnoxem_trace
    do 
        python3 generate_data.py -o $h5fname
    done
fi

if [ $plot_data = true ]
then
    # python3 plot_data.py \
    #     -o nah_resp_exact nah_resp_tomo_rc_raw nah_resp_tomo_rc_pur nah_resp_tomo_rc_trace \
    #     -l Exact Raw Purified Trace \
    #     -n nah_rc
    # python3 plot_data.py \
    #     -o nah_resp_exact nah_resp_tomo_rcnoxraw_raw nah_resp_tomo_rcnoxraw_pur nah_resp_tomo_rcnoxraw_trace \
    #     -l Exact Raw Purified Trace \
    #     -n nah_rcnoxraw
    # python3 plot_data.py \
    #     -o nah_resp_exact nah_resp_tomo_rcnoxem_raw nah_resp_tomo_rcnoxem_pur nah_resp_tomo_rcnoxem_trace \
    #     -l Exact Raw Purified Trace \
    #     -n nah_rcnoxem
    
    for datfname in fid_mat_resp_tomo_rc_raw fid_mat_resp_tomo_rcnoxraw_raw fid_mat_resp_tomo_rcnoxem_raw \
                    fid_mat_resp_tomo_rc_pur fid_mat_resp_tomo_rcnoxraw_pur fid_mat_resp_tomo_rcnoxem_pur
    do        
        python3 plot_data.py -f $datfname
    done

    for datfname in trace_mat_resp_tomo_rc_raw trace_mat_resp_tomo_rcnoxraw_raw trace_mat_resp_tomo_rcnoxem_raw
    do
        python3 plot_data.py -t trace_mat_resp_exact $datfname
    done
fi
