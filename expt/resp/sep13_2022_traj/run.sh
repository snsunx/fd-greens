#!/bin/bash

process_results=false
generate_data=false
plot_data=true

if [ $process_results = true ]
then
    for fname in nah_resp_circ0u1u nah_resp_circ0u1u2q kh_resp_circ0u1u kh_resp_circ0u1u2q
    do
        python3 process_results.py $fname ${fname}_results
    done
    # python3 process_results.py kh_resp_circ0u1u    kh_by_depth_0813
    # python3 process_results.py kh_resp_circ0u1u2q  kh_2q_by_depth_0813
fi

if [ $generate_data = true ]
then
    for h5fname in nah_resp_circ0u1u nah_resp_circ0u1u2q kh_resp_circ0u1u kh_resp_circ0u1u2q
    do
        python3 generate_data.py $h5fname
    done
fi

if [ $plot_data = true ]
then
    python3 plot_data.py fid_vs_depth_nah_resp_circ0u1u fid_vs_depth_nah_resp_circ0u1u2q
    python3 plot_data.py fid_vs_depth_kh_resp_circ0u1u  fid_vs_depth_kh_resp_circ0u1u2q
fi
