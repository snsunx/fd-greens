#!/bin/bash

process_results=false
generate_data=false
plot_data=true

if [ $process_results = true ]
then
    # python3 process_results.py nah_resp_circ0u1d   nah_by_depth_0813
    python3 process_results.py nah_resp_circ0u1d2q nah_2q_by_depth_0813 --npyfname ../../params/response_greens_0814
    # python3 process_results.py kh_resp_circ0u1d    kh_by_depth_0813
    # python3 process_results.py kh_resp_circ0u1d2q  kh_2q_by_depth_0813
fi

if [ $generate_data = true ]
then
    # python3 generate_data.py nah_resp_circ0u1d_expt
    python3 generate_data.py nah_resp_circ0u1d2q
    # python3 generate_data.py kh_resp_circ0u1d_expt
    # python3 generate_data.py kh_resp_circ0u1d2q_expt
fi

if [ $plot_data = true ]
then
    python3 plot_data.py fid_vs_depth_nah_resp_circ0u1d_expt fid_vs_depth_nah_resp_circ0u1d2q_expt
    python3 plot_data.py fid_vs_depth_kh_resp_circ0u1d_expt  fid_vs_depth_kh_resp_circ0u1d2q_expt
fi
