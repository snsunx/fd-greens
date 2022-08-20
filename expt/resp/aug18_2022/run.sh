#!/bin/bash

process_results=true
generate_data=true
plot_data=false

if [ $process_results = true ]
then
    # python3 process_results.py nah_resp_circ0u1d   nah_resp_alltomorun_comb    # --npyfname ../../params/response_greens_0814
    python3 process_results.py nah_resp_circ0u1d2q nah_resp_alltomo2qrun_comb # --npyfname ../../params/response_greens_0814
    # python3 process_results.py kh_resp_circ0u1d    kh_resp_alltomorun_comb
    # python3 process_results.py kh_resp_circ0u1d2q  kh_resp_alltomo2qrun_comb
fi

if [ $generate_data = true ]
then
    # python3 generate_data.py nah_resp_circ0u1d
    python3 generate_data.py nah_resp_circ0u1d2q
    # python3 generate_data.py kh_resp_circ0u1d
    # python3 generate_data.py kh_resp_circ0u1d2q
fi

if [ $plot_data = true ]
then
    python3 plot_data.py fid_vs_depth_nah_resp_circ0u1d_expt fid_vs_depth_nah_resp_circ0u1d2q_expt
    python3 plot_data.py fid_vs_depth_kh_resp_circ0u1d_expt  fid_vs_depth_kh_resp_circ0u1d2q_expt
fi
