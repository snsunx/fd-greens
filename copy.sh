#!/bin/bash

function copy_fidelity_and_trace {
    cp ../../expt/resp/sep13_2022_$1/data/mat/fid_mat_$1_resp_tomo_raw.dat .
    cp ../../expt/resp/sep13_2022_$1/data/mat/fid_mat_$1_resp_tomo_rc_raw.dat .
    cp ../../expt/resp/sep13_2022_$1/data/mat/fid_mat_$1_resp_tomo_pur.dat .
    cp ../../expt/resp/sep13_2022_$1/data/mat/fid_mat_$1_resp_tomo_rc_pur.dat .
}

function copy_response_function {
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_exact_chi00.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_exact_chi01.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo_pur_chi00.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo2q_pur_chi00.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo_rc_pur_chi00.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo2q_rc_pur_chi00.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo_pur_chi01.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo2q_pur_chi01.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo_rc_pur_chi01.dat .
    cp ../../expt/resp/sep13_2022_$1/data/obs/$1_resp_tomo2q_rc_pur_chi01.dat .
}

if [ $(basename `pwd`) == fig5_fidelity_and_trace ]; then
    copy_fidelity_and_trace nah
elif [ $(basename `pwd`) == figs2_fidelity_and_trace_kh ]; then
    copy_fidelity_and_trace kh
fi

if [ $(basename `pwd`) == fig6_response_function ]; then
    copy_response_function nah
elif [ $(basename `pwd`) == figs3_response_function_kh ]; then
    copy_response_function kh
fi
