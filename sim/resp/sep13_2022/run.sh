#!/bin/bash

create_hdf5=true
generate_data=false
plot_data=false

if [ $create_hdf5 = true ]
then
    # export CONVERT_CCZ_TO_ITOFFOLI=1
    python3 create_hdf5.py nah_resp_exact    
    # python3 create_hdf5.py nah_resp_tomo --method tomo
    python3 create_hdf5.py kh_resp_exact
    # python3 create_hdf5.py kh_resp_tomo --method tomo

    # export CONVERT_CCZ_TO_ITOFFOLI=0
    # python3 create_hdf5.py nah_resp_exact2q
    # python3 create_hdf5.py nah_resp_tomo2q --method tomo
    # python3 create_hdf5.py kh_resp_exact2q
    # python3 create_hdf5.py kh_resp_tomo2q --method tomo

    # for h5fname in nah_resp_exact nah_resp_exact2q kh_resp_exact kh_resp_exact2q
    # do
    #     echo "Generating fidelity vs depth circuits for ${h5fname}"
    #     python3 create_hdf5.py $h5fname --circuit circ0u1u
    #     python3 create_hdf5.py $h5fname --circuit circ0u1u --noise ../../../expt/params/gate_fidelities_0708
    # done

    # python3 create_hdf5.py nah_resp_exact --circuit circ0u1u
    # python3 create_hdf5.py nah_resp_exact --circuit circ0u1u --noise ../../../expt/params/gate_fidelities_0708
    
    # python3 create_hdf5.py nah_resp_exact2q --circuit circ0u1u
    # python3 create_hdf5.py nah_resp_exact2q --circuit circ0u1u --noise ../../../expt/params/gate_fidelities_0708

    # python3 create_hdf5.py kh_resp_exact --circuit circ0u1u
    # python3 create_hdf5.py kh_resp_exact --circuit circ0u1u --noise ../../../expt/params/gate_fidelities_0708

    # python3 create_hdf5.py kh_resp_exact2q --circuit circ0u1u
    # python3 create_hdf5.py kh_resp_exact2q --circuit circ0u1u --noise ../../../expt/params/gate_fidelities_0708
fi

if [ $generate_data = true ]
then
    # for h5fname in nah_resp_exact nah_resp_tomo nah_resp_tomo2q kh_resp_exact kh_resp_tomo kh_resp_tomo2q
    # do
    #     python3 generate_data.py --observable $h5fname
    # done

    # python3 generate_data.py --trace-matrix nah_resp_exact 
    # python3 generate_data.py --trace-matrix kh_resp_exact

    # for h5fname in nah_resp_tomo nah_resp_tomo2q
    # do
    #     python3 generate_data.py --fidelity-matrix nah_resp_exact $h5fname
    #     python3 generate_data.py --trace-matrix $h5fname
    # done

    # for h5fname in kh_resp_tomo kh_resp_tomo2q
    # do
    #     python3 generate_data.py --fidelity-matrix kh_resp_exact $h5fname
    #     python3 generate_data.py --trace-matrix $h5fname
    # done


    for h5fname in nah_resp_circ0u1u   nah_resp_circ0u1u_n0708 \
                   nah_resp_circ0u1u2q nah_resp_circ0u1u2q_n0708 \
                   kh_resp_circ0u1u    kh_resp_circ0u1u_n0708 \
                   kh_resp_circ0u1u2q  kh_resp_circ0u1u2q_n0708
    do
        python3 generate_data.py --fidelity $h5fname
    done
    # python3 generate_data.py --fidelity nah_resp_circ0u1u
    # python3 generate_data.py --fidelity nah_resp_circ0u1u_n0708

    # python3 generate_data.py --fidelity nah_resp_circ0u1u2q
    # python3 generate_data.py --fidelity nah_resp_circ0u1u2q_n0708

    # python3 generate_data.py --fidelity kh_resp_circ0u1u
    # python3 generate_data.py --fidelity kh_resp_circ0u1u_n0708

    # python3 generate_data.py --fidelity kh_resp_circ0u1u2q
    # python3 generate_data.py --fidelity kh_resp_circ0u1u2q_n0708
fi

if [ $plot_data = true ]
then
    # python3 plot_data.py nah_resp_exact nah_resp_tomo nah_resp_tomo2q --labels Exact Tomo "Tomo 2Q"
    # python3 plot_data.py kh_resp_exact  kh_resp_tomo  kh_resp_tomo2q  --labels Exact Tomo "Tomo 2Q"
    
    # for datfname in fid_mat_nah_resp_tomo fid_mat_nah_resp_tomo2q fid_mat_kh_resp_tomo fid_mat_kh_resp_tomo2q
    # do
    #     python3 plot_data.py --fidelity-matrix $datfname
    # done

    # for datfname in trace_mat_nah_resp_tomo trace_mat_nah_resp_tomo2q
    # do
    #     python3 plot_data.py --trace-matrix trace_mat_nah_resp_exact $datfname
    # done

    # for datfname in trace_mat_kh_resp_tomo trace_mat_kh_resp_tomo2q
    # do
    #     python3 plot_data.py --trace-matrix trace_mat_kh_resp_exact $datfname
    # done

    python3 plot_data.py --fidelity fid_vs_depth_nah_resp_circ0u1u_n0708 fid_vs_depth_nah_resp_circ0u1u2q_n0708
    python3 plot_data.py --fidelity fid_vs_depth_kh_resp_circ0u1u_n0708  fid_vs_depth_kh_resp_circ0u1u2q_n0708
fi
