#!/bin/bash

create_hdf5=true
generate_data=false
print_information=true

if [ $create_hdf5 = true ]
then
    # Create exact HDF5 files
    #python3 create_hdf5.py nah_greens_exact exact
    #python3 create_hdf5.py kh_greens_exact exact

    # Create HDF5 files with iToffoli decomposition
    export CONVERT_CCZ_TO_ITOFFOLI=1
    python3 create_hdf5.py nah_greens_tomo tomo
    # python3 create_hdf5.py kh_greens_tomo tomo

    # Create HDF5 files with CZ decomposition
    export CONVERT_CCZ_TO_ITOFFOLI=0
    # python3 create_hdf5.py nah_greens_tomo2q tomo
    # python3 create_hdf5.py kh_greens_tomo2q tomo
fi


if [ $generate_data = true ]
then
    # Generate data
    python3 generate_data.py nah_greens_exact nah_greens_tomo nah_greens_tomo2q 
    python3 generate_data.py kh_greens_exact kh_greens_tomo kh_greens_tomo2q
fi

if [ $print_information = true ]
then
    python3 print_information.py --circuit nah_greens_tomo 
fi
