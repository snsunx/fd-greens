"""
==============================================================
HDF5 Creation Utilities (:mod:`fd_greens.hdf5_creation_utils`)
==============================================================
"""

from typing import Optional

import cirq
import numpy as np
import h5py 

from .molecular_hamiltonian import get_alkali_hydride_hamiltonian
from .helpers import initialize_hdf5
from .ground_state_solver import GroundStateSolver
from .excited_states_solver import ExcitedStatesSolver
from .excited_amplitudes_solver import ExcitedAmplitudesSolver
from .response_function import ResponseFunction
from .classical_amplitudes_solver import ClassicalAmplitudesSolver
from .circuit_string_converter import CircuitStringConverter
from .general_utils import get_non_z_locations, get_gate_counts
from .noise_parameters import NoiseParameters

__all__ = ["create_hdf5", "create_hdf5_by_depth"]

def create_hdf5(
    h5fname: str,
    method: str = "exact",
    noise_fname: Optional[str] = None,
    repetitions: int = 10000
) -> None:
    """Creates an HDF5 file.
    
    Args:
        h5fname: The HDF5 file name.
        method: The method to generate the files.
        noise_fname: The noise parameter file name.
        repetitions: The number of repetitions to run the quantum circuits.
    """
    qubits = cirq.LineQubit.range(4)
    if "nah" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
    else:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)

    initialize_hdf5(h5fname, mode='resp', spin='')

    gs_solver = GroundStateSolver(hamiltonian, qubits, fname=h5fname)
    gs_solver.run()

    es_solver = ExcitedStatesSolver(hamiltonian, fname=h5fname)
    es_solver.run()

    amp_solver = ExcitedAmplitudesSolver(
        hamiltonian,
        qubits,
        method=method,
        fname=h5fname,
        noise_fname=noise_fname,
        repetitions=repetitions
    )
    amp_solver.run()

    resp = ResponseFunction(hamiltonian, fname=h5fname, method=method)
    resp.process()

    if method == 'exact':
        N = resp.N['n']

        classical_solver = ClassicalAmplitudesSolver(hamiltonian, verbose=False)
        classical_solver.compute_N()
        N_classical = classical_solver.N['n']

        print(np.allclose(N, N_classical))

def create_hdf5_by_depth(h5fname: str, circuit_name: str, noise_fname: Optional[str] = None) -> None:
    """Creates an HDF5 file by circuit depth.
    
    Args:
        h5fname: The HDF5 file name.
        circuit_name: Name of the circuit to be run at each depth.
        noise_fname: The noise parameter file name.
    """
    int_to_letter = {1: 's', 2: 'd', 3: 't'}

    if noise_fname is None:
        h5fname_new = f"{h5fname}_{circuit_name}.h5"
    else:
        h5fname_new = f"{h5fname}_{circuit_name}_{noise_fname[-4:]}.h5"
    
    qubits = cirq.LineQubit.range(4)
    converter = CircuitStringConverter(qubits)

    circuit = converter.load_circuit(h5fname, circuit_name + '/transpiled')
    non_z_locations = get_non_z_locations(circuit)
    
    h5file = h5py.File(h5fname_new, 'w')
    for i in non_z_locations:
        # converter.save_circuit(h5file, f"circuit/{i}", circuit[:i])
        # n_1q_gates = get_gate_counts(circuit[i], num_qubits=1)
        # n_2q_gates = get_gate_counts(circuit[i], num_qubits=2)
        # n_3q_gates = get_gate_counts(circuit[i], num_qubits=3)
        # if n_1q_gates > 0:
        #     index_string = str(i) + 's'
        # if n_2q_gates > 0:
        #     index_string = str(i) + 'd'
        # if n_3q_gates > 0:
        #     index_string = str(i) + 't'

        for n in [1, 2, 3]:
            if get_gate_counts(circuit[i], num_qubits=n) > 0:
                index_string = str(i) + int_to_letter[n]
        
        if noise_fname is None:
            state_vector = cirq.final_state_vector(circuit[:i])
            h5file[f'psi/{index_string}'] = state_vector
        else:
            noise_params = NoiseParameters.from_file(noise_fname)
            circuit_i = noise_params.add_noise_to_circuit(circuit[:i])
            density_matrix = cirq.final_density_matrix(circuit_i)
            h5file[f'rho/{index_string}'] = density_matrix
    
    h5file.close()
