"""
==============================================================
HDF5 Creation Utilities (:mod:`fd_greens.hdf5_creation_utils`)
==============================================================
"""

from typing import Optional, Sequence

import cirq
import numpy as np
import h5py 

from .molecular_hamiltonian import get_alkali_hydride_hamiltonian
from .helpers import initialize_hdf5
from .ground_state_solver import GroundStateSolver
from .eh_states_solver import EHStatesSolver
from .eh_amplitudes_solver import EHAmplitudesSolver
from .excited_states_solver import ExcitedStatesSolver
from .excited_amplitudes_solver import ExcitedAmplitudesSolver
from .greens_function import GreensFunction
from .response_function import ResponseFunction
from .classical_amplitudes_solver import ClassicalAmplitudesSolver
from .circuit_string_converter import CircuitStringConverter
from .circuit_constructor import CircuitConstructor
from .general_utils import get_non_z_locations, get_gate_counts, histogram_to_array, quantum_state_tomography
from .noise_parameters import NoiseParameters

__all__ = ["create_greens_hdf5", "create_resp_hdf5", "create_resp_hdf5_by_depth"]


def create_greens_hdf5(
    h5fname: str,
    qubits: Optional[Sequence[cirq.Qid]] = None,
    method: str = "exact",
    noise_fname: Optional[str] = None,
    repetitions: int = 10000
) -> None:
    if qubits is None:
        qubits = cirq.LineQubit.range(4)
    if "lih" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Li", 3.0)
    elif "nah" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
    elif "kh" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)

    initialize_hdf5(h5fname, calculation_mode="greens")

    gs_solver = GroundStateSolver(hamiltonian, qubits, h5fname=h5fname)
    gs_solver.run()

    es_solver = EHStatesSolver(hamiltonian, h5fname=h5fname)
    es_solver.run()

    amp_solver = EHAmplitudesSolver(
        hamiltonian,
        qubits,
        method=method,
        h5fname=h5fname,
        noise_fname=noise_fname,
        repetitions=repetitions)
    amp_solver.run()

    obs_solver = GreensFunction(hamiltonian, h5fname=h5fname, method=method)
    obs_solver.process()

def create_resp_hdf5(
    h5fname: str,
    qubits: Optional[Sequence[cirq.Qid]] = None,
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
    if qubits is None:
        qubits = cirq.LineQubit.range(4)
    if "nah" in h5fname:
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
    else:
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)

    initialize_hdf5(h5fname, calculation_mode="resp")

    gs_solver = GroundStateSolver(hamiltonian, qubits, h5fname=h5fname, calculation_mode="resp")
    gs_solver.run()

    es_solver = ExcitedStatesSolver(hamiltonian, h5fname=h5fname)
    es_solver.run()

    amp_solver = ExcitedAmplitudesSolver(
        hamiltonian, 
        qubits,
        method=method,
        h5fname=h5fname,
        noise_fname=noise_fname,
        repetitions=repetitions)
    amp_solver.run()

    obs_solver = ResponseFunction(hamiltonian, h5fname=h5fname, method=method)
    obs_solver.process()

    if method == "exact":
        N = obs_solver.N['n']

        classical_solver = ClassicalAmplitudesSolver(hamiltonian, verbose=True)
        classical_solver.compute_N()
        N_classical = classical_solver.N['n']

        is_all_close = np.allclose(N, N_classical)
        if is_all_close:
            print("###### Passed #####")
        else:
            print("##### Not equal, norm is ", np.linalg.norm(N - N_classical), "#####")

        # print(N)
        # print(N_classical)

create_hdf5 = create_resp_hdf5

def create_resp_hdf5_by_depth(
    h5fname: str,
    circuit_name: str,
    noise_fname: Optional[str] = None,
    repetitions: int = 1000
) -> None:
    """Creates an HDF5 file by circuit depth.
    
    Args:
        h5fname: The HDF5 file name.
        circuit_name: Name of the circuit to be run at each depth.
        noise_fname: The noise parameter file name.
        repetitions: The repetitions for noisy simulation.
    """
    int_to_letter = {1: 's', 2: 'd', 3: 't'}

    h5fname_new = '_'.join(h5fname.split('_')[:-1])
    suffix_2q = "2q" if "2q" in h5fname else ""
    if noise_fname is None:
        h5fname_new = f"{h5fname_new}_{circuit_name}{suffix_2q}.h5"
        noise_params = None
        simulator = cirq.Simulator()
    else:
        h5fname_new = f"{h5fname_new}_{circuit_name}{suffix_2q}_n{noise_fname[-4:]}.h5"
        noise_params = NoiseParameters.from_file(noise_fname)
        simulator = cirq.DensityMatrixSimulator()

    # print(noise_params.noise_channels)

    qubits = cirq.LineQubit.range(4)
    converter = CircuitStringConverter(qubits)

    circuit = converter.load_circuit(h5fname, circuit_name + '/transpiled')
    non_z_locations = get_non_z_locations(circuit)
    
    h5file = h5py.File(h5fname_new, 'w')
    for i in non_z_locations:
        print("i = ", i)

        # Get the index string, which consists of the current depth and the nq_gates letter.
        circuit_component = circuit[i]
        for n in [1, 2, 3]:
            if get_gate_counts(circuit_component, num_qubits=n) > 0:
                index_string = str(i) + int_to_letter[n]
        
        # Save transpiled and tomography circuits to file.
        if i != non_z_locations[-1]:
            circuit_current = circuit[:i + 1] # i + 1 is for including the first i elements in front
        else:
            circuit_current = circuit[:] # For the last location, include all gate layers
        converter.save_circuit(h5file, f"circ{i}/transpiled", circuit_current)
        tomography_circuits = CircuitConstructor.build_tomography_circuits(
            circuit_current, tomographed_qubits=qubits)

        # Obtain the state vector and save to file.
        state_vector = cirq.final_state_vector(circuit_current)
        h5file[f"psi/{index_string}"] = state_vector
        
        for tomo_label, tomo_circuit in tomography_circuits.items():
            converter.save_circuit(h5file, f"circ{i}/{tomo_label}", tomo_circuit, return_dataset=True)

            # Save simulated counts to the dataset.
            if noise_params is not None:
                tomo_circuit = noise_params.add_noise_to_circuit(tomo_circuit)
            result = simulator.run(tomo_circuit, repetitions=repetitions)
            histogram = result.multi_measurement_histogram(keys=[str(q) for q in qubits])
            h5file[f"circ{i}/{tomo_label}"].attrs["counts"] = histogram_to_array(histogram, n_qubits=4)
    
    h5file.close()
