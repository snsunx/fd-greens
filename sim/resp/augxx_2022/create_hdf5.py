import sys

sys.path.append('../../..')
import argparse
from typing import Optional

import h5py
import numpy as np
import cirq

from fd_greens import (
    GroundStateSolver, 
    ExcitedStatesSolver, 
    ExcitedAmplitudesSolver, 
    ResponseFunction, 
    get_alkali_hydride_hamiltonian,
    initialize_hdf5,
    ClassicalAmplitudesSolver,
    CircuitStringConverter,
    get_non_z_locations,
    NoiseParameters,
    get_gate_counts
)
from fd_greens.cirq_ver.helpers import print_circuit

def create_hdf5(fname: str, method: str, noise_fname: str, repetitions: int) -> None:
    qubits = cirq.LineQubit.range(4)
    hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)

    initialize_hdf5(fname, mode='resp', spin='')

    gs_solver = GroundStateSolver(hamiltonian, qubits, fname=fname)
    gs_solver.run()

    es_solver = ExcitedStatesSolver(hamiltonian, fname=fname)
    es_solver.run()

    amp_solver = ExcitedAmplitudesSolver(hamiltonian, qubits, method=method, fname=fname, noise_fname=noise_fname, repetitions=repetitions)
    amp_solver.run()

    resp = ResponseFunction(hamiltonian, fname=fname, method=method)
    resp.process()

    if method == 'exact':
        N = resp.N['n']

        classical_solver = ClassicalAmplitudesSolver(hamiltonian, verbose=False)
        classical_solver.compute_N()
        N_classical = classical_solver.N['n']

        print(np.allclose(N, N_classical))


def create_hdf5_by_depth(h5fname: str, noise_fname: Optional[str], circuit_name: str) -> None:
    qubits = cirq.LineQubit.range(4)
    converter = CircuitStringConverter(qubits)

    circuit = converter.load_circuit(h5fname, circuit_name + '/transpiled')
    non_z_locations = get_non_z_locations(circuit)

    if noise_fname is None:
        h5fname_new = f"{h5fname}_{circuit_name}.h5"
    else:
        h5fname_new = f"{h5fname}_{circuit_name}n.h5"
    h5file = h5py.File(h5fname_new, 'w')
    for i in non_z_locations:
        # converter.save_circuit(h5file, f"circuit/{i}", circuit[:i])
        n_1q_gates = get_gate_counts(circuit[i], num_qubits=1)
        n_2q_gates = get_gate_counts(circuit[i], num_qubits=2)
        n_3q_gates = get_gate_counts(circuit[i], num_qubits=3)
        if n_1q_gates > 0:
            index_string = str(i) + 's'
        if n_2q_gates > 0:
            index_string = str(i) + 'd'
        if n_3q_gates > 0:
            index_string = str(i) + 't'
        
        if noise_fname is None:
            state_vector = cirq.final_state_vector(circuit[:i])
            h5file[f'psi/{index_string}'] = state_vector
        else:
            noise_params = NoiseParameters.from_file(noise_fname)
            circuit_i = noise_params.add_noise_to_circuit(circuit[:i])
            # print(f"circuit {i}\n")
            # print_circuit(circuit_i)
            density_matrix = cirq.final_density_matrix(circuit_i)
            h5file[f'rho/{index_string}'] = density_matrix
    
    h5file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    parser.add_argument("-m", "--method", type=str, default="exact")
    parser.add_argument("-n", "--noise", type=str, default=None)
    parser.add_argument("-c", "--circuit", type=str, default=None)
    parser.add_argument("-r", "--repetitions", type=int, default=10000)
    args = parser.parse_args()

    print(args)

    if args.circuit is None:
        print("Calling create_hdf5")
        create_hdf5(args.fname, args.method, args.noise, args.repetitions)
    else:
        print("Calling create_hdf5_by_depth")
        create_hdf5_by_depth(args.fname, args.noise, args.circuit)


if __name__ == '__main__':
    main()
