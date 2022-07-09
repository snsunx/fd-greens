import sys
sys.path.append('../../..')

import numpy as np
import cirq
from fd_greens import (
    GroundStateSolver, 
    ExcitedStatesSolver, 
    ExcitedAmplitudesSolver, 
    ResponseFunction, 
    get_lih_hamiltonian, 
    initialize_hdf5,
    ClassicalAmplitudesSolver)

def main():
    qubits = cirq.LineQubit.range(4)
    hamiltonian = get_lih_hamiltonian(3.0)
    fname = 'lih_resp_tomo_trial'
    method = 'tomo'

    initialize_hdf5(fname, mode='resp', spin='')

    gs_solver = GroundStateSolver(hamiltonian, qubits, fname=fname)
    gs_solver.run()

    es_solver = ExcitedStatesSolver(hamiltonian, fname=fname)
    es_solver.run()

    amp_solver = ExcitedAmplitudesSolver(hamiltonian, qubits, method=method, fname=fname, repetitions=100000)
    amp_solver.run()

    resp = ResponseFunction(hamiltonian, fname=fname, method=method)
    resp.process()

    if method == 'exact':
        N = resp.N['n']

        classical_solver = ClassicalAmplitudesSolver(hamiltonian, verbose=False)
        classical_solver.compute_N()
        N1 = classical_solver.N['n']

        print(np.allclose(N, N1))

if __name__ == '__main__':
    main()
