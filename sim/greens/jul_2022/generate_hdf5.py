import sys
sys.path.append('../../..')

import cirq
from fd_greens import (
    GroundStateSolver, 
    EHStatesSolver, 
    EHAmplitudesSolver, 
    GreensFunction, 
    get_lih_hamiltonian, 
    initialize_hdf5
)

def main():
    qubits = cirq.LineQubit.range(4)
    hamiltonian = get_lih_hamiltonian(3.0)
    fname = 'lih_greens_alltomo2q_trial'
    method = 'alltomo'
    spin = 'u'
    
    initialize_hdf5(fname, mode='greens', spin=spin)

    gs_solver = GroundStateSolver(hamiltonian, qubits, spin=spin, fname=fname)
    gs_solver.run()

    es_solver = EHStatesSolver(hamiltonian, spin=spin, fname=fname)
    es_solver.run()

    amp_solver = EHAmplitudesSolver(hamiltonian, qubits, spin=spin, method=method, fname=fname, repetitions=10000)
    amp_solver.run()

    greens = GreensFunction(hamiltonian, fname=fname, method=method, spin=spin)
    greens.process()

if __name__ == '__main__':
    main()
