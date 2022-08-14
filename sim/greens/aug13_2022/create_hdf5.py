import sys
sys.path.append('../../..')

import cirq
from fd_greens import (
    GroundStateSolver, 
    EHStatesSolver, 
    EHAmplitudesSolver, 
    GreensFunction, 
    get_alkali_hydride_hamiltonian,
    initialize_hdf5
)

def main():
    qubits = cirq.LineQubit.range(4)
    noise_fname = '../../../expt/params/gate_fidelities_0708'
    
    assert len(sys.argv) == 3
    fname = sys.argv[1] # 'nah_greens_exact'
    method = sys.argv[2] # 'exact'

    assert fname[:2] in ['na', 'kh']
    if fname[:2] == 'na':
        hamiltonian = get_alkali_hydride_hamiltonian("Na", 3.7)
    elif fname[:2] == 'kh':
        hamiltonian = get_alkali_hydride_hamiltonian("K", 3.9)
    spin = 'u'
    repetitions = 10000
    
    initialize_hdf5(fname, mode='greens', spin=spin)

    gs_solver = GroundStateSolver(hamiltonian, qubits, spin=spin, fname=fname)
    gs_solver.run()

    es_solver = EHStatesSolver(hamiltonian, spin=spin, fname=fname)
    es_solver.run()

    amp_solver = EHAmplitudesSolver(
        hamiltonian, qubits, spin=spin, method=method, 
        noise_fname=noise_fname, fname=fname, repetitions=repetitions)
    amp_solver.run()

    greens = GreensFunction(hamiltonian, fname=fname, method=method, spin=spin)
    greens.process()

if __name__ == '__main__':
    main()
