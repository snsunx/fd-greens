from typing import Union 

import numpy as np
import h5py

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import Barrier
from qiskit.extensions import UnitaryGate, SwapGate

from hamiltonians import MolecularHamiltonian
#from ground_state_solvers import GroundStateSolver
#from number_states_solvers import EHStatesSolver, ExcitedStatesSolver
#from amplitudes_solvers import EHAmplitudesSolver, ExcitedAmplitudesSolver

def get_lih_hamiltonian(r: float) -> MolecularHamiltonian:
    """Returns the HOMO-LUMO LiH Hamiltonian with bond length r."""
    hamiltonian = MolecularHamiltonian(
        [['Li', (0, 0, 0)], ['H', (0, 0, r)]], 'sto3g', 
        occ_inds=[0], act_inds=[1, 2])
    return hamiltonian

def get_quantum_instance(type_str) -> QuantumInstance:
    """Returns the QuantumInstance from type string."""
    if type_str == 'sv':
        q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    elif type_str == 'qasm':
        q_instance = QuantumInstance(Aer.get_backend('qasm_simulator', shots=10000), shots=10000)
    elif type_str == 'noisy':
        q_instance = QuantumInstance(Aer.get_backend('qasm_simulator', shots=10000, noise_model_name='ibmq_jakarta'), shots=10000)
    return q_instance

def get_berkeley_ccx_data():
    iX = np.array([[0, 1j], [1j, 0]])
    ccx_data = [(SwapGate(), [1, 2]), 
                (Barrier(4), [0, 1, 2, 3]), 
                (UnitaryGate(iX).control(2), [0, 2, 1]), 
                (Barrier(4), [0, 1, 2, 3]),
                (SwapGate(), [1, 2])]
    return ccx_data

def save_eh_data(gs_solver: 'GroundStateSolver', 
                 es_solver: 'EHStatesSolver',
                 amp_solver: 'EHAmplitudesSolver',
                 fname: str = 'lih') -> None:
    f = h5py.File(fname + '_eh.h5py', 'w')
    f['energy_gs'] = gs_solver.energy
    f['energies_e'] = es_solver.energies_e
    f['energies_h'] = es_solver.energies_h
    f['B_e'] = amp_solver.B_e
    f['B_h'] = amp_solver.B_h
    h = amp_solver.h
    e_orb = np.diag(h.molecule.orbital_energies)
    f['e_orb'] = e_orb[h.act_inds][:, h.act_inds]
    f.close()

def save_exc_data(gs_solver: 'GroundStateSolver', 
                  es_solver: 'ExcitedStatesSolver',
                  amp_solver: 'ExcitedAmplitudesSolver',
                  fname: str = 'lih') -> None:
    f = h5py.File(fname + '_exc.h5py', 'w')
    f['energy_gs'] = gs_solver.energy
    f['energies_s'] = es_solver.energies_s
    f['energies_t'] = es_solver.energies_t
    f['L'] = amp_solver.L
    f['n_states'] = amp_solver.n_states
    f.close()
