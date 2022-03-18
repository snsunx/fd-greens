from itertools import product
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

from tools import *

def build_qubit_op(geometry, basis='sto3g', multiplicity=1, charge=0, 
                         occupied_indices=None, active_indices=None):
    """Constructs the Qiskit qubit operator."""
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    run_pyscf(molecule)
    if occupied_indices is None and active_indices is None:
        active_indices = range(molecule.n_orbitals)
    hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices)
    fermion_operator = get_fermion_operator(hamiltonian)
    qubit_op = jordan_wigner(fermion_operator)
    qubit_op.compress()
    #mat = get_sparse_operator(qubit_op)
    #e, v = eigsh(mat, k=1, which='SA')
    #print(r, active_indices, e[0])
    
    qubit_op = openfermion_to_qiskit_operator(qubit_op)
    return qubit_op


def run_vqe(ansatz, 
            qubit_op, 
            optimizer=None, 
            quantum_instance=None, 
            mode='min'):
    """Main funciton for VQE calculation."""
    assert mode in ['min', 'max']
    if optimizer is None:
        optimizer = L_BFGS_B()
    if quantum_instance is None:
        quantum_instance = Aer.get_backend('statevector_simulator')

    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
    if mode == 'max':
        qubit_op *= -1
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    energy = result.optimal_value
    if mode == 'max':
        energy *= -1
    ansatz.assign_parameters(result.optimal_parameters, inplace=True)
    return energy, ansatz
