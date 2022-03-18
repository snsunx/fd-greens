"""
def main(r, run_qse=False):
    '''
    problem = ElectronicStructureProblem(
        driver=PySCFDriver(atom=f'H 0 0 0; H 0 0 {r}'),
        q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True)])
    fermionic_op = problem.second_q_ops()[0]
    qubit_op = QubitConverter(mapper=JordanWignerMapper()).convert(fermionic_op)
    '''
    ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=1)
    print(ansatz)
    exit()
    optimizer = L_BFGS_B()
    backend = Aer.get_backend('statevector_simulator')
    qubit_op = build_qubit_operator(r, occupied_indices=[0, 1], active_indices=[2, 3])
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=backend)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    e = result.eigenvalue

    if run_qse:
        hamiltonian = qubit_op.to_matrix()
        strs = ['IIII', 'IIIX', 'IIIY', 'IIIZ', 'IIXI', 'IIYI', 'IIZI',
                'IXII', 'IYII', 'IZII', 'XIII', 'YIII', 'ZIII']
        subspace_paulis = [Pauli(s).to_matrix() for s in strs]
        H, S = build_qse_matrices(hamiltonian, result.eigenstate, subspace_paulis)
        w, v = roothaan_eig(H, S)
        print(list(w))
    return list(w)
        #e, v = np.linalg.eigh(qubit_op.to_matrix())
    #e_gs = np.min(e)
    #print(e_gs)
"""