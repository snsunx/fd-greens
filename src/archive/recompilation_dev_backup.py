# TODO: Need to modify the following
"""
def recompile_with_densitymatrix(
        density_matrix, target_unitary, domain_qubits, initial_guess=None,
        gate_1q='U3', gate_2q='CZ', n_gate_rounds=4, 
        periodic=False):
    # If unitary is an identity matrix, no gates are needed to be added.
    # if not np.any(target_unitary - np.eye(target_unitary.shape[0])):
    #     return [], [], None
    #gate_1q = 'U3'

    n_qubits = int(np.log2(target_unitary.shape[0]))
    ansatz_circuit = build_ansatz_circuit(
        n_qubits, gate_1q=gate_1q,
        gate_2q=gate_2q, n_gate_rounds=n_gate_rounds, 
        periodic=periodic)
    if initial_guess is not None:
        ansatz_circuit.update_params_from(initial_guess)
    U_ansatz = ansatz_circuit.uni
    #print(U_ansatz)

    rho = qtn.Tensor(
        data=density_matrix.reshape((2, ) * 2 * n_qubits), 
        inds=[f'k{i}' for i in range(n_qubits)] + \
             [f'l{i}' for i in range(n_qubits)], 
        tags={'I0', 'I1', 'I2', 'I3'})
    
    U_target = qtn.Tensor(
        data=target_unitary.reshape((2, ) * 2 * n_qubits), 
        inds=[f'c{i}' for i in range(n_qubits)] + \
             [f'k{i}' for i in range(n_qubits)], 
        tags={'I0', 'I1', 'I2', 'I3'})

    U_target = density_matrix @ target_unitary.conj().T
    #U_target = target_unitary @ density_matrix
    U_target = qtn.Tensor(
        data=U_target.reshape([2] * (2 * n_qubits)),
        inds=[f'k{i}' for i in range(n_qubits)] + \
             [f'b{i}' for i in range(n_qubits)],
        tags={'U_TARGET'})

    # TODO: Jozsa fidelity not working yet.

    def infidelity(U_ansatz, rho, U_target):
        U_ansatz_H = U_ansatz.copy()
        U_ansatz_H = U_ansatz_H.H
        #print(type(U_ansatz_H))
        U_ansatz_H.reindex({'k0': 'l0', 'k1': 'l1', 'k2': 'l2', 'k3': 'l3',
                            'b0': 'c0', 'b1': 'c1', 'b2': 'c2', 'b3': 'c3'},
                            inplace=True)
        U_target_H = U_target.copy()
        U_target_H = U_target_H.H
        U_target_H.reindex({'k0': 'l0', 'k1': 'l1', 'k2': 'l2', 'k3': 'l3',
                            'c0': 'b0', 'c1': 'b1', 'c2': 'b2', 'c3': 'b3'},
                            inplace=True)
        #print(U_ansatz)
        #print(U_ansatz_H)
        rho_ansatz = (U_ansatz | rho | U_ansatz_H).contract(all, optimize='auto-hq')
        rho_target = U_target @ rho @ U_target_H
        print(rho_ansatz)
        print(rho_target)
        return 1 - abs(rho_ansatz @ rho_target) ** 2
        #return 1 - (recompiled_unitary | density_matrix | target_unitary.H).contract(
        #    all, optimize='auto-hq').real
    def infidelity(U_ansatz, U_target):
        return 1 - (U_ansatz | U_target).contract(all, optimize='auto-hq').real ** 2

    optimizer = TNOptimizer(
        U_ansatz, loss_fn=infidelity,
        loss_constants={'U_target': U_target},
        constant_tags=[gate_2q, 'PSI0'],
        autograd_backend='jax', optimizer='L-BFGS-B')
    if initial_guess is None:
        U_recompiled = optimizer.optimize_basinhopping(n=500, nhop=10)
    else:
        U_recompiled = optimizer.optimize(n=1000)

    ansatz_circuit.update_params_from(U_recompiled)
    psi = ansatz_circuit.to_dense()
    quimb_gates = ansatz_circuit.gates
    cirq_ops = quimb_gates_to_cirq_ops(quimb_gates, domain_qubits)
    return cirq_ops, U_recompiled


# (SS): qubits is passed in because circuit.all_qubits() might give a 
# different order than qubits
def recompile_circuit(circuit, qubits, initial_guess=None,
                      gate_1q='U3', gate_2q='CNOT', 
                      n_gate_rounds=8, periodic=False):
    print('recompiled circuit is called')
    n_qubits= len(qubits)
    ansatz_circuit = build_ansatz_circuit(
        n_qubits, gate_1q=gate_1q, 
        gate_2q=gate_2q, n_gate_rounds=n_gate_rounds, 
        periodic=periodic)
    if initial_guess is not None:
        ansatz_circuit.update_params_from(initial_guess)
    psi_ansatz = ansatz_circuit.psi

    cirq_ops = list(circuit.all_operations())
    quimb_gates = cirq_ops_to_quimb_gates(cirq_ops, qubits)
    target_circuit = qtn.Circuit(n_qubits)
    target_circuit.apply_gates(quimb_gates)
    psi_target = target_circuit.psi

    '''
    circuit_target = qtn.Circuit(n_qubits, tags='psi_target')
    circuit_target.apply_gates(gates_for_psi_target)
    psi_target = circuit_target.psi
    for tag in psi_target.tags:
        if tag.startswith("GATE"):
            psi_target.drop_tags(tags=tag)
            psi_target.add_tag('psi_target')
    '''

    def infidelity(psi_ansatz, psi_target):
        return 1 - abs(psi_ansatz.H @ psi_target) ** 2

    optimizer = TNOptimizer(
        psi_ansatz, loss_fn=infidelity,
        loss_constants={'psi_target': psi_target},
        constant_tags=[gate_2q, 'PSI0'],
        autograd_backend='jax', optimizer='L-BFGS-B')
    if initial_guess is None:
        psi_recompiled = optimizer.optimize_basinhopping(n=500, nhop=10)
    else:
        psi_recompiled = optimizer.optimize(n=1000)

    ansatz_circuit.update_params_from(psi_recompiled)
    quimb_gates = ansatz_circuit.gates
    cirq_ops = quimb_gates_to_cirq_ops(quimb_gates, qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq_ops)
    return circuit, psi_recompiled
"""