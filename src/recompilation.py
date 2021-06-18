import numpy as np
from scipy.linalg import expm
import cirq
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize_autograd import TNOptimizer

import more_quimb_gates
# from quimb.tensor.optimize import TNOptimizer


def construct_ansatz_circuit(initial_state, pauli_domain,
                             single_qubit_gate='U3', two_qubit_gate='CZ', 
                             n_gate_rounds=4, periodic=False):
    """Construct ansatz circuit."""
    n_qubits = len(pauli_domain)
    _qubits = np.arange(n_qubits)
    qubit_map = dict(zip(_qubits, pauli_domain))

    N = initial_state.nsites
    if not initial_state is None:
        qtn_circuit = qtn.Circuit(N, tags='PSI0', psi0=qtn.Dense1D(initial_state))
    else:
        qtn_circuit = qtn.Circuit(N, tags='U0')

    params = (0., ) if single_qubit_gate == 'RY' else (0., 0., 0.)

    for i in range(n_qubits):
        qtn_circuit.apply_gate(single_qubit_gate, *params, qubit_map[i], gate_round=0, parametrize=True)

    end_qubit = n_qubits if periodic else n_qubits - 1

    for r in range(n_gate_rounds):
        for i in range(r % 2, end_qubit, 2):
            qtn_circuit.apply_gate(two_qubit_gate, qubit_map[i % N],
                                   qubit_map[(i + 1) % N], gate_round=r)
            qtn_circuit.apply_gate(single_qubit_gate, *params, qubit_map[i % N], gate_round=r, parametrize=True)
            qtn_circuit.apply_gate(single_qubit_gate, *params,
                                   qubit_map[(i + 1) % N], gate_round=r, parametrize=True)

    return qtn_circuit


def construct_ansatz_circuit_basis(n_qubits, tup=None, basis='Z', single_qubit_gate='U3', 
                                   two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Construct ansatz circuit."""

    qtn_circuit = qtn.Circuit(n_qubits, tags='U0')

    for i, t in enumerate(tup):
        if t == 1:
            qtn_circuit.apply_gate('X', i)
        if basis == 'X':
            qtn_circuit.apply_gate('H', i)

    params = (0., ) if single_qubit_gate == 'RY' else (0., 0., 0.)

    for i in range(n_qubits):
        qtn_circuit.apply_gate(single_qubit_gate, *params, i, gate_round=0, parametrize=True)

    end_qubit = n_qubits if periodic else n_qubits - 1

    x_terms = [(0, 1), (3, 4), (6, 7), (2, 8), (9, 10), (5, 11)]
    y_terms = [(1, 2), (4, 5), (0, 6), (3, 9)]
    z_terms = [(2, 3), (0, 5), (1, 7), (4, 10)]
    for r in range(n_gate_rounds):
        for term in x_terms:
            qtn_circuit.apply_gate(two_qubit_gate, term[0], term[1], gate_round=r)
            qtn_circuit.apply_gate(single_qubit_gate, *params, term[0], gate_round=r, parametrize=True)
            qtn_circuit.apply_gate(single_qubit_gate, *params, term[1], gate_round=r, parametrize=True)
        for term in y_terms:
            qtn_circuit.apply_gate(two_qubit_gate, term[0], term[1], gate_round=r)
            qtn_circuit.apply_gate(single_qubit_gate, *params, term[0], gate_round=r, parametrize=True)
            qtn_circuit.apply_gate(single_qubit_gate, *params, term[1], gate_round=r, parametrize=True)
        for term in z_terms:
            qtn_circuit.apply_gate(two_qubit_gate, term[0], term[1], gate_round=r)
            qtn_circuit.apply_gate(single_qubit_gate, *params, term[0], gate_round=r, parametrize=True)
            qtn_circuit.apply_gate(single_qubit_gate, *params, term[1], gate_round=r, parametrize=True)

    # qtn_circuit.psi.graph(color=['CZ', 'RY'])
    '''

    for r in range(n_gate_rounds):
        for i in range(r % 2, end_qubit, 2):
            qtn_circuit.apply_gate(two_qubit_gate, i % n_qubits, (i + 1) % n_qubits, gate_round=r)
            qtn_circuit.apply_gate(single_qubit_gate, *params, i % n_qubits, gate_round=r, parametrize=True)
            qtn_circuit.apply_gate(single_qubit_gate, *params, (i + 1) % n_qubits, gate_round=r, parametrize=True)


    '''
    return qtn_circuit


def construct_ansatz_circuit_dm(n_qubits, single_qubit_gate='U3', two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Construct ansatz circuit."""

    qtn_circuit = qtn.Circuit(n_qubits, tags='U0')

    params = (0., ) if single_qubit_gate == 'RY' else (0., 0., 0.)

    for i in range(n_qubits):
        qtn_circuit.apply_gate(single_qubit_gate, *params, i, gate_round=0, parametrize=True)

    end_qubit = n_qubits if periodic else n_qubits - 1

    for r in range(n_gate_rounds):
        for i in range(r % 2, end_qubit, 2):
            qtn_circuit.apply_gate(two_qubit_gate, i % n_qubits, (i + 1) % n_qubits, gate_round=r)
            qtn_circuit.apply_gate(single_qubit_gate, *params, i % n_qubits, gate_round=r, parametrize=True)
            qtn_circuit.apply_gate(single_qubit_gate, *params, (i + 1) % n_qubits, gate_round=r, parametrize=True)

    return qtn_circuit


def recompile_state(state, target_state, pauli_domain,
                    single_qubit_gate='U3', two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Recompile the circuit on a state to target state."""
    ansatz_circuit = construct_ansatz_circuit(state, pauli_domain, single_qubit_gate=single_qubit_gate,
                                              two_qubit_gate=two_qubit_gate, n_gate_rounds=n_gate_rounds, periodic=periodic)
    state_to_be_optimized = ansatz_circuit.psi

    def loss(_state, _target_state):
        return -(_state.H @ _target_state).real

    optimizer = TNOptimizer(
        state_to_be_optimized, loss_fn=loss,
        loss_constants={'_target_state': target_state},
        constant_tags=[two_qubit_gate, 'PSI0'],
        autograd_backend='jax', optimizer='L-BFGS-B')
    # autodiff_backend='jax', optimizer='L-BFGS-B')
    optimized_state = optimizer.optimize_basinhopping(n=500, nhop=10)
    ansatz_circuit.update_params_from(optimized_state)
    gates = ansatz_circuit.gates

    return gates


def recompile_with_dm(density_matrix, unitary_target_dense, pauli_domain, unitary_for_guess=None,
                      single_qubit_gate='U3', two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Recompile the circuit on a state to target state."""
    n_qubits = int(np.log2(unitary_target_dense.shape[0]))
    ansatz_circuit = construct_ansatz_circuit_dm(n_qubits, single_qubit_gate=single_qubit_gate,
                                                 two_qubit_gate=two_qubit_gate, n_gate_rounds=n_gate_rounds, periodic=periodic)
    if not unitary_for_guess is None:
        ansatz_circuit.update_params_from(unitary_for_guess)
    unitary_to_be_optimized = ansatz_circuit.uni

    rho_times_unitary_target_dense = density_matrix @ unitary_target_dense.conj().transpose()

    U_target = qtn.Tensor(
        data=rho_times_unitary_target_dense.reshape([2] * (2 * n_qubits)),
        inds=[f'k{i}' for i in range(n_qubits)] + [f'b{i}' for i in range(n_qubits)],
        tags={'U_TARGET'}
    )

    def loss(unitary_to_be_optimized, U_target):
        return 1 - (unitary_to_be_optimized | U_target).contract(all, optimize='auto-hq').real

    optimizer = TNOptimizer(
        unitary_to_be_optimized, loss_fn=loss,
        loss_constants={'U_target': U_target},
        constant_tags=[two_qubit_gate, 'PSI0'],
        autograd_backend='jax', optimizer='L-BFGS-B')
    # autodiff_backend='jax', optimizer='L-BFGS-B')

    if unitary_for_guess is None:
        optimized_unitary = optimizer.optimize_basinhopping(n=500, nhop=10)
    else:
        optimized_unitary = optimizer.optimize(n=1000)
    ansatz_circuit.update_params_from(optimized_unitary)
    gates = ansatz_circuit.gates

    return gates, optimized_unitary


def quimb_to_cirq_gates(gates, qubit_map):
    cirq_gates = []
    for gate in gates:
        if gate[0] == 'RY':
            cirq_gates.append(cirq.ry(gate[1])(qubit_map[gate[2]]))
        elif gate[0] == 'U3':
            # text{U3}(theta, phi, lambda) = R_z(phi) R_y(theta) R_z(lambda)
            # Why 3-1-2 and not 2-1-3?
            cirq_gates.append(cirq.rz(gate[3])(qubit_map[gate[4]]))
            cirq_gates.append(cirq.ry(gate[1])(qubit_map[gate[4]]))
            cirq_gates.append(cirq.rz(gate[2])(qubit_map[gate[4]]))
        elif gate[0] == 'PHXZ':
            cirq_gates.append(cirq.PhasedXZGate(
                x_exponent=gate[1], z_exponent=gate[2], axis_phase_exponent=gate[3])(qubit_map[gate[4]]))
        elif gate[0] == 'CZ':
            cirq_gates.append(cirq.CZ(qubit_map[gate[1]], qubit_map[gate[2]]))
    return cirq_gates


def propagate_recompiled(qubits, state, coefficients, pauli_strings, pauli_domain,
                         single_qubit_gate='U3', two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Propagate with recompiled gates."""
    n_terms = len(pauli_strings)
    dense_pauli_strings = [pauli_strings[i].dense(qubits) for i in range(n_terms)]
    operator_on_exponent = np.zeros(dense_pauli_strings[0]._unitary_().shape, dtype=complex)
    for i in range(n_terms):
        operator_on_exponent += coefficients[i] * dense_pauli_strings[i]._unitary_()
    unitary = expm(-1j * operator_on_exponent)

    # If unitary is an identity matrix, no gates are needed to be added.
    if not np.any(unitary - np.eye(unitary.shape[0])):
        return [], []

    N = int(np.log2(state.shape[0]))
    unitary_padded = qu.pkron(unitary, dims=(2,) * N, inds=pauli_domain)

    '''
    target_state = psi.copy()
    target_state.gate_(unitary, pauli_domain, tags='U', contract=False)
    '''

    target_state = unitary_padded @ state
    state = qtn.Dense1D(state)
    target_state = qtn.Dense1D(target_state)
    gates = recompile_state(state, target_state, pauli_domain, single_qubit_gate=single_qubit_gate,
                            two_qubit_gate=two_qubit_gate, n_gate_rounds=n_gate_rounds, periodic=periodic)

    qubit_map = dict(zip(pauli_domain, qubits))
    cirq_gates = quimb_to_cirq_gates(gates, qubit_map)

    return cirq_gates, gates


def propagate_recompiled_dm(qubits, density_matrix, coefficients, pauli_strings, pauli_domain,
                            unitary_for_guess=None,
                            single_qubit_gate='U3', two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Propagate with recompiled gates."""
    n_terms = len(pauli_strings)
    dense_pauli_strings = [pauli_strings[i].dense(qubits) for i in range(n_terms)]
    operator_on_exponent = np.zeros(dense_pauli_strings[0]._unitary_().shape, dtype=complex)
    for i in range(n_terms):
        operator_on_exponent += coefficients[i] * dense_pauli_strings[i]._unitary_()
    unitary_target_dense = expm(-1j * operator_on_exponent)

    # If unitary is an identity matrix, no gates are needed to be added.
    if not np.any(unitary_target_dense - np.eye(unitary_target_dense.shape[0])):
        return [], [], None

    gates, optimized_unitary = recompile_with_dm(density_matrix, unitary_target_dense, pauli_domain,
                                                 unitary_for_guess=unitary_for_guess,
                                                 single_qubit_gate=single_qubit_gate, two_qubit_gate=two_qubit_gate,
                                                 n_gate_rounds=n_gate_rounds, periodic=periodic)

    quimb_qubits = np.arange(len(pauli_domain))
    qubit_map = dict(zip(quimb_qubits, qubits))
    cirq_gates = quimb_to_cirq_gates(gates, qubit_map)

    return cirq_gates, gates, optimized_unitary


def recompile_circuit(qubits, gates_for_psi_target, psi_for_guess=None,
                      single_qubit_gate='U3', two_qubit_gate='CZ', n_gate_rounds=4, periodic=False):
    """Recompile the circuit more aggresively"""
    n_qubits = len(qubits)
    ansatz_circuit = construct_ansatz_circuit_dm(
        n_qubits, single_qubit_gate=single_qubit_gate, two_qubit_gate=two_qubit_gate, n_gate_rounds=n_gate_rounds, periodic=periodic)
    # Use initial guess from the previous step
    if not psi_for_guess is None:
        ansatz_circuit.update_params_from(psi_for_guess)
    psi_to_be_optimized = ansatz_circuit.psi

    circuit_target = qtn.Circuit(n_qubits, tags='psi_target')
    circuit_target.apply_gates(gates_for_psi_target)
    psi_target = circuit_target.psi
    for key in psi_target.tags:
        if key.startswith("GATE"):
            psi_target.drop_tags(tags=key)
            psi_target.add_tag('psi_target')

    def loss(psi_to_be_optimized, psi_target):
        return -(psi_to_be_optimized.H @ psi_target).real

        # 1 - (unitary_to_be_optimized | U_target).contract(all, optimize='auto-hq').real

    optimizer = TNOptimizer(
        psi_to_be_optimized, loss_fn=loss,
        loss_constants={'psi_target': psi_target},
        constant_tags=[two_qubit_gate, 'U0', 'psi_target'],
        autograd_backend='jax', optimizer='L-BFGS-B')

    if psi_for_guess is None:
        optimized_psi = optimizer.optimize_basinhopping(n=500, nhop=10)
    else:
        optimized_psi = optimizer.optimize(n=1000)

    ansatz_circuit.update_params_from(optimized_psi)
    gates = ansatz_circuit.gates
    cirq_gates = quimb_to_cirq_gates(gates, qubits)

    return cirq_gates, optimized_psi


def recompile_circuit_uni(qubits, gates_for_unitary_target, pauli_domain, unitary_for_guess=None,
                          single_qubit_gate='U3', two_qubit_gate='CZ',
                          n_gate_rounds=4, periodic=False):
    """Recompile the circuit more aggresively"""
    n_qubits = len(qubits)
    ansatz_circuit = construct_ansatz_circuit_dm(n_qubits, single_qubit_gate=single_qubit_gate,
                                                 two_qubit_gate=two_qubit_gate, n_gate_rounds=n_gate_rounds, periodic=periodic)
    # Use initial guess from the previous step
    if not unitary_for_guess is None:
        ansatz_circuit.update_params_from(unitary_for_guess)
    unitary_to_be_optimized = ansatz_circuit.uni

    circuit_target = qtn.Circuit(n_qubits, tags='unitary_target')
    circuit_target.apply_gates(gates_for_unitary_target)
    unitary_target = circuit_target.uni
    unitary_target_dense = unitary_target.to_dense([f'k{i}' for i in range(n_qubits)], [
                                                   f'b{i}' for i in range(n_qubits)])

    density_matrix = np.zeros((16, 16))
    density_matrix[0, 0] = 1.0
    where_apply_dm = np.arange(n_qubits)
    unitary_target.gate_(density_matrix, where_apply_dm, tags='rho', contract=False)

    def loss(unitary_to_be_optimized, unitary_target):
        # return 1 - (unitary_to_be_optimized | unitary_target).contract(all, optimize='auto-hq').real / (2 ** n)
        return 1 - (unitary_to_be_optimized | unitary_target).contract(all, optimize='auto-hq').real

    optimizer = TNOptimizer(
        unitary_to_be_optimized, loss_fn=loss,
        loss_constants={'unitary_target': unitary_target},
        constant_tags=[two_qubit_gate, 'U0', 'unitary_target', 'rho'],
        autograd_backend='jax', optimizer='L-BFGS-B')

    if unitary_for_guess is None:
        optimized_unitary = optimizer.optimize_basinhopping(n=500, nhop=10)
    else:
        optimized_unitary = optimizer.optimize(n=1000)

    qubit_map = dict(zip(pauli_domain, qubits))

    ansatz_circuit.update_params_from(optimized_unitary)
    gates = ansatz_circuit.gates
    cirq_gates = quimb_to_cirq_gates(gates, qubit_map)

    return cirq_gates, optimized_unitary
