"""recompilation.py from the dev branch."""

from typing import Union, Sequence
from math import pi
import numpy as np

from scipy.linalg import expm
import quimb as qu
import quimb.tensor as qtn

from quimb.tensor.optimize_autograd import TNOptimizer

from cache import CacheRecompilation


def build_ansatz_circuit(n_qubits: int, 
                         psi0: qtn.Dense1D = None,
                         gate_1q: str ='U3', 
                         gate_2q: str ='CZ', 
                         n_gate_rounds: int = 4, 
                         periodic: bool = False) -> qtn.Circuit:
    """Constructs ansatz circuit. """
    assert gate_1q in ['U3', 'RY']
    assert gate_2q in ['CNOT', 'CZ']

    if psi0 is None:
        ansatz_circ = qtn.Circuit(n_qubits, tags='PSI0')
    else:
        ansatz_circ = qtn.Circuit(n_qubits, psi0=psi0, tags='PSI0')

    params = (0., ) if gate_1q == 'RY' else (0., 0., 0.)
    end_qubit = n_qubits if periodic else n_qubits - 1

    # Base layer of single-qubit gates.
    for i in range(n_qubits):
        ansatz_circ.apply_gate(
            gate_1q, *params, i, gate_round=0, parametrize=True)
    
    # Subsequent gate rounds of interleaving single-qubit gate and two-qubit
    # gate layers.
    for r in range(n_gate_rounds):
        for i in range(r % 2, end_qubit, 2):
            ansatz_circ.apply_gate(
                gate_2q, (i + 1) % n_qubits, i % n_qubits, gate_round=r)
            ansatz_circ.apply_gate(
                gate_1q, *params, i % n_qubits, gate_round=r, parametrize=True)
            ansatz_circ.apply_gate(
                gate_1q, *params, (i + 1) % n_qubits, gate_round=r, 
                parametrize=True)
    return ansatz_circ


def recompile_with_statevector(statevector: np.ndarray, 
                               target_unitary: np.ndarray,
                               init_guess: qu.tensor.circuit.Circuit = None, 
                               gate_1q: str ='U3', 
                               gate_2q: str ='CNOT', 
                               n_gate_rounds: int = 4, 
                               periodic: bool = False,
                               cache_options: dict = None):
    
    """Recompiles a unitary with respect to a statevector.
       Optionally provide `cache_options` to use the cache."""
    
    # If unitary is an identity matrix, no gates need to be added.
    # if not np.any(unitary - np.eye(unitary.shape[0])):
    #     return [], []

    # If cache read enabled, check if circuit is already cached.
    if cache_options is not None and cache_options['read']:
        quimb_gates = CacheRecompilation.load_recompiled_circuit(
            cache_options['hamiltonian'], cache_options['type'])
        if quimb_gates is not None:
            return quimb_gates

    # Proceed with recompilation.
    n_qubits = int(np.log2(len(statevector)))
    psi0 = qtn.Dense1D(statevector)
    ansatz_circ = build_ansatz_circuit(
        n_qubits, psi0=psi0, gate_1q=gate_1q,
        gate_2q=gate_2q, n_gate_rounds=n_gate_rounds, 
        periodic=periodic)
    if init_guess is not None:
        ansatz_circ.update_params_from(init_guess)
    psi_ansatz = ansatz_circ.psi

    psi_target = qtn.Dense1D(target_unitary @ statevector)

    def loss(psi_ansatz, psi_target):
        return 1 - abs(psi_ansatz.H @ psi_target) ** 2

    optimizer = TNOptimizer(
        psi_ansatz, loss_fn=loss,
        loss_constants={'psi_target': psi_target},
        constant_tags=[gate_2q, 'PSI0'],
        autograd_backend='jax', optimizer='L-BFGS-B')
    if init_guess is None:
        psi_recompiled = optimizer.optimize_basinhopping(n=500, nhop=10)
    else:
        psi_recompiled = optimizer.optimize(n=1000)

    ansatz_circ.update_params_from(psi_recompiled)
    quimb_gates = ansatz_circ.gates

    # If cache write enabled, save the circuit.
    if cache_options is not None and cache_options['write']:
        CacheRecompilation.save_recompiled_circuit(
            cache_options['hamiltonian'], cache_options['type'], quimb_gates)

    return quimb_gates


def apply_quimb_gates(quimb_gates, circ, reverse=False):
    """Apply quimb gates to a Qiskit circuit."""
    qreg = circ.qregs[0]
    if reverse:
        q_map = {len(qreg) - i - 1: qreg[i] for i in range(len(qreg))}
    else:
        q_map = {i: qreg[i] for i in range(len(qreg))}

    for gate in quimb_gates:
        if gate[0] == 'RY':
            circ.ry(gate[1], q_map[int(gate[2])])
        elif gate[0] == 'U3':
            circ.u3(gate[1], gate[2], gate[3], q_map[int(gate[4])])
        elif gate[0] == 'CNOT':
            circ.cx(q_map[int(gate[1])], q_map[int(gate[2])])
    return circ