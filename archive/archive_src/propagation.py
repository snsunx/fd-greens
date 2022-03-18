import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm

from qiskit import *
from qiskit.quantum_info import Pauli
from qiskit.extensions import *
from qiskit.quantum_info.synthesis.two_qubit_decompose import *
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize_autograd import TNOptimizer
import cotengra as ctg

from subroutines import get_psi


def ansatz_circuit(n, D, n_layers, gate1: str ='RY', gate2: str = 'CNOT', psi=None, periodic=False):
    """Construct a circuit of single qubit and entangling gates.
    This circuit is optimized later by the compress_unitary function.

    Args:
        D: domain size
        n_layers: number of layers
        gate1: Single-qubit gate (default: 'RY')
        gate2: Two-qubit gate (default: 'CNOT')
        periodic: if periodic, insert a long-range two-qubit gate

    Returns:
        circ: Quimb circuit
    """
    if psi is not None:
        psi = qtn.Dense1D(psi)
        circ = qtn.Circuit(n, tags='PSI0', psi0=psi)
    else:
        circ = qtn.Circuit(n, tags='PSI0')

    if gate1 == 'U3':
        params = [0.0, 0.0, 0.0]
    elif gate1 == 'RY':
        params = [0.0]

    regs = list(reversed(range(D)))

    for i in range(len(regs)):
        circ.apply_gate(gate1, *params, regs[i], gate_round=0, parametrize=True)

    if periodic:
        end_q = D
    else:
        end_q = D - 1

    for r in range(n_layers):
        if r % 2 == 0:
            for j in range(0, end_q, 2):
                j1 = j % D
                j2 = (j + 1) % D
                circ.apply_gate(gate2, regs[j1], regs[j2], gate_round=r)
                circ.apply_gate(gate1, *params, regs[j1], gate_round=r, parametrize=True)
                circ.apply_gate(gate1, *params, regs[j2], gate_round=r, parametrize=True)
        else:
            for j in range(1, end_q, 2):
                j1 = j % D
                j2 = (j + 1) % D
                circ.apply_gate(gate2, regs[j1], regs[j2], gate_round=r)
                circ.apply_gate(gate1, *params, regs[j1], gate_round=r, parametrize=True)
                circ.apply_gate(gate1, *params, regs[j2], gate_round=r, parametrize=True)
    return circ

def compress_Upsi(D, psi, targ_psi, gate1='RY', n_layers=3):
    """Compresses U\ket{\psi}, where U is constructed from
    the exponentiated matrix of paulis and their corresponding coeffcients:
    U_ref = exp(-1j * A)
    Args:

    Returns:
        circ.gates: gates from Quimb with the optimized parameters for 1-qubit gates
        psi_opt_dense: array, optimized state
    """
    n = int(np.log2(psi.shape[0]))

    gate2 = 'CNOT'
    circ = ansatz_circuit(n, D, n_layers, psi=psi, gate1=gate1, gate2=gate2, periodic=False)
    psi = circ.psi

    targ_psi = qtn.Dense1D(targ_psi)

    def loss(psi, targ_psi):
        return -(psi.H @ targ_psi).real

    optmzr = TNOptimizer(
        psi, loss_fn=loss, loss_constants={'targ_psi': targ_psi},
        constant_tags=[gate2, 'PSI0'], autograd_backend='jax',
        optimizer='L-BFGS-B')

    psi_opt = optmzr.optimize_basinhopping(n=500, nhop=10)
    psi_opt_dense = psi_opt.to_dense()
    psi_opt_dense = np.asarray(psi_opt_dense).flatten()
    circ.update_params_from(psi_opt)

    return circ.gates, psi_opt_dense


def propagate_compressed(qreg, circ, psi, paulis, coeffs, richardson, gate1='RY', n_layers=3):
    r"""Propagate by a unitary operator optimized from
        exp(-i \sum_i x_i \sigma_i)

    Args:
        qreg (QuantumRegister): The quantum register
        circ (QuantumCircuit): The quantum circuit
        paulis ([Pauli]): A list of Paulis
        x (np.ndarray): The coefficients vector
        richardson (int): The number of redundant gates used in Richardson
            extrapolation

    Returns:
        circ_new (QuantumCircuit): The quantum circ after propagation gates
            are appended
    """
    if richardson is None:
        richardson = 0
        has_barrier = False
    else:
        has_barrier = True

    n = int(np.log2(psi.shape[0]))
    D = len(qreg)

    hermitian = coo_matrix((2 ** n, 2 ** n), dtype=np.complex128)
    for i, pauli in enumerate(paulis):
        hermitian += coeffs[i] * pauli.to_matrix()
    U = expm(-1j * hermitian)

    #print('initial psi', np.linalg.norm(psi), psi)
    targ_psi = U @ psi
    #print('target psi', np.linalg.norm(targ_psi), targ_psi)
    gates, psi_opt_dense = compress_Upsi(D, psi, targ_psi, gate1=gate1, n_layers=n_layers)
    #print("*" * 80, "\n", targ_psi)
    #print("*" * 80, "\n", psi_opt_dense)

    q_map = {len(qreg) - i - 1: qreg[i] for i in range(len(qreg))}
    for gate in gates:
        if gate[0] == 'RY':
            for _ in range(2 * richardson + 1):
                circ.ry(gate[1] / (2 * richardson + 1), q_map[int(gate[2])])
                if has_barrier:
                    circ.barrier()
        elif gate[0] == 'U3': # XXX: Not correct
            for _ in range(2 * richardson + 1):
                circ.u3(gate[1] / (2 * richardson + 1),
                        gate[2] / (2 * richardson + 1),
                        gate[3] / (2 * richardson + 1),
                        q_map[int(gate[4])])
                if has_barrier:
                    circ.barrier()
        elif gate[0] == 'CNOT':
            for _ in range(2 * richardson + 1):
                circ.cx(q_map[int(gate[1])], q_map[int(gate[2])])
                if has_barrier:
                    circ.barrier()
    return circ