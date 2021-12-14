"""recompilation.py from the dev branch."""

from typing import Union, Sequence, Optional, List, Tuple
from math import pi
import numpy as np

from qiskit import QuantumCircuit
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize_autograd import TNOptimizer

from utils import get_statevector, get_unitary


from io_utils import CacheRecompilation

QuimbGates = List[Tuple[str, Tuple[int]]]

class CircuitRecompiler:
    """A class for recompilation of quantum circuits."""

    def __init__(self,
                 gate_1q: str = 'RY',
                 gate_2q: str = 'CZ',
                 n_rounds: int = 3,
                 tol: float = 1e-8,
                 periodic: bool = False,
                 cache_options: Optional[dict] = None) -> None:
        """Creates a CircuitRecompiler object.

        Args:
            gate_1q: The single-qubit gate used in recompilation.
            gate_2q: The two-qubit gate used in recompilation.
            n_rounds: Number of gate rounds.
            periodic: Whether the system is periodic.
            cache_options: Options for saving and loading circuits.
        """
        assert gate_1q in ['U3', 'RY']
        assert gate_2q in ['CNOT', 'CZ']

        self.gate_1q = gate_1q
        self.gate_2q = gate_2q
        self.n_rounds = n_rounds
        self.tol = tol
        self.periodic = periodic
        self.cache_options = cache_options

    def _build_ansatz_circuit(self, n_qubits: int, psi0: qtn.Dense1D = None, n_rounds: int = 2) -> qtn.Circuit:
        """Constructs an ansatz circuit for recompilation."""
        # Initialize the Quimb circuit
        if psi0 is None:
            ansatz_circ = qtn.Circuit(n_qubits, tags='U0')
        else:
            ansatz_circ = qtn.Circuit(n_qubits, psi0=psi0, tags='PSI0')
        params = (0.,) if self.gate_1q == 'RY' else (0., 0., 0.)
        q_end = n_qubits if self.periodic else n_qubits - 1

        # Base layer of single-qubit gates
        for i in range(n_qubits):
            ansatz_circ.apply_gate(self.gate_1q, *params, i,
                                   gate_round=0, parametrize=True)

        # Subsequent rounds interleave single-qubit gate layer
        # and two-qubit gate layer
        for r in range(n_rounds):
            for i in range(r % 2, q_end, 2):
                ansatz_circ.apply_gate(self.gate_2q, (i+1) % n_qubits,
                                       i % n_qubits, gate_round=r)
                ansatz_circ.apply_gate(self.gate_1q, *params, i % n_qubits,
                                       gate_round=r, parametrize=True)
                ansatz_circ.apply_gate(self.gate_1q, *params, (i+1) % n_qubits,
                                       gate_round=r, parametrize=True)
        return ansatz_circ


    def recompile_all(self, circ) -> QuantumCircuit:
        """Recompiles all parts of a circuit between barriers."""
        data_all = [] # for statevector
        data_tmp = [] # for uni
        n_qubits = len(circ.qregs[0])
        circ_new = QuantumCircuit(len(circ.qregs[0]))

        circ_data = circ.data

        while len(circ_data) != 0:
            inst_tup = circ_data.pop(0)
            if inst_tup[0].name != 'barrier' and len(circ_data) != 0:
                # data_all.append(inst_tup)
                data_tmp.append(inst_tup)
            else:
                circ_new.barrier()
                if len(data_tmp) != 1:
                    uni = get_unitary(data_tmp, n_qubits=n_qubits)
                    sv = get_statevector(circ_new)

                    quimb_gates = self._recompile_with_statevector(uni, sv)
                    #quimb_gates = self._recompile_unitary(uni)
                    apply_quimb_gates(quimb_gates, circ_new)
                else:
                    try:
                        circ_new.append(*data_tmp[0])
                    except:
                        inst = data_tmp[0][0]
                        qargs = data_tmp[0][1]
                        qargs = [q.index for q in qargs]
                        circ_new.append(inst, qargs)
                data_tmp = []
        return circ_new



    def recompile(self,
                  targ_uni: np.ndarray,
                  data: Optional[np.ndarray] = None,
                  init_guess: Optional[qtn.Circuit] = None,
                  n_rounds: Optional[int] = None,
                  tol: Optional[float] = None) -> QuimbGates:
        """Recompiles a target unitary."""

        if data is not None:
            if len(data.shape) == 1:
                # Recompile with statevector
                quimb_gates = self._recompile_with_statevector(
                    targ_uni, data, init_guess=init_guess, n_rounds=n_rounds, tol=tol)
            else:
                # Recompile with density matrix
                quimb_gates = self._recompile_with_densitymatrix(
                    targ_uni, data, init_guess=init_guess)
        else:
            # Recompile the unitary itself
            quimb_gates = self._recompile_unitary(targ_uni, init_guess=init_guess, n_rounds=n_rounds, tol=tol)

        return quimb_gates

    def _recompile_with_statevector(self,
                                    targ_uni: np.ndarray,
                                    statevector: np.ndarray,
                                    init_guess: Optional[qtn.Circuit] = None,
                                    tol: Optional[float] = None,
                                    n_rounds: Optional[int] = None
                                    ) -> QuimbGates:
        """Recompiles a target unitary with respect to a statevector."""
        # If cache read enabled, check if circuit is already cached
        if self.cache_options is not None and self.cache_options['read']:
            quimb_gates = CacheRecompilation.load_recompiled_circuit(
                self.cache_options['hamiltonian'],
                self.cache_options['index'],
                self.cache_options['states'])
            if quimb_gates is not None:
                return quimb_gates

        print('!' * 80)
        print(targ_uni.shape)
        print(statevector.shape)

        if n_rounds is None:
            n_rounds = self.n_rounds
        if tol is None:
            tol = self.tol

        infid = 1.
        def infidelity(psi1, psi2):
            return abs(1 - abs(psi1.H @ psi2) ** 2)


        n_qubits = int(np.log2(len(statevector)))
        psi0 = qtn.Dense1D(statevector)
        psi_target = qtn.Dense1D(targ_uni @ statevector)

        while infid >= tol:
            print(f"Recompiling circuit in statevector mode with {n_rounds} gate rounds")
            # Construct ansatz circuit and update parameters with initial guess
            ansatz_circ = self._build_ansatz_circuit(n_qubits, psi0=psi0, n_rounds=n_rounds)
            if init_guess is not None:
                ansatz_circ.update_params_from(init_guess)

            # Define the ansatz and optimizer
            psi_ansatz = ansatz_circ.psi
            optimizer = TNOptimizer(
                psi_ansatz, loss_fn=infidelity,
                loss_constants={'psi2': psi_target},
                constant_tags=[self.gate_2q, 'PSI0'],
                autograd_backend='jax',
                optimizer='L-BFGS-B')

            # Carry out the optimization
            if init_guess is None:
                tn_opt, infid = optimizer.optimize_basinhopping(n=500, nhop=10)
            else:
                tn_opt, infid = optimizer.optimize(n=1000)

            n_rounds += 1

        # Extract the quimb gates from optimized parameters
        ansatz_circ.update_params_from(tn_opt)
        quimb_gates = ansatz_circ.gates

        # If cache write enabled, save the circuit
        if self.cache_options is not None and self.cache_options['write']:
            CacheRecompilation.save_recompiled_circuit(
                self.cache_options['hamiltonian'], self.cache_options['index'],
                self.cache_options['states'], quimb_gates)

        return quimb_gates

    def _recompile_with_densitymatrix(self, targ_uni, densitymatrix, init_guess=None):
        raise NotImplementedError("Recompilation with a density matrix is not implemented")

    def _recompile_unitary(self,
                           targ_uni: np.ndarray,
                           init_guess: Optional[qtn.Circuit] = None,
                           n_rounds: Optional[int] = None,
                           tol: Optional[float] = None,
                           ) -> QuimbGates:
        """Recompiles the target unitary itself."""
        if n_rounds is None:
            n_rounds = self.n_rounds
        if tol is None:
            tol = self.tol

        n_qubits = int(np.log2(targ_uni.shape[0]))

        infid = 1.
        def infidelity(U1, U2):
            return abs(1 - (U1 | U2).contract(all, optimize='auto-hq').real / 2**n_qubits)
        U_targ = qtn.Tensor(
            data=targ_uni.reshape([2] * (2 * n_qubits)),
            inds=[f'k{i}' for i in range(n_qubits)] + [f'b{i}' for i in range(n_qubits)],
            tags={'U_TARGET'})

        while infid >= tol:
            print(f"Recompiling circuit in unitary mode with {n_rounds} gate rounds")
            # Construct the ansatz circuit
            ansatz_circ = self._build_ansatz_circuit(n_qubits, n_rounds=n_rounds)
            if init_guess is not None:
                ansatz_circ.update_params_from(init_guess)


            # Define the ansatz and optimizer
            U_ansatz = ansatz_circ.uni
            optimizer = TNOptimizer(
                U_ansatz, loss_fn=infidelity,
                loss_constants={'U2': U_targ},
                constant_tags=[self.gate_2q, 'U0', 'U2'],
                autograd_backend='jax',
                optimizer='L-BFGS-B')

            # Carry out optimization
            if init_guess is None:
                tn_opt, infid = optimizer.optimize_basinhopping(n=500, nhop=100)
            else:
                tn_opt, infid = optimizer.optimize(n=1000)

            n_rounds += 1

        # Extract the quimb gates from optimized parameters
        ansatz_circ.update_params_from(tn_opt)
        quimb_gates = ansatz_circ.gates

        return quimb_gates

def apply_quimb_gates(quimb_gates: QuimbGates,
                      circ: QuantumCircuit,
                      reverse: bool = False):
    """Applies quimb gates to a Qiskit circuit."""
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
        elif gate[0] == 'CZ':
            circ.cz(q_map[int(gate[1])], q_map[int(gate[2])])
    return circ