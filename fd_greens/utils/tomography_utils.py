from typing import Optional
import itertools
import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

# Basis matrix for tomography
basis_matrix = []
bases = list(itertools.product('xyz', 'xyz', '01', '01'))
states = {'x0': np.array([1.0, 1.0]) / np.sqrt(2),
          'x1': np.array([1.0, -1.0]) / np.sqrt(2),
          'y0': np.array([1.0, 1.0j]) / np.sqrt(2),
          'y1': np.array([1.0, -1.0j]) / np.sqrt(2),
          'z0': np.array([1.0, 0.0]),
          'z1': np.array([0.0, 1.0])}

for basis in bases:
    label0 = ''.join([basis[0], basis[3]])
    label1 = ''.join([basis[1], basis[2]])
    state0 = states[label0]
    state1 = states[label1]
    state = np.kron(state1, state0)
    rho_vec = np.outer(state, state.conj()).reshape(-1)
    basis_matrix.append(rho_vec)

basis_matrix = np.array(basis_matrix)

def state_tomography(circ: QuantumCircuit,
                     q_instance: Optional[QuantumInstance] = None
                     ) -> np.ndarray:
    """Performs state tomography on a quantum circuit.

    Args:
        circ: The quantum circuit to perform tomography on.
        q_instance: The QuantumInstance to execute the circuit.

    Returns:
        The density matrix obtained from state tomography.
    """
    if q_instance is None:
        backend = Aer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend, shots=8192)

    qreg = circ.qregs[0]
    qst_circs = state_tomography_circuits(circ, qreg)
    if True:
        fig = qst_circs[5].draw(output='mpl')
        fig.savefig('qst_circ.png')
    result = q_instance.execute(qst_circs)
    qst_fitter = StateTomographyFitter(result, qst_circs)
    rho_fit = qst_fitter.fit(method='lstsq')
    return rho_fit