"""Utility functions"""

import os
import h5py
import math
from typing import Optional, Union, Iterable, List, Tuple, Sequence, Mapping
import numpy as np
from collections import defaultdict
from hamiltonians import MolecularHamiltonian

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.utils import QuantumInstance
from qiskit.ignis.verification.tomography import (state_tomography_circuits,
                                                  StateTomographyFitter)
from qiskit.circuit import Instruction
from qiskit.opflow import PauliSumOp
from qiskit.result import Result, Counts

CircuitData = Iterable[Tuple[Instruction, List[int], Optional[List[int]]]]

def get_statevector(circ: QuantumCircuit,
                    reverse: bool = False) -> np.ndarray:
    """Returns the statevector of a quantum circuit.

    Args:
        circ: The circuit on which the state is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        The statevector array of the circuit.
    """
    if isinstance(circ, list): # CircuitData
        circ = data_to_circuit(circ)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    statevector = result.get_statevector()
    if reverse:
        statevector = reverse_qubit_order(statevector)
    return statevector

def get_unitary(circ: Union[QuantumCircuit, CircuitData],
                reverse: bool = False, n_qubits = None) -> np.ndarray:
    """Returns the unitary of a quantum circuit.

    Args:
        circ: The circuit on which the unitary is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        The unitary array of the circuit.
    """

    if isinstance(circ, list): # CircuitData
        circ = data_to_circuit(circ, n_qubits=n_qubits)
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circ, backend)
    result = job.result()
    unitary = result.get_unitary()
    if reverse:
        unitary = reverse_qubit_order(unitary)
    return unitary

def remove_barriers(circ_data):
    """Removes barriers in circuit data."""
    circ_data_new = []
    for inst_tup in circ_data:
        if inst_tup[0].name != 'barrier':
            circ_data_new.append(inst_tup)
    return circ_data_new

def data_to_circuit(data, n_qubits=None, remove_barr=True):
    """Converts circuit data to circuit."""
    if remove_barr:
        data = remove_barriers(data)
    if n_qubits is None:
        try:
            n_qubits = max([max(x[1]) for x in data]) + 1
        except:
            n_qubits = max([max([y.index for y in x[1]]) for x in data]) + 1
    circ_new = QuantumCircuit(n_qubits)
    for inst_tup in data:
        inst, qargs = inst_tup[:2]
        try:
            circ_new.append(inst, qargs)
        except:
            qargs = [q.index for q in qargs]
            circ_new.append(inst, qargs)
    return circ_new

def reverse_qubit_order(arr: np.ndarray) -> np.ndarray:
    """Reverses qubit order in a 1D or 2D array.

    Args:
        The array on which the qubit order is to be reversed.

    Returns:
        The array after qubit order is reversed.
    """
    if len(arr.shape) == 1:
        dim = arr.shape[0]
        num = int(np.log2(dim))
        shape_expanded = (2,) * num
        inds_transpose = np.flip(np.arange(num))
        arr = arr.reshape(*shape_expanded)
        arr = arr.transpose(*inds_transpose)
        arr = arr.reshape(dim)
    elif len(arr.shape) == 2:
        shape_original = arr.shape
        dim = shape_original[0]
        num = int(np.log2(dim))
        shape_expanded = (2,) * 2 * num
        inds_transpose = list(reversed(range(num))) + list(reversed(range(num, 2 * num)))
        arr = arr.reshape(*shape_expanded)
        arr = arr.transpose(*inds_transpose)
        arr = arr.reshape(*shape_original)
    else:
        raise NotImplementedError("Reversing qubit order of array with more"
                                  "than two dimensions is not implemented.")
    return arr

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

def get_lih_hamiltonian(r: float) -> MolecularHamiltonian:
    """Returns the HOMO-LUMO LiH Hamiltonian with bond length r."""
    hamiltonian = MolecularHamiltonian(
        [['Li', (0, 0, 0)], ['H', (0, 0, r)]], 'sto3g', 
        occ_inds=[0], act_inds=[1, 2])
    return hamiltonian

def get_quantum_instance(type_str: str) -> QuantumInstance:
    """Returns the QuantumInstance from type string."""
    if type_str == 'sv':
        q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    elif type_str == 'qasm':
        q_instance = QuantumInstance(Aer.get_backend('qasm_simulator', shots=10000), shots=10000)
    elif type_str == 'noisy':
        q_instance = QuantumInstance(Aer.get_backend('qasm_simulator', shots=10000, noise_model_name='ibmq_jakarta'), shots=10000)
    return q_instance

def save_circuit(circ: QuantumCircuit, 
                 fname: str,
                 savetxt: bool = True,
                 savefig: bool = True) -> None:
    """Saves a circuit to disk in QASM string form and/or figure form.
    
    Args:
        fname: The file name.
        savetxt: Whether to save the QASM string of the circuit as a text file.
        savefig: Whether to save the figure of the circuit.
    """
        
    if savefig:
        fig = circ.draw(output='mpl')
        fig.savefig(fname + '.png')
    
    if savetxt:
        circ_data = []
        for inst_tup in circ.data:
            if inst_tup[0].name != 'c-unitary':
                circ_data.append(inst_tup)
        circ = data_to_circuit(circ_data, remove_barr=False)
        # for inst_tup in circ_data:
        #    print(inst_tup[0].name)
        # exit()
        f = open(fname + '.txt', 'w')
        qasm_str = circ.qasm()
        f.write(qasm_str)
        f.close()

def solve_energy_probabilities(a, b):
    A = np.array([[1, 1], [a[0], a[1]]])
    x = np.linalg.inv(A) @ np.array([1.0, b])
    return x

def measure_operator(circ: QuantumCircuit,
                     op: PauliSumOp,
                     q_instance: QuantumInstance,
                     anc_state: Sequence[int] = [],
                     qreg: Optional[QuantumRegister] = None) -> float:
    """Measures an operator on a circuit.

    Args:
        circ: The quantum circuit to be measured.
        op: The operator to be measured.
        q_instance: The QuantumInstance on which to execute the circuit.
        anc_state: A sequence of integers indicating the ancilla states.
        qreg: The quantum register on which the operator is measured.
    
    Returns:
        The value of the operator on the circuit.
    """
    if qreg is None:
        qreg = circ.qregs[0]
    n_qubits = len(qreg)
    n_anc = len(anc_state)
    n_sys = n_qubits - n_anc

    circ_list = []
    for term in op:
        label = term.primitive.table.to_labels()[0]
        
        # Create circuit for measuring this Pauli string
        circ_meas = circ.copy()
        creg = ClassicalRegister(n_qubits)
        circ_meas.add_register(creg)

        for i in range(n_anc):
            circ_meas.measure(i, i)
        
        for i in range(n_sys):
            if label[n_sys - 1 - i] == 'X':
                circ_meas.h(qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == 'Y':
                circ_meas.rx(np.pi / 2, qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == 'Z':
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])

        circ_list.append(circ_meas)
    
    result = q_instance.execute(circ_list)
    counts_list = result.get_counts()

    value = 0
    for i, counts in enumerate(counts_list):
        # print('counts =', counts)
        coeff = op.primitive.coeffs[i]
        counts_new = counts.copy()

        for key, val in counts.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if key_anc != anc_state:
                del counts_new[key]
        
        shots = sum(counts_new.values())
        for key, val in counts_new.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if sum(key_sys) % 2 == 0:
                value += coeff * val / shots
            else:
                value -= coeff * val / shots
    return value

def get_overlap(state1: np.ndarray, state2: np.ndarray) -> float:
    """Returns the overlap of two states in either statevector or density matrix form.
    
    Args:
        state1: The numpy array corresponding to the first state.
        state2: The numpy array corresponding to the second state.

    Returns:
        The overlap between the two states.
    """
    if len(state1.shape) == 1 and len(state2.shape) == 1:
        return abs(state1.conj() @ state2) ** 2
    
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        return (state1.conj() @ state2 @ state1).real
    
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        return (state2.conj() @ state1 @ state2).real

    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        return np.trace(state1.conj().T @ state2).real

def get_counts(result: Result) -> Mapping[str, int]:
    """Returns the counts from a Result object with default set to 0."""
    counts = defaultdict(lambda: 0)
    counts.update(result.get_counts())
    return counts


def counts_dict_to_arr(counts: Counts, n_qubits: int = None) -> np.ndarray:
    """Converts counts from dictionary form to array form."""
    if n_qubits is None:
        n_qubits = math.ceil(np.log2(max(counts.keys())))

    arr = np.zeros((2 ** n_qubits,))
    for key, val in counts.items():
        arr[key] = val
    return arr

def counts_arr_to_dict(arr: np.ndarray) -> np.ndarray:
    """Converts counts from array form to dictionary form."""
    data = {}
    for i in range(arr.shape[0]):
        data[i] = arr[i]

    counts = Counts(data)
    return counts

def split_counts_on_anc(counts: Union[Counts, np.ndarray], n_anc: int = 1) -> Counts:
    """Splits the counts on ancilla qubit state."""
    if isinstance(counts, Counts):
        counts = counts_dict_to_arr(counts)
    step = 2 ** n_anc
    if n_anc == 1:
        counts0 = counts[::step]
        counts1 = counts[1::step]
        counts0 /= np.sum(counts0)
        counts1 /= np.sum(counts1)
        return counts0, counts1
    elif n_anc == 2:
        counts00 = counts[::step]
        counts01 = counts[1::step]
        counts10 = counts[2::step]
        counts11 = counts[3::step]
        counts00 /= np.sum(counts00)
        counts01 /= np.sum(counts01)
        counts10 /= np.sum(counts10)
        counts11 /= np.sum(counts11)
        return counts00, counts01, counts10, counts11


def save_eh_data(gs_solver: 'GroundStateSolver', 
                 es_solver: 'EHStatesSolver',
                 amp_solver: 'EHAmplitudesSolver',
                 fname: str = 'lih',
                 dsetname: str = 'eh') -> None:
    """Saves N+/-1 electron states data to file.
    
    Args:
        gs_solver: The ground state solver.
        es_solver: The N+/-1 electron states solver.
        amp_solver: The transition amplitudes solver.
        fname: The file name string.
        dsetname: The dataset name string.
    """
    fname += '.hdf5'
    if os.path.exists(fname):
        f = h5py.File(fname, 'r+')
    else:
        f = h5py.File(fname, 'w')
    if dsetname in f.keys(): 
        dset = f[dsetname]
    else: 
        dset = f.create_dataset(dsetname, shape=())
    dset.attrs['energy_gs'] = gs_solver.energy
    dset.attrs['energies_e'] = es_solver.energies_e
    dset.attrs['energies_h'] = es_solver.energies_h
    dset.attrs['B_e'] = amp_solver.B_e
    dset.attrs['B_h'] = amp_solver.B_h
    e_orb = np.diag(amp_solver.h.molecule.orbital_energies)
    act_inds = amp_solver.h.act_inds
    dset.attrs['e_orb'] = e_orb[act_inds][:, act_inds]
    f.close()

def save_exc_data(gs_solver: 'GroundStateSolver', 
                  es_solver: 'ExcitedStatesSolver',
                  amp_solver: 'ExcitedAmplitudesSolver',
                  fname: str = 'lih',
                  dsetname: str = 'exc') -> None:
    """Saves excited states data to file.
    
    Args:
        gs_solver: The ground state solver.
        es_solver: The excited states solver.
        amp_solver: The transition amplitudes solver.
        fname: The file name string.
        dsetname: The dataset name string.
    """
    fname += '.hdf5'
    if os.path.exists(fname):
        f = h5py.File(fname, 'r+')
    else:
        f = h5py.File(fname, 'w')
    if dsetname in f.keys():
        dset = f[dsetname]
    else:
        dset = f.create_dataset(dsetname, shape=())
    dset.attrs['energy_gs'] = gs_solver.energy
    dset.attrs['energies_s'] = es_solver.energies_s
    dset.attrs['energies_t'] = es_solver.energies_t
    dset.attrs['L'] = amp_solver.L
    dset.attrs['n_states'] = amp_solver.n_states
    f.close()
