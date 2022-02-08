"""Utility functions"""

import os
import h5py
from itertools import product
import math
from typing import Optional, Union, Iterable, List, Tuple, Sequence, Mapping
import numpy as np
from collections import defaultdict
from hamiltonians import MolecularHamiltonian

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.utils import QuantumInstance
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.circuit import Instruction, Qubit, Clbit
from qiskit.opflow import PauliSumOp
from qiskit.result import Result, Counts

QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]
QuantumCircuitLike = Union[QuantumCircuit, Iterable[InstructionTuple]]

# Functions to obtain physical quantites
def get_statevector(circ_like: QuantumCircuitLike, reverse: bool = False) -> np.ndarray:
    """Returns the statevector of a quantum circuit.

    Args:
        circ_like: The circuit or instruction tuples on which the state is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        The statevector array of the circuit.
    """
    if isinstance(circ_like, QuantumCircuit):
        circ = circ_like
    else: # instruction tuples
        circ = create_circuit_from_inst_tups(circ_like)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    statevector = result.get_statevector()
    if reverse:
        statevector = reverse_qubit_order(statevector)
    return statevector

def get_unitary(circ_like: QuantumCircuitLike, reverse: bool = False) -> np.ndarray:
    """Returns the unitary of a quantum circuit.

    Args:
        circ_like: The circuit or instruction tuples on which the unitary is to be obtained.
        reverse: Whether qubit order is reversed.

    Returns:
        The unitary array of the circuit.
    """
    circ_like = remove_instructions(circ_like, ['barrier', 'measure'])
    if isinstance(circ_like, QuantumCircuit):
        circ = circ_like
    else: # instruction tuples
        circ = create_circuit_from_inst_tups(circ_like)

    backend = Aer.get_backend('unitary_simulator')
    result = execute(circ, backend).result()
    unitary = result.get_unitary()
    if reverse:
        unitary = reverse_qubit_order(unitary)
    return unitary

def compare_matrices(mat1: np.ndarray, mat2: np.ndarray) -> bool:
    """Determines whether two matrices are equal up to a phase.
    
    Args:
        mat1: The first matrix.
        mat2: The second matrix.
        
    Returns:
        Whether two matrices are equal up to a phase.
    """
    phase1 = mat1[0, 0] / abs(mat1[0, 0])
    phase2 = mat2[0, 0] / abs(mat2[0, 0])
    mat1_new = mat1 / phase1
    mat2_new = mat2 / phase2
    equal = np.allclose(mat1_new, mat2_new)
    return equal

def get_overlap(state1: np.ndarray, state2: np.ndarray) -> float:
    """Returns the overlap of two states in either statevector or density matrix form.
    
    Args:
        state1: A 1D or 2D numpy array corresponding to the first state.
        state2: A 1D or 2D numpy array corresponding to the second state.

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

# Circuit utility functions
def remove_instructions_in_circuit(circ_like: QuantumCircuitLike,
                        instructions: Iterable[str]
                        ) -> QuantumCircuitLike:
    """Removes certain instructions in a circuit.
    
    Args:
        circ_like: The circuit or instruction tuples on which certain instructions are removed.
        instructions: An iterable of strings representing instruction names.
        
    Returns:
        A new quantum circuit on which certain instructions are removed.
    """
    if isinstance(circ_like, QuantumCircuit):
        inst_tups = circ_like.data.copy()
    else:
        inst_tups = circ_like
    qreg, creg = get_registers_in_inst_tups(inst_tups)
    # qreg = circ.qregs[0] if circ.qregs != [] else None
    # creg = circ.cregs[0] if circ.cregs != [] else None

    inst_tups_new = []
    for inst_tup in inst_tups:
        if inst_tup[0].name not in instructions:
            inst_tups_new.append(inst_tup)

    if isinstance(circ_like, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new, qreg=qreg, creg=creg)
        return circ_new
    else:
        return inst_tups_new
remove_instructions = remove_instructions_in_circuit

def get_registers_in_circuit(circ_like: QuantumCircuitLike
                            ) -> Tuple[QuantumRegister, ClassicalRegister]:
    """Returns the quantum and classical registers from instruction tuples. 
    
    Qubits and classical bits can be specified either as Qubit/Clbit instances or as 
    integers. If registers not found, return as None.

    Args:
        The instruction tuples from which qreg and creg are extracted.

    Returns:
        qreg: The quantum register in the instruction tuples.
        creg: The classical register in the instruction tuples.
    """
    if isinstance(circ_like, QuantumCircuit):
        inst_tups = circ_like.data.copy()
    else:
        inst_tups = circ_like

    n_qubits = 0
    n_clbits = 0
    for inst_tup in inst_tups:
        _, qargs, cargs = inst_tup
        if qargs != []: 
            if isinstance(qargs[0], int): # int
                max_q = max(qargs)
            else: # Qubit
                max_q = max([q._index for q in qargs])
            if max_q + 1 > n_qubits:
                n_qubits = max_q + 1
        if cargs != []:
            if isinstance(cargs[0], int): # int
                max_c = max(cargs)
            else: # Qubit
                max_c = max([c._index for c in cargs])
            if max_c + 1 > n_clbits: 
                n_clbits = max_c + 1

    # If specified as int, create new qreg and creg
    if n_qubits > 0:
        qreg = QuantumRegister(n_qubits, name='q')
    else:
        qreg = None
    if n_clbits > 0:
        creg = ClassicalRegister(n_clbits, name='c')
    else:
        creg = None
    return qreg, creg
get_registers_in_inst_tups = get_registers_in_circuit

def create_circuit_from_inst_tups(
        inst_tups: Iterable[InstructionTuple],
        qreg: Optional[QuantumRegister] = None,
        creg: Optional[ClassicalRegister] = None) -> QuantumCircuit:
    """Creates a circuit from instruction tuples.
    
    Args:
        inst_tups: Instruction tuples from which the circuit is to be constructed.
        n_qubits: Number of qubits.
        qreg: The quantum register.
        
    Returns:
    	A quantum circuit constructed from the instruction tuples.
    """
    if qreg is None and creg is None:
        qreg, creg = get_registers_in_inst_tups(inst_tups)
    regs = [reg for reg in [qreg, creg] if reg is not None]
    circ = QuantumCircuit(*regs)
    for inst, qargs, cargs in inst_tups:
        # print('inst =', inst.name, inst.params, 'qargs =', qargs)
        try:
            qargs = [q._index for q in qargs]
        except:
            pass
        circ.append(inst, qargs, cargs)
    return circ

def split_circuit_across_barriers(circ: QuantumCircuit) -> List[List[InstructionTuple]]:
    """Splits a circuit into instruction tuples across barriers."""
    inst_tups = circ.data.copy()
    inst_tups_all = [] # all inst_tups_single
    inst_tups_single = [] # temporary variable to hold inst_tups_all components

    # Split when encoutering a barrier
    for i, inst_tup in enumerate(inst_tups):
        if inst_tup[0].name == 'barrier': # append and start a new inst_tups_single
            inst_tups_all.append(inst_tups_single)
            inst_tups_single = []
        elif i == len(inst_tups) - 1: # append and stop
            inst_tups_single.append(inst_tup)
            inst_tups_all.append(inst_tups_single)
        else: # just append
            inst_tups_single.append(inst_tup)

    return inst_tups_all


# Other utility functions
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

def solve_energy_probabilities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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

# Counts utility functions
def get_counts(result: Result) -> Mapping[str, int]:
    """Returns the counts from a Result object with default set to 0."""
    counts = defaultdict(lambda: 0)
    counts.update(result.get_counts())
    return counts
get_counts_default_0 = get_counts

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
        n_counts = np.sum(counts)
        counts0 = counts0 / n_counts
        counts1 = counts1 / n_counts
        print('np.sum(counts) =', np.sum(counts))
        print('##########################################################')
        print(counts0, np.sum(counts0))
        print(counts1, np.sum(counts1))
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

def get_counts_from_key(counts, anc_inds, anc_loc):
    if isinstance(counts, Counts):
        counts = counts_dict_to_arr(counts)
    n_tot = int(np.log2())
    sys_inds = list(product([0, 1], repeat=n_tot))
    
    return

# HDF5 utility function
def initialize_hdf5(fname: str = 'lih') -> None:
    """Creates the hdf5 file and dataset if they do not exist."""
    fname += '.h5'
    if os.path.exists(fname): f = h5py.File(fname, 'r+')
    else: f = h5py.File(fname, 'w')
    for gname in ['gs', 'eh', 'amp', 'circ0', 'circ1', 'circ01']:
        if gname in f.keys(): del f[gname]
        f.create_group(gname)
    f.close()

'''
def check_ccx_inst_tups(inst_tups):
    """Checks whether the instruction tuples are equivalent to CCX up to a phase."""
    ccx_inst_tups_matrix = get_unitary(ccx_inst_tups)
    self.ccx_angle = polar(ccx_inst_tups_matrix[3, 7])[1]
    ccx_inst_tups_matrix[3, 7] /= np.exp(1j * self.ccx_angle)
    ccx_inst_tups_matrix[7, 3] /= np.exp(1j * self.ccx_angle)
    ccx_matrix = CCXGate().to_matrix()
    assert np.allclose(ccx_inst_tups_matrix, ccx_matrix)
'''

# Helper functions
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
        q_instance = QuantumInstance(
            Aer.get_backend('qasm_simulator', shots=10000),
            shots=10000)
    elif type_str == 'noisy':
        q_instance = QuantumInstance(
            Aer.get_backend('qasm_simulator', shots=100000, noise_model_name='ibmq_jakarta'),
            shots=100000)
    return q_instance