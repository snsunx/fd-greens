
'''
def build_diagonal_circuits(ansatz, ind, measure=False):
    """Returns the quantum circuits to calculate diagonal elements of the 
    Green's functions."""
    # Create a new circuit with the ancilla as qubit 0
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 1)
    creg_anc = ClassicalRegister(1)
    creg_sys = ClassicalRegister(n_qubits) 
    circ = QuantumCircuit(qreg, creg_anc, creg_sys)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 1 for qubit in qargs]
        circ.append(inst, qargs, cargs)
    
    # TODO: The following way of writing is not efficient
    arr = np.zeros((n_qubits,))
    arr[ind] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms = list(qubit_op.terms)

    circ.barrier()
    circ.h(0)
    circ.barrier()
    apply_cU(circ, terms[0], ctrl=0, offset=1)
    circ.barrier()
    apply_cU(circ, terms[1], ctrl=1, offset=1)
    circ.barrier()
    circ.h(0)
    circ.barrier()

    if measure:
        circ.measure(qreg[0], creg_anc)
    # TODO: Implement phase estimation on the system qubits

    return circ
'''

'''
def build_off_diagonal_circuits(ansatz, ind_left, ind_right, add_barriers=True, measure=True):
    """Returns the quantum circuits to calculate off-diagonal 
    elements of the Green's function."""
    # Create a new circuit with the ancillas as qubits 0 and 1
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 2)
    creg = ClassicalRegister(2)
    # creg_sys = ClassicalRegister(n_qubits)
    circ = QuantumCircuit(qreg, creg)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 2 for qubit in qargs]
        circ.append(inst, qargs, cargs)

    # Define the creation/annihilation term to be appended
    # TODO: The following can be written more efficiently
    arr = np.zeros((n_qubits,))
    arr[ind_left] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms_left = list(qubit_op.terms)

    arr = np.zeros((n_qubits,))
    arr[ind_right] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    terms_right = list(qubit_op.terms)

    # Build the circuit
    circ.barrier()
    circ.h([0, 1])
    circ.barrier()
    apply_ccU(circ, terms_left[0], ctrl=(0, 0))
    circ.barrier()
    apply_ccU(circ, terms_left[1], ctrl=(1, 0))
    circ.barrier()
    circ.rz(np.pi / 4, qreg[1])
    circ.barrier()
    apply_ccU(circ, terms_right[0], ctrl=(0, 1))
    circ.barrier()
    apply_ccU(circ, terms_right[1], ctrl=(1, 1))
    circ.barrier()
    circ.h([0, 1])
    if add_barriers:
        circ.barrier()

    if measure:
        circ.measure([0, 1], [0, 1])

    return circ 
'''



    @staticmethod
    def get_hamiltonian_shift_parameters(hamiltonian: MolecularHamiltonian,
                                         states_str: str = 'e'
                                         ) -> Tuple[float, float]:
        """Obtains the scaling factor and constant shift of the Hamiltonian 
        for phase estimation.
        
        Args:
            hamiltonian: The MolecularHamiltonian object.
            states_str: 'e' or 'h', which indicates whether the parameters 
                are to be obtained for the (N+1)- or the (N-1)-electron states.
        
        Returns:
            scaling: The scaling factor of the Hamiltonian.
            shift: The constant shift factor of the Hamiltonian.
        """
        assert states_str in ['e', 'h']

        # Obtain the scaling factor
        greens_function = GreensFunction(None, hamiltonian)
        greens_function.compute_eh_states()
        if states_str == 'h':
            E_low = greens_function.eigenenergies_h[0]
            E_high = greens_function.eigenenergies_h[2]
        elif states_str == 'e':
            E_low = greens_function.eigenenergies_e[0]
            E_high = greens_function.eigenenergies_e[2]
        scaling = np.pi / (E_high - E_low)

        # Obtain the constant shift factor
        greens_function = GreensFunction(
            None, hamiltonian, scaling=scaling)
        greens_function.compute_eh_states()
        if states_str == 'h':
            shift = -greens_function.eigenenergies_h[0]
        elif states_str == 'e':
            shift = -greens_function.eigenenergies_e[0]

        return scaling, shift