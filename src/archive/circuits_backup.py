# XXX: Deprecated.
def get_diagonal_circuits1(ansatz, ind, measure=True):
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
    
    shape_expanded = (2,) * (ind + 1) * 2
    transpose_inds = np.hstack((np.flip(np.arange(ind + 1)), 
                                np.flip(np.arange(ind + 1) + ind + 1)))
    shape_compact = (2 ** (ind + 1), 2 ** (ind + 1))

    arr = np.zeros((n_qubits,))
    arr[ind] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    # TODO: The following is one way to construct the U0 and U1 gates
    for key, val in qubit_op.terms.items():
        if np.isreal(val):
            U0_op = 2 * val * QubitOperator(key)
            U0_mat = np.asarray(get_sparse_operator(U0_op).todense())
            U0_mat = U0_mat.reshape(*shape_expanded)
            U0_mat = U0_mat.transpose(*transpose_inds)
            U0_mat = U0_mat.reshape(*shape_compact)
            U0_mat = np.kron(np.eye(2 ** (n_qubits - ind - 1)), U0_mat)
            U0_gate = UnitaryGate(U0_mat)
            cU0_gate = U0_gate.control(ctrl_state=0)
        else:
            U1_op = 2 * val * QubitOperator(key)
            U1_mat = np.asarray(get_sparse_operator(U1_op).todense())
            U1_mat = U1_mat.reshape(*shape_expanded)
            U1_mat = U1_mat.transpose(*transpose_inds)
            U1_mat = U1_mat.reshape(*shape_compact)
            U1_mat = np.kron(np.eye(2 ** (n_qubits - ind - 1)), U1_mat)
            U1_gate = UnitaryGate(U1_mat)
            cU1_gate = U1_gate.control(ctrl_state=1)

    circ.h(0)
    circ.append(cU0_gate, qreg)
    circ.append(cU1_gate, qreg)
    circ.h(0)
    if measure:
        circ.measure(qreg[0], creg_anc)
    # TODO: Implement phase estimation on the system qubits
    
    return circ


# XXX: Deprecated
def get_off_diagonal_circuits1(ansatz, ind_left, ind_right, measure=True):
    """Returns the quantum circuits to calculate off-diagonal 
    elements of the Green's function."""
    # Create a new circuit with the ancillas as qubits 0 and 1
    n_qubits = ansatz.num_qubits
    qreg = QuantumRegister(n_qubits + 2)
    creg_anc = ClassicalRegister(2)
    creg_sys = ClassicalRegister(n_qubits)
    circ = QuantumCircuit(qreg, creg_anc, creg_sys)
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + 2 for qubit in qargs]
        circ.append(inst, qargs, cargs)


    shape_expanded = (2,) * (ind_left + 1) * 2
    transpose_inds = np.hstack((np.flip(np.arange(ind_left + 1)), 
                                np.flip(np.arange(ind_left + 1) + ind_left + 1)))
    shape_compact = (2 ** (ind_left + 1), 2 ** (ind_left + 1))
    arr = np.zeros((n_qubits,))
    arr[ind_left] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    for key, val in qubit_op.terms.items():
        if np.isreal(val):
            U0_op = 2 * val * QubitOperator(key)
            U0_mat = np.asarray(get_sparse_operator(U0_op).todense())
            U0_mat = U0_mat.reshape(*shape_expanded)
            U0_mat = U0_mat.transpose(*transpose_inds)
            U0_mat = U0_mat.reshape(*shape_compact)
            U0_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_left)), U0_mat)
            U0_gate = UnitaryGate(U0_mat)
            cU0_gate_left = U0_gate.control(num_ctrl_qubits=2, ctrl_state='00')
        else:
            U1_op = 2 * val * QubitOperator(key)
            U1_mat = np.asarray(get_sparse_operator(U1_op).todense())
            U1_mat = U1_mat.reshape(*shape_expanded)
            U1_mat = U1_mat.transpose(*transpose_inds)
            U1_mat = U1_mat.reshape(*shape_compact)
            U1_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_left)), U1_mat)
            U1_gate = UnitaryGate(U1_mat)
            cU1_gate_left = U1_gate.control(num_ctrl_qubits=2, ctrl_state='01')

    shape_expanded = (2,) * (ind_right + 1) * 2
    transpose_inds = np.hstack((np.flip(np.arange(ind_right + 1)), 
                                np.flip(np.arange(ind_right + 1) + ind_right + 1)))
    shape_compact = (2 ** (ind_right + 1), 2 ** (ind_right + 1))
    arr = np.zeros((n_qubits,))
    arr[ind_right] = 1.
    poly_tensor = PolynomialTensor({(0,): arr})
    ferm_op = get_fermion_operator(poly_tensor)
    qubit_op = jordan_wigner(ferm_op)
    for key, val in qubit_op.terms.items():
        if np.isreal(val):
            U0_op = 2 * val * QubitOperator(key)
            U0_mat = np.asarray(get_sparse_operator(U0_op).todense())
            U0_mat = U0_mat.reshape(*shape_expanded)
            U0_mat = U0_mat.transpose(*transpose_inds)
            U0_mat = U0_mat.reshape(*shape_compact)
            U0_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_right)), U0_mat)
            U0_gate = UnitaryGate(U0_mat)
            cU0_gate_right = U0_gate.control(num_ctrl_qubits=2, ctrl_state='10')
        else:
            U1_op = 2 * val * QubitOperator(key)
            U1_mat = np.asarray(get_sparse_operator(U1_op).todense())
            U1_mat = U1_mat.reshape(*shape_expanded)
            U1_mat = U1_mat.transpose(*transpose_inds)
            U1_mat = U1_mat.reshape(*shape_compact)
            U1_mat = np.kron(np.eye(2 ** (n_qubits - 1 - ind_right)), U1_mat)
            U1_gate = UnitaryGate(U1_mat)
            cU1_gate_right = U1_gate.control(num_ctrl_qubits=2, ctrl_state='11')

    # Build the circuit
    circ.h(qreg[:2])
    circ.append(cU0_gate_left, qreg)
    circ.append(cU1_gate_left, qreg)
    circ.rz(np.pi / 4, qreg[1])
    circ.append(cU0_gate_right, qreg)
    circ.append(cU1_gate_right, qreg)
    circ.h(qreg[:2])
    if measure:
        circ.measure(qreg[:2], creg_anc)
    # TODO: Implement phase estimation on the system qubits

    return circ 

'''
# TODO: Pass in n_gate_rounds and cache_options in a simpler way
def append_qpe_circuit(circ: QuantumCircuit,
                       hamiltonian_arr: np.ndarray,
                       ind_qpe: int,
                       recompiled: bool = False,
                       n_gate_rounds=None,
                       cache_options=None
                       ) -> QuantumCircuit:
    """Appends single-qubit QPE circuit to a given circuit.

    Args:
        circ: The quantum circuit on which the QPE circuit is to be appended.
        hamiltonian_arr: The Hamiltonian in array form.
        ind_qpe: Index of the QPE ancilla qubit.
        recompiled: Whether the controlled e^{iHt} gate is recompiled.

    Returns:
        A new quantum circuit on which the QPE circuit has been appended.
    """
    U_mat = expm(1j * hamiltonian_arr * HARTREE_TO_EV)
    n_sys = int(np.log2(U_mat.shape[0]))
    n_all = len(circ.qregs[0])
    n_anc = n_all - n_sys
    circ = copy_circuit_with_ancilla(circ, [ind_qpe])

    if recompiled:
        # Construct the controlled e^{iHt}
        cU_mat = np.kron(np.diag([1, 0]), np.eye(2 ** n_sys)) \
                + np.kron(np.diag([0, 1]), U_mat)
        cU_mat = np.kron(np.eye(2 ** n_anc), cU_mat)

        # Append single-qubit QPE circuit
        circ.barrier()
        circ.h(ind_qpe)
        statevector = reverse_qubit_order(get_statevector(circ))
        quimb_gates = recompile_with_statevector(
            statevector, cU_mat, n_gate_rounds=n_gate_rounds, cache_options=cache_options)
        circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=False)
        circ.h(ind_qpe)
        circ.barrier()
    else:
        # Construct the controlled e^{iHt}
        cU_gate = UnitaryGate(U_mat).control(1)

        # Append single-qubit QPE circuit
        circ.barrier()
        circ.h(ind_qpe)
        circ.append(cU_gate, np.arange(n_sys + 1) + n_anc)
        circ.h(ind_qpe)
        circ.barrier()
    return circ
'''

class CircuitTranspiler:
    """A class for circuit transpilation."""

    def __init__(self, 
                 basis_gates: Sequence[str] = params.basis_gates,
                 swap_gates_pushed: bool = True,
                 swap_direcs_round1: List[List[str]] = params.swap_direcs_round1,
                 swap_direcs_round2: List[List[str]] = params.swap_direcs_round2
                ) -> None:
        """Initializes a CircuitTranspiler object.
        
        Args:
            basis_gates: A sequence of strings representing the basis gates used in transpilation.
            swap_gates_pushed: Whether SWAP gates are pushed.
            swap_direcs_round1: Strings indicating to which direction each SWAP gate is pushed 
                in the first round.
            swap_direcs_round2: Strings indicating to which direction each SWAP gate is pushed 
                in the second round.
        """
        self.basis_gates = basis_gates
        self.swap_gates_pushed = swap_gates_pushed
        self.swap_direcs_round1 = swap_direcs_round1
        self.swap_direcs_round2 = swap_direcs_round2

    def transpile(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpiles a circuit using Qiskit's built-in transpile function."""
        import qiskit
        circ_new = qiskit.transpile(circ, basis_gates=self.basis_gates)
        return circ_new

    def transpile_across_barriers(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpiles a circuit across barriers."""
        inst_tups_all = split_circuit_across_barriers(circ)

        def remove_first_swap_gate(circ_):
            """Removes the first SWAP gate in a circuit."""
            for i, inst_tup in enumerate(circ_.data):
                if inst_tup[0].name == 'swap':
                    del circ_.data[i]
                    break

        def swap_cp_and_u3(circ_):
            """Swaps the positions of CPhase and U3 gates."""
            tmp = circ_.data[6]
            circ_.data[6] = circ_.data[5]
            circ_.data[5] = circ_.data[4]
            circ_.data[4] = tmp

        # Transpile except for three-qubit gates
        qreg = circ.qregs[0]
        circ_new = QuantumCircuit(qreg)
        count = 0
        for i, inst_tups_single in enumerate(inst_tups_all):
            if len(inst_tups_single) > 1: # not 3-qubit gate
                # Create a single circuit between the barriers
                circ_single = create_circuit_from_inst_tups(inst_tups_single, qreg=qreg)
                circ_single = self.transpile(circ_single)

                # Swap positions of CPhase and U3 in the 5th sub-circuit
                if i == 4: swap_cp_and_u3(circ_single)
                
                # Push SWAP gates
                if self.swap_gates_pushed:
                    # First round pushes do not push through two-qubit gates
                    circ_single = self.push_swap_gates(
                        circ_single, 
                        direcs=self.swap_direcs_round1[count].copy(),
                        qreg=qreg)
                    circ_single = self.combine_swap_gates(circ_single)

                    # Second-round pushes push through two-qubit gates
                    circ_single = self.push_swap_gates(
                        circ_single, 
                        direcs=self.swap_direcs_round2[count].copy(),
                        qreg=qreg, 
                        push_through_2q=True)
                    # Final transpilation
                    circ_single = self.transpile(circ_single)

                    # Remove SWAP gates at the beginning
                    if i == 0: remove_first_swap_gate(circ_single)
                
                circ_new += circ_single
                count += 1

            else: # 3-qubit gate
                circ_new.barrier()
                circ_new.append(*inst_tups_single[0])
                circ_new.barrier()

        return circ_new

    @staticmethod
    def push_swap_gates(circ: QuantumCircuit, 
                        direcs: List[str] = [],
                        qreg: Optional[QuantumRegister] = None,
                        push_through_2q: bool = False) -> QuantumCircuit:
        """Pushes the swap gates across single- and two-qubit gates.
        
        Args:
            circ: The quantum circuit on which SWAP gates are pushed.
            direcs: The directions to which each SWAP gate is pushed.
            qreg: The quantum register of the circuit.
            push_through_2q: Whether to push through two-qubit gates.

        Returns:
            A new circuit on which SWAP gates are pushed.
        """
        assert set(direcs).issubset({'left', 'right', None})
        if direcs == []: return circ
        if qreg is None: qreg = circ.qregs[0]
        n_qubits = len(qreg)
        
        # Prepend and append barriers for easy processing
        inst_tups = circ.data.copy()
        inst_tups = [(Barrier(n_qubits), qreg, [])] + inst_tups
        inst_tups = inst_tups + [(Barrier(n_qubits), qreg, [])]
        n_inst_tups = len(inst_tups)

        # Store positions of SWAP gates that need to be pushed
        swap_gate_pos = []
        count = 0
        for i, inst_tup in enumerate(inst_tups):
            if inst_tup[0].name == 'swap' and direcs[count] is not None:
                swap_gate_pos.append(i)
                count += 1
        direcs = [d for d in direcs if d is not None]

        # Reorder SWAP gate positions due to pushing rightmost gates when pushing to the right
        for i, direc in enumerate(direcs):
            if direc == 'right':
                swap_gate_pos[i] *= -1
        sort_inds = np.argsort(swap_gate_pos)
        swap_gate_pos = [abs(swap_gate_pos[i]) for i in sort_inds]
        direcs = [direcs[i] for i in sort_inds]

        for i in swap_gate_pos:
            inst_tup = inst_tups[i]
            qargs = inst_tup[1]
            direc = direcs.pop(0)
            delete_pos = i + int(direc == 'left') # if direc == 'left', delete at i + 1

            # Push the SWAP gate to the left or the right. Insertion done before removal
            j = i + (-1) ** (direc == 'left')
            while j != 0 or j != n_inst_tups - 1:
                insert_pos = j + int(direc == 'left') # if direc == left, insert at right side
                inst_, qargs_, cargs_ = inst_tups[j]
                if inst_.name == 'barrier': # barrier
                    inst_tups.insert(insert_pos, inst_tup)
                    break
                else: # gate
                    if len(qargs_) == 1: # 1q gate
                        # Swap the indices and continue
                        if qargs_ == [qargs[0]]:
                            inst_tups[j] = (inst_, [qargs[1]], cargs_)
                        elif qargs_ == [qargs[1]]:
                            inst_tups[j] = (inst_, [qargs[0]], cargs_)
                    elif len(qargs_) == 2: # 2q gate but not SWAP gate
                        common_qargs = set(qargs).intersection(set(qargs_))
                        if len(common_qargs) > 0:
                            if push_through_2q and len(common_qargs) == 2:
                                # Swap the two-qubit gate and continue
                                inst_tups[j]  = (inst_, [qargs_[1], qargs_[0]], cargs_)
                            else:
                                # Insert here and exit the loop
                                inst_tups.insert(insert_pos, inst_tup)
                                break
                    else: # multi-qubit gate
                        # Insert here and exit the loop
                        inst_tups.insert(insert_pos, inst_tup)
                        break
                j += (-1) ** (direc == 'left') # j += 1 for right, j += -1 for left
            del inst_tups[delete_pos]

        # Remove the barriers and create new circuit
        inst_tups = inst_tups[1:-1]
        circ_new = create_circuit_from_inst_tups(inst_tups, qreg=qreg)
        return circ_new

    @staticmethod
    def combine_swap_gates(circ: QuantumCircuit) -> QuantumCircuit:
        """Combines adjacent SWAP gates."""
        inst_tups = circ.data.copy()
        n_inst_tups = len(inst_tups)
        inst_tup_pos = list(range(n_inst_tups))

        for i in range(n_inst_tups - 1):
            inst0, qargs0, _ = inst_tups[i]
            inst1, qargs1, _ = inst_tups[i + 1]
            if inst0.name == 'swap' and inst1.name == 'swap':
                common_qargs = set(qargs0).intersection(set(qargs1))
                if len(common_qargs) == 2:
                    inst_tup_pos.remove(i)
                    inst_tup_pos.remove(i + 1)

        inst_tups_new = [inst_tups[i] for i in inst_tup_pos]
        qreg = circ.qregs[0]
        circ_new = create_circuit_from_inst_tups(inst_tups_new, qreg=qreg)
        return circ_new

    def transpile_last_section(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpiles the last section of the circuit."""
        inst_tups = []
        while True:
            inst_tup = circ.data.pop()
            if inst_tup[0].name != 'barrier':
                inst_tups.insert(0, inst_tup)
            else:
                break

        qreg = circ.qregs[0]
        circ_last = create_circuit_from_inst_tups(inst_tups, qreg=qreg)
        circ_last = self.push_swap_gates(circ_last, direcs=['right'])
        # circ_last = transpile(circ_last, basis_gates=['u3', 'swap'])
        circ_last = self.transpile(circ_last)
        circ.barrier()
        circ += circ_last
        return circ

    def _apply_controlled_gate(self,
                              circ: QuantumCircuit,
                              op: SparsePauliOp,
                              ctrl: Sequence[int] = [1],
                              n_anc: int = 1) -> None:
        """Applies a controlled-U gate to a quantum circuit.

        Args:
            circ: The quantum circuit on which the controlled-U gate is applied.
            op: The operator from which the controlled gate is constructed.
            ctrl: The qubit state on which the cU gate is controlled on.
            n_anc: Number of ancilla qubits.

        Raises:
            NotImplementedError: Control on more than two qubits is not implemented.
        """
        # Sanity check
        assert set(ctrl).issubset({0, 1})
        assert len(ctrl) <= 2
        assert len(op.coeffs) == 1
        coeff = op.coeffs[0]
        label = op.table.to_labels()[0]
        amp, angle = polar(coeff)
        assert amp == 1

        ind_max = len(label) - 1
        label_tmp = label
        for i in range(len(label)):
            if label_tmp[0] == 'I':
                label_tmp = label_tmp[1:]
                ind_max -= 1

        ind_min = 0
        label_tmp = label
        for i in range(len(label)):
            if label_tmp[-1] == 'I':
                label_tmp = label_tmp[:-1]
                ind_min += 1

        # Prepend X gates for control on 0
        for i in range(len(ctrl)):
            if ctrl[i] == 0:
                circ.x(i)

        # Prepend rotation gates for Pauli X and Y
        for i, c in enumerate(label[::-1]):
            if c == 'X':
                circ.h(i + n_anc)
            elif c == 'Y':
                circ.rx(np.pi / 2, i + n_anc)

        # Prepend CNOT gates for Pauli strings
        for i in range(ind_max + n_anc, ind_min + n_anc, -1):
            circ.cx(i, i - 1)

        # Prepend SWAP gates when ind_min != 0
        if ind_min != 0:
            circ.swap(n_anc, ind_min + n_anc)
        
        # Apply the controlled gate
        if len(ctrl) == 1: # single-controlled gate
            if coeff != 1:
                circ.p(angle, 0)
            if set(list(label)) == {'I'}:
                pass
            else:
                circ.cz(0, n_anc)
        elif len(ctrl) == 2: # double-controlled gate
            if coeff != 1:
                circ.cp(angle, 0, 1)
            if set(list(label)) == {'I'}:
                if self.ccx_angle != 0:
                    circ.cp(self.ccx_angle, 0, 1)
            else:
                circ.h(n_anc)
                # Apply the central CCX gate
                if self.ccx_inst_tups is not None:
                    for inst_tup in self.ccx_inst_tups:
                        circ.append(*inst_tup)
                else:
                    circ.ccx(0, n_anc, 1)
                circ.h(n_anc)
        else:
            raise NotImplementedError("Control on more than two qubits is not implemented")

        # Append SWAP gate when ind_min != 0
        if ind_min != 0:
            circ.swap(n_anc, ind_min + n_anc)

        # Append CNOT gates for Pauli strings
        for i in range(ind_min + n_anc, ind_max + n_anc):
            circ.cx(i + 1, i)

        # Append rotation gates for Pauli X and Y
        for i, c in enumerate(label[::-1]):
            if c == 'X':
                circ.h(i + n_anc)
            elif c == 'Y':
                circ.rx(-np.pi / 2, i + n_anc)

        # Append X gates for control on 0
        for i in range(len(ctrl)):
            if ctrl[i] == 0:
                circ.x(i)

    def build_eh_off_diagonal(self,
                              a_op_m: SparsePauliOp,
                              a_op_n: SparsePauliOp
                              ) -> QuantumCircuit:
        """Constructs the circuit to calculate off-diagonal transition amplitudes.

        Args:
            a_op_m: The first creation/annihilation operator of the circuit.
            a_op_n: The second creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the two creation/annihilation operators added.
        """
        # Copy the circuit with empty ancilla positions
        circ = self._copy_circuit_with_ancilla([0, 1])

        # Apply the gates corresponding to the creation/annihilation terms
        # if self.add_barriers: circ.barrier()
        circ.h([0, 1])
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_m[0], ctrl=[0, 0], n_anc=2)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_m[1], ctrl=[1, 0], n_anc=2)
        if self.add_barriers: circ.barrier()
        circ.rz(np.pi / 4, 1)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_n[0], ctrl=[0, 1], n_anc=2)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op_n[1], ctrl=[1, 1], n_anc=2)
        if self.add_barriers: circ.barrier()
        circ.h([0, 1])

        return circ

    def build_eh_diagonal(self, a_op: SparsePauliOp) -> QuantumCircuit:
        """Constructs the circuit to calculate a diagonal transition amplitude.

        Args:
            The creation/annihilation operator of the circuit.

        Returns:
            The new circuit with the creation/annihilation operator appended.
        """
        # Copy the circuit with empty ancilla positions
        circ = self._copy_circuit_with_ancilla([0])

        # Apply the gates corresponding to the creation/annihilation terms
        if self.add_barriers: circ.barrier()
        circ.h(0)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op[0], ctrl=[0], n_anc=1)
        if self.add_barriers: circ.barrier()
        self._apply_controlled_gate(circ, a_op[1], ctrl=[1], n_anc=1)
        if self.add_barriers: circ.barrier()
        circ.h(0)
        if self.add_barriers: circ.barrier()

        return circ


    def _copy_circuit_with_ancilla(self, anc: Sequence[int]) -> QuantumCircuit:
        """Copies a circuit with specific indices for ancillas.

        Args:
            Indices of the ancilla qubits.

        Returns:
            The new quantum circuit with empty ancilla positions.
        """
        # Create a new circuit along with the quantum registers
        n_sys = self.ansatz.num_qubits
        n_anc = len(anc)
        n_qubits = n_sys + n_anc
        inds_new = [i for i in range(n_qubits) if i not in anc]
        qreg_new = QuantumRegister(n_qubits, name='q')
        circ_new = QuantumCircuit(qreg_new)

        # Copy instructions from the ansatz circuit to the new circuit
        for inst, qargs, cargs in self.ansatz.data:
            qargs = [inds_new[q._index] for q in qargs]
            circ_new.append(inst, qargs, cargs)
        return circ_new

def replace_swap_with_iswap(circ: QuantumCircuit, qubit_pairs: Sequence[Tuple[int, int]] = None) -> QuantumCircuit:
    inst_tups = circ.data.copy()
    inst_tups_new = []

    for inst, qargs, cargs in inst_tups:
        if inst.name == 'swap':
            q_pair = [x._index for x in qargs]
            q_pair_in = True
            if qubit_pairs is not None: q_pair_in = q_pair in qubit_pairs
            if q_pair_in:
                inst_tups_new.append((iSwapGate(), qargs, cargs))
                # inst_tups_new = [(RXGate(np.pi/2), [qargs[0]], []), 
                #                  (RXGate(np.pi/2), [qargs[1]], []), 
                #                  (CZGate(), [qargs[0], qargs[1]], [])] * 2 + \
                #                 [(RXGate(np.pi/2), [qargs[0]], []),
                #                  (RXGate(np.pi/2), [qargs[1]], []),
                #                  (RZGate(np.pi/2), [qargs[0]], []),
                #                  (RZGate(np.pi/2), [qargs[1]], [])] + \
                #                 inst_tups_new
        else:
            inst_tups_new.append((inst, qargs, cargs))
    
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
   #  assert circuit_equal(circ, circ_new)
    return circ_new 

def replace_with_cixc(circ: QuantumCircuit) -> QuantumCircuit:
    inst_tups = []
    for inst, qargs, cargs in circ.data.copy():
        if inst.name == 'ccx_o0':
            # inst_tups += [(CCXGate(ctrl_state='00'), qargs, [])]
            inst_tups += [(UnitaryGate(np.array([[0, 1j], [1j, 0]])).control(2, ctrl_state='00'), qargs, [])]
        else:
            inst_tups += [(inst, qargs, cargs)]

    circ_new = create_circuit_from_inst_tups(inst_tups)
    return circ_new

def special_transpilation(circ: QuantumCircuit) -> QuantumCircuit:
    """A special transpilation function for the 01d circuit. Should be deprecated."""
    count_x = 0
    count_cz = 0

    process_counts_x = [1, 3]
    del_counts_cz = [2, 3, 8, 9]

    insert_inds_z = []
    del_inds_cz = []

    inst_tups = circ.data.copy()
    for i, (inst, qargs, cargs) in enumerate(inst_tups):
        if inst.name == 'x' and qargs[0]._index == 1:
            if count_x in process_counts_x:
                insert_inds_z.append(i)
            count_x += 1
        elif inst.name == 'cz' and [q._index for q in qargs] == [0, 1]:
            if count_cz in del_counts_cz:
                del_inds_cz.append(i)
            count_cz += 1

    inst_tups_new = [inst_tups[i] for i in range(len(inst_tups)) if i not in del_inds_cz]
    for i in insert_inds_z:
        inst_tups_new.insert(i, (RZGate(np.pi), [0], []))
    circ_new = create_circuit_from_inst_tups(inst_tups_new)
    fig = circ_new.draw('mpl')
    fig.savefig(f'figs/circ01d_stage3.png', bbox_inches='tight')
    assert circuit_equal(circ, circ_new)
    return circ_new

