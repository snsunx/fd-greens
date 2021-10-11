'''
class SingleQubitGateChain:
    def __init__(self, *, qubit, gates) -> None:
        self.qubit = qubit
        self.gates = gates
        self.n_gates = len(gates)

    @property
    def combined_gate(self):
        """Returns the U3 gate from combining a chain of single-qubit gates."""
        matrix = reduce(np.dot, [gate.to_matrix() for gate in self.gates[::-1]])
        decomposer = OneQubitEulerDecomposer()
        gate = decomposer(matrix).data[0][0]
        return gate

    @classmethod
    def combine_single_qubit_gates(cls, circ: QuantumCircuit) -> QuantumCircuit:
        """Combines each single-qubit gate chain to a single U3 gate on a circuit.

        Args:
            The circuit on which all chains of single-qubit gates are to be combined.

        Returns:
            A new circuit after all single-qubit gates have been combined.
        """
        gate_1q_chains = []
        gates_2q = []

        insert_inds = []

        i = 0
        while len(circ.data) != 0:
            print('i =', i)
            inst, qargs, cargs = circ.data[i]
            subtract_1 = False
            print(circ)
            if len(qargs) == 1:
                insert_inds.append(i)
                delete_inds = [i]
                gates = [inst]
                qubit = qargs[0]
                for j, (inst, qargs, cargs) in enumerate(circ.data[i+1:]):
                    if len(qargs) == 1 and qargs[0] == qubit:
                        # Append to single-qubit gate chain
                        gates.append(inst)
                        delete_inds.append(j + i + 1)
                    elif len(qargs) > 1 and qubit in qargs:
                        # Stop appending
                        i += 1
                        break
                    elif len(qargs) == 1 and qargs[0] != qubit:
                        subtract_1 = True
                    else:
                        # Ignore
                        continue

                # Delete the gates
                count = 0
                for k in delete_inds:
                    del circ.data[k - count]
                    count += 1


                # Append single-qubit gate chain
                gate_chain = cls(qubit=qubit, gates=gates)
                gate_chains.append(gate_chain)


        # Adjust insertion indices
        insert_inds = [ind + k for k, ind in enumerate(insert_inds)]
        for i, gate_chain in enumerate(gate_chains):
            circ.data.insert(insert_inds[i], (gate_chain.combined_gate, [gate_chain.qubit], []))
        return circ
'''