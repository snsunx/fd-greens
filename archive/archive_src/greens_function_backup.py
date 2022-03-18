                elif recompiled == 1:
                    Umat_rev = reverse_qubit_order(Umat)
                    cUmat = np.kron(Umat_rev, np.diag([0, 1])) + np.kron(np.eye(2 ** 4), np.diag([1, 0]))
                    cUmat = np.kron(cUmat, np.eye(2))
                    '''
                    cUmat0 = np.kron(np.eye(2), 
                        np.kron(np.diag([1, 0]), np.eye(2 ** 4))
                         + np.kron(np.diag([0, 1]), Umat))
                    cUmat0_rev = reverse_qubit_order(cUmat0)
                    print(np.allclose(cUmat, cUmat0_rev))
                    '''

                    circ.barrier()
                    circ.h(1)
                    statevector = get_statevector(circ)
                    quimb_gates = recompile_with_statevector(
                        statevector, cUmat, n_gate_rounds=6)
                    circ = apply_quimb_gates(quimb_gates, circ.copy(), reverse=True)
                    circ.h(1)
                    circ.barrier()

                    circ_recompiled1 = get_statevector(circ)
                    np.save('circ_recompiled1.npy', circ_recompiled1)