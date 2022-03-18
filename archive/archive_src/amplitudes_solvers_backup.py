    def compute_diagonal_amplitudes(self) -> None:
        """Calculates diagonal transition amplitudes."""
        print("----- Calculating diagonal transition amplitudes -----")
        inds_anc_e = QubitIndices(['1'])
        inds_anc_h = QubitIndices(['0'])
        inds_tot_e = self.inds_e + inds_anc_e
        inds_tot_h = self.inds_h + inds_anc_h
        for m in range(self.n_orb):
            a_op_m = self.pauli_dict[(m, self.spin[1])]
            circ = self.circuit_constructor.build_diagonal_circuits(a_op_m)
            fname = f'circuits/circuit_{m}' + self.suffix
            if self.transpiled: circ = transpile(circ, basis_gates=['u3', 'swap', 'cz', 'cp'])
            if self.save: save_circuit(circ, fname)

            if self.method == 'exact' and self.backend.name() == 'statevector_simulator':
                result = self.q_instance.execute(circ)
                psi = result.get_statevector()

                psi_e = psi[inds_tot_e.int_form]
                B_e_mm = np.abs(self.states_e.conj().T @ psi_e) ** 2

                psi_h = psi[inds_tot_h.int_form]
                B_h_mm = np.abs(self.states_h.conj().T @ psi_h) ** 2

            elif self.method == 'energy':
                circ_anc = circ.copy()
                circ_anc.add_register(ClassicalRegister(1))
                circ_anc.measure(0, 0)
                
                result = self.q_instance.execute(circ_anc)
                counts = get_counts(result)
                shots = sum(counts.values())
                p_e = counts[inds_anc_e.str_form[0]] / shots
                p_h = counts[inds_anc_h.str_form[0]] / shots

                energy_e = measure_operator(circ.copy(), self.qiskit_op_spin, 
                                            q_instance=self.q_instance, 
                                            anc_state=inds_anc_e.list_form[0])
                energy_h = measure_operator(circ.copy(), self.qiskit_op_spin,
                                            q_instance=self.q_instance,
                                            anc_state=inds_anc_h.list_form[0])

                B_e_mm = p_e * solve_energy_probabilities(self.energies_e, energy_e)
                B_h_mm = p_h * solve_energy_probabilities(self.energies_h, energy_h)
                
            elif self.method == 'tomo':
                rho = state_tomography(circ, q_instance=self.q_instance)

                rho_e = rho[inds_tot_e.int_form][:, inds_tot_e.int_form]
                rho_h = rho[inds_tot_h.int_form][:, inds_tot_h.int_form]

                B_e_mm = np.zeros((self.n_e,), dtype=float)
                B_h_mm = np.zeros((self.n_h,), dtype=float)
                for i in range(self.n_e):
                    B_e_mm[i] = get_overlap(self.states_e[i], rho_e)
                for i in range(self.n_h):
                    B_h_mm[i] = get_overlap(self.states_h[i], rho_h)

            self.B_e[m, m] = B_e_mm
            self.B_h[m, m] = B_h_mm

            print(f'B_e[{m}, {m}] = {self.B_e[m, m]}')
            print(f'B_h[{m}, {m}] = {self.B_h[m, m]}')
        print("------------------------------------------------------")

    def compute_off_diagonal_amplitudes(self) -> None:
        """Calculates off-diagonal transition amplitudes."""
        print("----- Calculating off-diagonal transition amplitudes -----")

        inds_anc_ep = QubitIndices(['01'])
        inds_anc_em = QubitIndices(['11'])
        inds_anc_hp = QubitIndices(['00'])
        inds_anc_hm = QubitIndices(['10'])
        inds_tot_ep = self.inds_e + inds_anc_ep
        inds_tot_em = self.inds_e + inds_anc_em
        inds_tot_hp = self.inds_h + inds_anc_hp
        inds_tot_hm = self.inds_h + inds_anc_hm

        for m in range(self.n_orb):
            a_op_m = self.pauli_dict[(m, self.spin[1])]
            for n in range(self.n_orb):
                if m < n:
                    a_op_n = self.pauli_dict[(n, self.spin[1])]

                    circ = self.circuit_constructor.build_off_diagonal_circuits(a_op_m, a_op_n)
                    fname = f'circuits/circuit_{m}{n}' + self.suffix
                    if self.transpiled:
                        circ = transpile_across_barrier(
                            circ, basis_gates=['u3', 'swap', 'cz', 'cp'], 
                            push=self.push, ind=(m, n))
                    if self.save: save_circuit(circ, fname)

                    if self.method == 'exact' or self.backend.name() == 'statevector_simulator':
                        result = self.q_instance.execute(circ)
                        psi = result.get_statevector()

                        psi_ep = psi[inds_tot_ep.int_form]
                        psi_em = psi[inds_tot_em.int_form]
                        D_ep_mn = abs(self.states_e.conj().T @ psi_ep) ** 2
                        D_em_mn = abs(self.states_e.conj().T @ psi_em) ** 2

                        psi_hp = psi[inds_tot_hp.int_form]
                        psi_hm = psi[inds_tot_hm.int_form]
                        D_hp_mn = abs(self.states_h.conj().T @ psi_hp) ** 2
                        D_hm_mn = abs(self.states_h.conj().T @ psi_hm) ** 2

                    elif self.method == 'energy':
                        circ_anc = circ.copy()
                        circ_anc.add_register(ClassicalRegister(2))
                        circ_anc.measure([0, 1], [0, 1])
                        
                        result = self.q_instance.execute(circ_anc)
                        counts = get_counts(result)
                        shots = sum(counts.values())
                        p_ep = counts[inds_anc_ep.str_form[0]] / shots
                        p_em = counts[inds_anc_em.str_form[0]] / shots
                        p_hp = counts[inds_anc_hp.str_form[0]] / shots
                        p_hm = counts[inds_anc_hm.str_form[0]] / shots

                        energy_ep = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance,
                                                     anc_state=inds_anc_ep.list_form[0])
                        energy_em = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance,
                                                     anc_state=inds_anc_em.list_form[0])
                        energy_hp = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance,
                                                     anc_state=inds_anc_hp.list_form[0])
                        energy_hm = measure_operator(circ.copy(), self.qiskit_op_spin,
                                                     q_instance=self.q_instance, 
                                                     anc_state=inds_anc_hm.list_form[0])

                        D_ep_mn = p_ep * solve_energy_probabilities(self.energies_e, energy_ep)
                        D_em_mn = p_em * solve_energy_probabilities(self.energies_e, energy_em)
                        D_hp_mn = p_hp * solve_energy_probabilities(self.energies_h, energy_hp)
                        D_hm_mn = p_hm * solve_energy_probabilities(self.energies_h, energy_hm)
                                
                    elif self.method == 'tomo':
                        rho = state_tomography(circ, q_instance=self.q_instance)

                        rho_ep = rho[inds_tot_ep.int_form][:, inds_tot_ep.int_form]
                        rho_em = rho[inds_tot_em.int_form][:, inds_tot_em.int_form]
                        rho_hp = rho[inds_tot_hp.int_form][:, inds_tot_hp.int_form]
                        rho_hm = rho[inds_tot_hm.int_form][:, inds_tot_hm.int_form]

                        D_ep_mn = np.zeros((self.n_e,), dtype=float)
                        D_em_mn = np.zeros((self.n_e,), dtype=float)
                        D_hp_mn = np.zeros((self.n_h,), dtype=float)
                        D_hm_mn = np.zeros((self.n_h,), dtype=float)

                        for i in range(self.n_e):
                            D_ep_mn[i] = get_overlap(self.states_e[i], rho_ep)
                            D_em_mn[i] = get_overlap(self.states_e[i], rho_em)

                        for i in range(self.n_h):
                            D_hp_mn[i] = get_overlap(self.states_h[i], rho_hp)
                            D_hm_mn[i] = get_overlap(self.states_h[i], rho_hm)

                    self.D_ep[m, n] = self.D_ep[n, m] = D_ep_mn
                    self.D_em[m, n] = self.D_em[n, m] = D_em_mn
                    self.D_hp[m, n] = self.D_hp[n, m] = D_hp_mn
                    self.D_hm[m, n] = self.D_hm[n, m] = D_hm_mn

                    print(f'D_ep[{m}, {n}] = {self.D_ep[m, n]}')
                    print(f'D_em[{m}, {n}] = {self.D_em[m, n]}')
                    print(f'D_hp[{m}, {n}] = {self.D_hp[m, n]}')
                    print(f'D_hm[{m}, {n}] = {self.D_hm[m, n]}')

        # Unpack D values to B values
        for m in range(self.n_orb):
            for n in range(self.n_orb):
                if m < n:
                    B_e_mn = np.exp(-1j * np.pi / 4) * (self.D_ep[m, n] - self.D_em[m, n])
                    B_e_mn += np.exp(1j * np.pi / 4) * (self.D_ep[n, m] - self.D_em[n, m])
                    # B_e_mn[abs(B_e_mn) < 1e-8] = 0
                    self.B_e[m, n] = self.B_e[n, m] = B_e_mn

                    B_h_mn = np.exp(-1j * np.pi / 4) * (self.D_hp[m, n] - self.D_hm[m, n])
                    B_h_mn += np.exp(1j * np.pi / 4) * (self.D_hp[n, m] - self.D_hm[n, m])
                    # B_h_mn[abs(B_h_mn) < 1e-8] = 0
                    self.B_h[m, n] = self.B_h[n, m] = B_h_mn
                    
                    print(f'B_e[{m}, {n}] = {self.B_e[m, n]}')
                    print(f'B_h[{m}, {n}] = {self.B_h[m, n]}')

        print("----------------------------------------------------------")
