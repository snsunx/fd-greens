                """
                qreg = circ.qregs[0]
                probs_e = np.zeros((self.n_e,)) # XXX
                probs_h = np.zeros((self.n_h,)) # XXX

                for lam in range(self.n_e): # XXX
                    circ_ = circ.copy()
                    circ_.append(cVs_e[lam], qreg)

                    circ_real = circ_.copy()
                    circ_real.h(0)
                    circ_real.measure(0) # XXX
                    result = self.q_instance.execute(circ_real)
                    counts = result.get_counts()
                    real_part = evaluate_counts(counts)

                    circ_imag = circ_.copy()
                    circ_imag.h(0)
                    circ_imag.measure(0) # XXX
                    result = self.q_instance.execute(circ_real)
                    counts = result.get_counts()
                    imag_part = evaluate_counts(counts)

                    prob = real_part ** 2 + imag_part ** 2
                    probs_e[lam] = prob

                for lam in range(self.n_h):
                    circ_ = circ.copy()
                    circ.append(cVs_h[lam], qreg)

                    circ_real = circ_.copy()
                    circ_real.h(0)
                    circ_real.measure(0) # XXX
                    result = self.q_instance.execute(circ_real)
                    counts = result.get_counts()
                    real_part = evaluate_counts(counts)

                    circ_imag = circ_.copy()
                    circ_imag.h(0)
                    circ_imag.measure(0) # XXX
                    result = self.q_instance.execute(circ_real)
                    counts = result.get_counts()
                    imag_part = evaluate_counts(counts)

                    prob = real_part ** 2 + imag_part ** 2
                    probs_e[lam] = prob
                """