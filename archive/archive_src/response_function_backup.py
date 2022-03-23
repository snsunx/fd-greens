        '''
        for omega in omegas:
            for lam in [1, 2, 3]: # XXX: Hardcoded
                chi = np.sum(self.N[:2,:2,lam]) / (omega + 1j*eta - (self.energies_s[lam] - self.energy_gs))
                chi += np.sum(self.N[:2,:2,lam]).conjugate() / (-omega - 1j*eta - (self.energies_s[lam] - self.energy_gs))
            chi0s.append(chi)

            for lam in [1, 2, 3]: # XXX: Hardcoded
                chi = np.sum(self.N[2:,2:,lam]) / (omega + 1j*eta - (self.energies_s[lam] - self.energy_gs))
                chi += np.sum(self.N[2:,2:,lam]).conjugate() / (-omega - 1j*eta - (self.energies_s[lam] - self.energy_gs))
            chi1s.append(chi)

            for lam in [1, 2, 3]: # XXX: Hardcoded
                chi = np.sum(self.N[:2,2:,lam]) / (omega + 1j*eta - (self.energies_s[lam] - self.energy_gs))
                chi += np.sum(self.N[:2,2:,lam]).conjugate() / (-omega - 1j*eta - (self.energies_s[lam] - self.energy_gs))
            chi01s.append(chi)

            for lam in [1, 2, 3]: # XXX: Hardcoded
                chi = np.sum(self.N[2:,:2,lam]) / (omega + 1j*eta - (self.energies_s[lam] - self.energy_gs))
                chi += np.sum(self.N[2:,:2,lam]).conjugate() / (-omega - 1j*eta - (self.energies_s[lam] - self.energy_gs))
            chi10s.append(chi)

        chi0s = np.array(chi0s)
        chi1s = np.array(chi1s)
        chi01s = np.array(chi01s)
        chi10s = np.array(chi10s)

        if save:
            np.savetxt('data/' + self.datfname + '_chi0.dat', np.vstack((omegas, chi0s.real, chi0s.imag)).T)
            np.savetxt('data/' + self.datfname + '_chi1.dat', np.vstack((omegas, chi1s.real, chi1s.imag)).T)
            np.savetxt('data/' + self.datfname + '_chi01.dat', np.vstack((omegas, chi01s.real, chi01s.imag)).T)
            np.savetxt('data/' + self.datfname + '_chi10.dat', np.vstack((omegas, chi10s.real, chi10s.imag)).T)
        else:
            return chi0s, chi1s, chi01s, chi10s
        '''
