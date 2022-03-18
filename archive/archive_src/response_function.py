    def cross_section(self, 
                      omegas: Sequence[float], 
                      eta: float = 0.0, 
                      save: bool = True) -> complex:
        """Returns the photo-absorption cross section.
        
        Args:
            omegas: The frequencies at which the response function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the photo-absorption cross section to file.

        Returns:
            (Optional) The photo-absorption cross section.
        """
        sigmas = []
        for omega in omegas:
            alpha = 0
            for i in range(2): # XXX: Hardcoded.
                alpha += -self.response_function([omega + 1j * eta], i, i, save=False)

            sigma = 4 * np.pi / params.c * omega * alpha.imag
            sigmas.append(sigma)
        sigmas = np.array(sigmas)

        if save:
            np.savetxt('data/' + self.fname + '_sigma.dat', np.vstack((omegas, sigmas.real, sigmas.imag)).T)
        else:
            return sigma