"""Make experimental density matrix physical/pure"""

import numpy as np
from scipy.stats import unitary_group

def generate_density_matrix(n_qubits):
    dim = 2 ** n_qubits

    eigvals = np.random.rand(dim) - 0.5
    eigvals = np.sort(eigvals)
    eigvals /= np.sum(eigvals)
    # print(f"{eigvals = }")

    eigvecs = unitary_group.rvs(dim)
    density_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    # print(density_matrix)
    return density_matrix

def project_and_normalize_density_matrix(rho_uncons):
    """Take a density matrix that is possibly not positive semi-definite, and also not trace one, and 
    return the closest positive semi-definite density matrix with trace-1 using the algorithm in
    PhysRevLett.108.070502. Note this method assumes additive Gaussian noise
    """

    # make the density matrix trace one
    rho_uncons = rho_uncons / np.trace(rho_uncons)

    d = rho_uncons.shape[0]  # the dimension of the Hilbert space
    [eigvals_un, eigvecs_un] = np.linalg.eigh(rho_uncons)

    # If matrix is already trace one Positive Semi-Definite, we are done
    if np.min(eigvals_un) >= 0:
        # print 'Already PSD'
        return rho_uncons
    # Otherwise, continue finding closest trace one,
    # Positive Semi-Definite matrix through eigenvalue modification
    eigvals_un = list(eigvals_un)
    eigvals_un.reverse()
    eigvals_new = [0.0] * len(eigvals_un)
    i = d
    a = 0.0  # Accumulator
    while eigvals_un[i - 1] + a / float(i) < 0:
        a += eigvals_un[i - 1]
        i -= 1
    for j in range(i):
        eigvals_new[j] = eigvals_un[j] + a / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_cons = np.dot(eigvecs_un, np.dot(np.diag(eigvals_new), np.conj(eigvecs_un.T)))

    return rho_cons

def mcweeny_purification(rho, n=10):
    """Calculate the closest idempotent matrix from rho with n steps.

    Args:
        rho (numpy.array()): density matrix
        n (int): number of steps

    Returns:
        rho_n (numpy.array()): idempotent density matrix
    """
    N = len(rho)
    rho_n = np.copy(rho)
    for k in range(n):
        rho_n = rho_n @ rho_n @ (3 * np.eye(N) - 2 * rho_n)
    return rho_n

if __name__ == '__main__':
    rho = generate_density_matrix(2)
    rho1 = project_and_normalize_density_matrix(rho)
    rho2 = project_density_matrix(rho)

    print(np.allclose(rho1, rho2))
