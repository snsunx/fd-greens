"""
======================================================
Response Function (:mod:`fd_greens.response_function`)
======================================================
"""

import os
from typing import Sequence

import h5py
import numpy as np

from fd_greens.cirq_ver.molecular_hamiltonian import MolecularHamiltonian


class ResponseFunction:
    """A class to calculate frequency-domain charge-charge response function."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        h5fname: str = 'lih',
        suffix: str = '',
        method: str = 'exact') -> None:
        """Initializes a ResponseFunction object.
        
        Args:
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        
        self.datfname = h5fname + suffix
        h5file = h5py.File(h5fname + ".h5", "r")
        self.energy_gs = h5file["gs/energy"]
        self.energies_s = h5file["es/energies_s"]
        # self.energies_t = h5file['es/energies_t']
        self.N = h5file[f"amp/N{suffix}"]

    def _initialize(self) -> None:
        """Initializes physical quantity attributes."""
        h5file = h5py.File(self.h5fname, "r+")

        # Occupied and active orbital indices.
        self.occ_inds = self.h.occ_inds
        self.act_inds = self.h.act_inds

        self.qinds_sys = transform_4q_indices(singlet_inds)

        # self.keys_diag = ['n', 'n_']
        # self.qinds_anc_diag = [[1], [0]]
        self.keys_diag = ["n"]
        self.qinds_anc_diag = [[1]]
        self.qinds_tot_diag = dict()
        for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
            self.qinds_tot_diag[key] = self.qinds_sys.insert_ancilla(qind)

        # self.keys_off_diag = ['np', 'nm', 'n_p', 'n_m']
        # self.qinds_anc_off_diag = [[1, 0], [1, 1], [0, 0], [0, 1]]
        self.keys_off_diag = ["np", "nm"]
        self.qinds_anc_off_diag = [[1, 0], [1, 1]]
        self.qinds_tot_off_diag = dict()
        for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
            self.qinds_tot_off_diag[key] = self.qinds_sys.insert_ancilla(qind)

        # Number of spatial orbitals and (N+/-1)-electron states.
        self.n_elec = self.h.molecule.n_electrons
        self.ansatz = QuantumCircuit.from_qasm_str(h5file["gs/ansatz"][()].decode())
        self.energies = h5file["es/energies_s"]
        self.states = h5file["es/states_s"][:]
        self.n_states = len(self.energies)
        self.n_orb = 2  # XXX: Hardcoded
        # self.n_occ = self.n_elec // 2 - len(self.occ_inds)
        # self.n_vir = self.n_orb - self.n_occ

        # Transition amplitudes arrays. Keys of N are 'n' and 'n_'.
        # Keys of T are 'np', 'nm', 'n_p', 'n_m'.
        self.N = defaultdict(
            lambda: np.zeros(
                (2 * self.n_orb, 2 * self.n_orb, self.n_states), dtype=complex
            )
        )
        self.T = defaultdict(
            lambda: np.zeros(
                (2 * self.n_orb, 2 * self.n_orb, self.n_states), dtype=complex
            )
        )

        # Create Pauli dictionaries for operators after symmetry transformation.
        charge_ops = ChargeOperators(self.n_elec)
        charge_ops.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.pauli_dict = charge_ops.get_pauli_dict()

        h5file.close()

    def process_diagonal(self) -> None:
        """Post-processes diagonal transition amplitudes circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        # for m in range(self.n_orb): # 0, 1
        #     for s in ['u', 'd']:
        for _ in [1]:
            for i in range(4):  # 4 is hardcoded
                # ms = 2 * m + (s == 'd')
                m, s = self.orb_labels[i]
                if self.method == "exact":
                    psi = h5file[f"circ{m}{s}/transpiled"].attrs[f"psi{self.suffix}"]
                    for key in self.keys_diag:
                        psi_key = self.qinds_tot_diag[key](psi)
                        self.N[key][i, i] = np.abs(self.states.conj().T @ psi_key)
                        print(f"N[{key}][{i}, {i}] = {self.N[key][i, i]}")
                else:  # Tomography
                    for key, qind in zip(self.keys_diag, self.qinds_anc_diag):
                        # Stack counts_arr over all tomography labels together.
                        counts_arr_key = np.array([])
                        for label in self.tomo_labels:
                            counts_arr = h5file[f"circ{m}{s}/{label}"].attrs[
                                f"counts{self.suffix}"
                            ]
                            start = int("".join([str(k) for k in qind])[::-1], 2)
                            counts_arr_label = counts_arr[
                                start::2
                            ]  # 2 is because 2 ** 1
                            counts_arr_label = counts_arr_label / np.sum(counts_arr)
                            counts_arr_key = np.hstack(
                                (counts_arr_key, counts_arr_label)
                            )

                        # Obtain the density matrix from tomography.
                        rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0]
                        rho = rho.reshape(4, 4, order="F")
                        rho = self.qinds_sys(rho)

                        self.N[key][i, i] = [
                            get_overlap(self.states[:, k], rho) for k in range(4)
                        ]
                        # XXX: 4 is hardcoded
                        print(f"N[{key}][{i}, {i}] = {self.N[key][i, i]}")

        h5file.close()

    def process_off_diagonal(self) -> None:
        """Post-processes off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")

        # for m in [0, 1]:
        #     for s, s_ in [('u', 'd'), ('d', 'u')]:
        #         ms = 2 * m + (s == 'd')
        #         ms_ = 2 * m + (s_ == 'd')
        for i in range(4):
            m, s = self.orb_labels[i]
            for j in range(i + 1, 4):
                m_, s_ = self.orb_labels[j]
                if self.method == "exact":
                    psi = h5file[f"circ{m}{s}{m_}{s_}/transpiled"].attrs[
                        f"psi{self.suffix}"
                    ]
                    for key in self.keys_off_diag:
                        psi_key = self.qinds_tot_off_diag[key](psi)
                        self.T[key][i, j] = np.abs(self.states.conj().T @ psi_key)
                        print(f"T[{key}][{i}, {j}] = {self.T[key][i, j]}")
                else:  # Tomography
                    for key, qind in zip(self.keys_off_diag, self.qinds_anc_off_diag):
                        counts_arr_key = np.array([])
                        for label in self.tomo_labels:
                            counts_arr = h5file[f"circ{m}{s}{m_}{s_}/{label}"].attrs[
                                f"counts{self.suffix}"
                            ]
                            start = int("".join([str(k) for k in qind])[::-1], 2)
                            counts_arr_label = counts_arr[
                                start::4
                            ]  # 4 is because 2 ** 2
                            counts_arr_label = counts_arr_label / np.sum(counts_arr)
                            counts_arr_key = np.hstack(
                                [counts_arr_key, counts_arr_label]
                            )

                        rho = np.linalg.lstsq(basis_matrix, counts_arr_key)[0]
                        rho = rho.reshape(4, 4, order="F")
                        rho = self.qinds_sys(rho)
                        self.T[key][i, j] = [
                            get_overlap(self.states[:, k], rho) for k in range(4)
                        ]  # 4 is hardcoded
                        print(f"T[{key}][{i}, {j}] = {self.T[key][i, j]}")

        # Unpack T values to N values based on Eq. (18) of Kosugi and Matsushita 2021.
        for key in self.keys_diag:
            # for ms, ms_ in [(0, 1), (2, 3)]:
            for i in range(4):
                for j in range(i + 1, 4):
                    self.N[key][i, j] = self.N[key][j, i] = np.exp(-1j * np.pi / 4) * (
                        self.T[key + "p"][i, j] - self.T[key + "m"][i, j]
                    ) + np.exp(1j * np.pi / 4) * (
                        self.T[key + "p"][j, i] - self.T[key + "m"][j, i]
                    )
                    print(f"N[{key}][{i}, {j}] =", self.N[key][i, j])
                # self.N[key][2, 3] = self.N[key][2, 3] = \
                #     np.exp(-1j * np.pi/4) * (self.T[key+'p'][2, 3] - self.T[key+'m'][2, 3]) \
                #     + np.exp(1j * np.pi/4) * (self.T[key+'p'][3, 2] - self.T[key+'m'][3, 2])
                # print(f'N[{key}][2, 3] =', self.N[key][2, 3])

            # write_hdf5(h5file, 'amp', f'N{self.suffix}', self.N[key])

        h5file.close()


    def response_function(
        self, omegas: Sequence[float], eta: float = 0.0, save_data: bool = True
    ) -> np.ndarray:
        """Returns the charge-charge response function at given frequencies.

        Args:
            omegas: The frequencies at which the response function is calculated.
            eta: The imaginary part, i.e. broadening factor.
            save: Whether to save the response function to file.

        Returns:
            (Optional) The charge-charge response function for orbital i and orbital j.
        """
        for label in ["00", "11", "01", "10"]:
            i = int(label[0])
            j = int(label[1])
            chis = []
            for omega in omegas:
                for lam in [1, 2, 3]:  # [1, 2, 3] is hardcoded
                    chi = np.sum(
                        self.N[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1), lam]
                    ) / (omega + 1j * eta - (self.energies_s[lam] - self.energy_gs))
                    chi += np.sum(
                        self.N[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1), lam]
                    ).conjugate() / (
                        -omega - 1j * eta - (self.energies_s[lam] - self.energy_gs)
                    )
                chis.append(chi)
            chis = np.array(chis)
            if save_data:
                if not os.path.exists("data"):
                    os.makedirs("data")
                np.savetxt(
                    f"data/{self.datfname}_chi{label}.dat",
                    np.vstack((omegas, chis.real, chis.imag)).T,
                )
            else:
                return chis
