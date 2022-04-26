"""
=========================================================================================
N-Electron Transition Amplitudes Solver (:mod:`fd_greens.main.excited_amplitudes_solver`)
=========================================================================================
"""

import itertools
from typing import Iterable, Optional, Sequence
from functools import partial
from collections import defaultdict

import h5py
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance

from .params import singlet_inds
from .molecular_hamiltonian import MolecularHamiltonian
from .operators import ChargeOperators
from .qubit_indices import QubitIndices
from .circuit_constructor import CircuitConstructor, InstructionTuple
# from .transpilation_backup import transpile_into_berkeley_gates
from .z2symmetries import transform_4q_indices, transform_4q_pauli
from ..utils import (
    get_overlap,
    counts_dict_to_arr,
    write_hdf5,
    basis_matrix,
    append_tomography_gates,
    append_measurement_gates,
    convert_circuit_to_string,
    convert_string_to_circuit,
)


class ExcitedAmplitudesSolver:
    """A class to calculate transition amplitudes between the ground state and the excited states."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        q_instance: Optional[QuantumInstance] = None,
        method: str = "exact",
        h5fname: str = "lih",
        suffix: str = "",
    ) -> None:
        """Initializes an ExcitedAmplitudesSolver object.

        Args:
            h: The molecular Hamiltonian.
            q_instance: The quantum instance for executing the circuits.
            method: The method for extracting the transition amplitudes.
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        assert method in ["exact", "tomo"]

        # Basic variables.
        self.h = h
        self.q_instance = q_instance
        self.method = method
        self.suffix = suffix

        self.circ_str_type = "qasm"

        # Load data and initialize quantities.
        self.h5fname = h5fname + ".h5"
        self._initialize()

        self.orb_labels = [(0, "u"), (0, "d"), (1, "u"), (1, "d")]

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

        # The circuit constructor and tomography labels.
        self.constructor = CircuitConstructor(self.ansatz)
        self.tomo_labels = [
            "".join(x) for x in itertools.product("xyz", repeat=2)
        ]  # 2 is hardcoded

        h5file.close()

    def build_diagonal(self) -> None:
        """Calculates diagonal transition amplitudes."""
        h5file = h5py.File(self.h5fname, "r+")
        # for m in range(self.n_orb): # 0, 1
        #     for s in ['u', 'd']:
        for _ in [1]:
            for i in range(4):  # 4 is hardcoded
                m, s = self.orb_labels[i]

                # Build the diagonal circuit and save to HDF5 file.
                U_op = self.pauli_dict[(m, s)]
                circ = self.constructor.build_diagonal(U_op)
                # write_hdf5(h5file, f'circ{m}{s}', 'untranspiled', circuit_to_qasm_str(circ))

                # Transpile the circuit and save to HDF5 file.
                circ = transpile_into_berkeley_gates(circ, f"r{m}{s}")
                circ_str = convert_circuit_to_string(circ, self.circ_str_type)
                write_hdf5(h5file, f"circ{m}{s}", "transpiled", circ_str)

                if self.method == "tomo":
                    for label in self.tomo_labels:
                        # Append tomography and measurement gates and save to HDF5 file.
                        tomo_circ = append_tomography_gates(circ, [1, 2], label)
                        tomo_circ = append_measurement_gates(tomo_circ)
                        circ_str = convert_circuit_to_string(
                            tomo_circ, self.circ_str_type
                        )
                        write_hdf5(h5file, f"circ{m}{s}", label, circ_str)

        h5file.close()

    def execute_diagonal(self) -> None:
        """Executes the diagonal circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        # for m in range(self.n_orb): # 0, 1
        #     for s in ['u', 'd']:
        for _ in [1]:
            for i in range(4):  # 4 is hardcoded
                m, s = self.orb_labels[i]
                if self.method == "exact":
                    dset = h5file[f"circ{m}{s}/transpiled"]
                    circ_str = dset[()]
                    circ = convert_string_to_circuit(circ_str, self.circ_str_type)
                    # circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                    result = self.q_instance.execute(circ)
                    # print(result)
                    psi = result.get_statevector()
                    dset.attrs[f"psi{self.suffix}"] = psi

                else:  # Tomography
                    for label in self.tomo_labels:
                        dset = h5file[f"circ{m}{s}/{label}"]
                        circ_str = dset[()]
                        circ = convert_string_to_circuit(circ_str, self.circ_str_type)
                        # circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                        result = self.q_instance.execute(circ)
                        counts = result.get_counts()
                        counts_arr = counts_dict_to_arr(counts.int_raw, n_qubits=3)
                        dset.attrs[f"counts{self.suffix}"] = counts_arr
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

    def build_off_diagonal(self) -> None:
        """Constructs off-diagonal excited-state transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        # for m in range(self.n_orb): # 0, 1
        #     for s, s_ in [('u', 'd'), ('d', 'u')]:
        for i in range(4):
            m, s = self.orb_labels[i]
            for j in range(i + 1, 4):
                m_, s_ = self.orb_labels[j]
                # print('in build_off_diagonal', m, s, m_, s_)

                # Build the off-diagonal circuit and save to HDF5 file.
                U_op = self.pauli_dict[(m, s)]
                U_op_ = self.pauli_dict[(m_, s_)]
                circ = self.constructor.build_off_diagonal(U_op, U_op_)
                # write_hdf5(h5file, f'circ{m}{s}{m_}{s_}', 'untranspiled', circuit_to_qasm_str(circ))

                # Transpile the circuit and save to HDF5 file.
                circ = transpile_into_berkeley_gates(circ, f"r{m}{s}{m_}{s_}")
                circ_str = convert_circuit_to_string(circ, self.circ_str_type)
                write_hdf5(h5file, f"circ{m}{s}{m_}{s_}", "transpiled", circ_str)

                if self.method == "tomo":
                    for label in self.tomo_labels:
                        # Append tomography and measurement gates and save to HDF5 file.
                        tomo_circ = append_tomography_gates(circ, [2, 3], label)
                        tomo_circ = append_measurement_gates(tomo_circ)
                        circ_str = convert_circuit_to_string(
                            tomo_circ, self.circ_str_type
                        )
                        write_hdf5(h5file, f"circ{m}{s}{m_}{s_}", label, circ_str)

        h5file.close()

    def execute_off_diagonal(self) -> None:
        """Executes the off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")
        # for m in [0, 1]:
        #     for s, s_ in [('u', 'd'), ('d', 'u')]:
        for i in range(4):
            m, s = self.orb_labels[i]
            for j in range(i + 1, 4):
                m_, s_ = self.orb_labels[j]
                if self.method == "exact":
                    print(m, s, m_, s_)
                    dset = h5file[f"circ{m}{s}{m_}{s_}/transpiled"]
                    circ_str = dset[()]
                    circ = convert_string_to_circuit(circ_str, self.circ_str_type)
                    # circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                    result = self.q_instance.execute(circ)
                    psi = result.get_statevector()
                    dset.attrs[f"psi{self.suffix}"] = psi

                else:  # Tomography
                    for label in self.tomo_labels:
                        dset = h5file[f"circ{m}{s}{m_}{s_}/{label}"]
                        circ_str = dset[()]
                        circ = convert_string_to_circuit(circ_str, self.circ_str_type)
                        # circ = QuantumCircuit.from_qasm_str(dset[()].decode())

                        result = self.q_instance.execute(circ)
                        counts = result.get_counts()
                        counts_arr = counts_dict_to_arr(counts.int_raw, n_qubits=4)
                        dset.attrs[f"counts{self.suffix}"] = counts_arr

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

    def save_data(self) -> None:
        """Saves transition amplitudes data to HDF5 file."""
        h5file = h5py.File(self.h5fname, "r+")
        for key in self.keys_diag:
            write_hdf5(h5file, "amp", f"N{self.suffix}", self.N[key])
        h5file.close()

    def run(
        self,
        method: Optional[str] = None,
        build: bool = True,
        execute: bool = True,
        process: bool = True,
    ) -> None:
        """Runs all the functions to compute transition amplitudes.

        Args:
            method: The method to calculate transition amplitudes. Either exact ('exact')
                or tomography ('tomo').
            build: Whether to construct the circuits.
            execute: Whether to execute the calculations.
            process: Whether to process the results.
        """
        assert method in [None, "exact", "tomo"]

        if method is not None:
            self.method = method
        if build:
            self.build_diagonal()
            self.build_off_diagonal()
        if execute:
            self.execute_diagonal()
            self.execute_off_diagonal()
        if process:
            self.process_diagonal()
            self.process_off_diagonal()
            self.save_data()
