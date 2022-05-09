"""
========================================================================================
(N±1)-Electron Transition Amplitudes Solver (:mod:`fd_greens.main.eh_amplitudes_solver`)
========================================================================================
"""

from itertools import product
from typing import Optional, Sequence
from functools import partial

import h5py
import json
import numpy as np
from scipy.special import binom

import cirq
from fd_greens.cirq_ver.utils import tomography_utils

from fd_greens.cirq_ver.utils.circuit_string_converter import CircuitStringConverter

from .qubit_indices import e_inds, h_inds
from .molecular_hamiltonian import MolecularHamiltonian
from .operators import SecondQuantizedOperators
from .z2symmetries import transform_4q_pauli, transform_4q_indices
from .circuit_constructor import CircuitConstructor

from ..utils import (
    print_information,
    transpile_into_berkeley_gates,
    histogram_to_array
)

np.set_printoptions(precision=6)


class EHAmplitudesSolver:
    """Transition amplitudes between ground state and (N±1)-electron states."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        qubits: Sequence[cirq.Qid],
        method: str = "exact",
        h5fname: str = "lih",
        spin: str = "d",
        suffix: str = "",
        verbose: bool = True,
    ) -> None:
        """Initializes an ``EHAmplitudesSolver`` object.

        Args:
            h: The molecular Hamiltonian.
            qubits: Qubits in the circuit.
            method: The method for calculating the transition amplitudes. Either ``'exact'`` or ``'tomo'``.
            h5fname: The HDF5 file name.
            spin: The spin of the creation and annihilation operators.
            suffix: The suffix for a specific experimental run.
            verbose: Whether to print out information about the calculation.
        """
        assert method in ["exact", "tomo"]
        assert spin in ["u", "d"]

        # Basic variables.
        self.h = h
        self.qubits = qubits
        self.method = method
        self.h5fname = h5fname + ".h5"
        self.spin = spin
        self.suffix = suffix
        self.verbose = verbose

        self.circuits = dict()
        self.circuit_string_converter = CircuitStringConverter(self.qubits)
        h5file = h5py.File(self.h5fname, 'r')
        with h5py.File(self.h5fname, 'r') as h5file:
            qtrl_strings = json.loads(h5file["gs/ansatz"][()])
            ansatz = self.circuit_string_converter.convert_strings_to_circuit(qtrl_strings)
        self.circuit_constructor = CircuitConstructor(ansatz, self.qubits)

        # Create dictionary for the creation/annihilation operators.
        second_quantized_operators = SecondQuantizedOperators(self.h.molecule.n_electrons)
        second_quantized_operators.transform(partial(transform_4q_pauli, init_state=[1, 1]))
        self.operators_dict = second_quantized_operators.get_operators_dict()

    def _run_diagonal_circuits(self) -> None:
        """Runs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orb):
            # Create circuit label.
            circuit_label = f"circ{m}{self.spin}"
            if f"{circuit_label}/transpiled" in h5file:
                del h5file[f"{circuit_label}/transpiled"]

            # Build the diagonal circuit based on the creation/annihilation operator.
            a_operator = self.operators_dict[(m, self.spin)]
            circuit = self.circuit_constructor.build_diagonal_circuit(a_operator)

            # Transpile the circuit and save to HDF5 file.
            circuit = transpile_into_berkeley_gates(circuit)
            self.circuits[circuit_label] = circuit
            qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
            dset_transpiled = h5file.create_dataset(
                f"{circuit_label}/transpiled", data=json.dumps(qtrl_strings)
            )

            # Execute the circuit and store the statevector in the HDF5 file.
            state_vector = cirq.sim.final_state_vector(circuit)
            state_vector[abs(state_vector) < 1e-8] = 0.0
            dset_transpiled.attrs[f"psi{self.suffix}"] = state_vector

            if self.method == "tomo":
                tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                    circuit, self.qubits[1:3]
                )

                for tomography_label, tomography_circuit in tomography_circuits.items():
                    self.circuits[circuit_label + tomography_label] = tomography_circuit

                    if f"{circuit_label}/{tomography_label}" in h5file:
                        del h5file[f"{circuit_label}/{tomography_label}"]
                    qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(
                        tomography_circuit
                    )
                    dset_tomography = h5file.create_dataset(
                        f"{circuit_label}/{tomography_label}", 
                        data=json.dumps(qtrl_strings))

                    result = cirq.Simulator().run(tomography_circuit)
                    histogram = result.multi_measurement_histogram(keys=[str(q) for q in self.qubits[1:3]])
                    dset_tomography.attrs[f"counts{self.suffix}"] = histogram_to_array(histogram)

        h5file.close()

    def _run_off_diagonal_circuits(self) -> None:
        """Runs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")

        circuit_label = f"circ01{self.spin}"
        if f"{circuit_label}/transpiled" in h5file:
            del h5file[f"{circuit_label}/transpiled"]

        # Build the off-diagonal circuit based on the creation/annihilation operators.
        a_operator_0 = self.operators_dict[(0, self.spin)]
        a_operator_1 = self.operators_dict[(1, self.spin)]
        circuit = self.circuit_constructor.build_off_diagonal_circuit(a_operator_0, a_operator_1)

        # Transpile the circuit and save to HDF5 file.
        circuit = transpile_into_berkeley_gates(circuit)
        self.circuits[circuit_label] = circuit
        qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
        dset_transpiled = h5file.create_dataset(f"{circuit_label}/transpiled", data=json.dumps(qtrl_strings))

        # Run simulation and save results to HDF5 file.
        state_vector = cirq.sim.final_state_vector(circuit)
        dset_transpiled.attrs[f"psi{self.suffix}"] = state_vector

        # Apply tomography and measurement gates and save to HDF5 file.
        if self.method == "tomo":
            tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                circuit, self.qubits[2:4]
            )

            for tomography_label, tomography_circuit in tomography_circuits.items():
                if f"{circuit_label}/{tomography_label}" in h5file:
                    del h5file[f"{circuit_label}/{tomography_label}"]

                self.circuits[circuit_label + tomography_label] = tomography_circuit
                qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(
                    tomography_circuit
                )
                dset_tomography = h5file.create_dataset(
                    f"{circuit_label}/{tomography_label}", data=json.dumps(qtrl_strings))

                # Run simulation and save result to HDF5 file.
                result = cirq.Simulator().run(tomography_circuit)
                histogram = result.multi_measurement_histogram(keys=[str(q) for q in self.qubits[2:4]])
                dset_tomography.attrs[f"counts{self.suffix}"] = histogram_to_array(histogram)

        h5file.close()

    def run(self, method: Optional[str] = None) -> None:
        """Runs all transition amplitude circuits.
        
        Args:
            method: The method to calculate transition amplitudes. Either exact 
                (``"exact"``) or tomography (``"tomo"``).
        """
        assert method in [None, "exact", "tomo"]

        if method is not None:
            self.method = method
        self._run_diagonal_circuits()
        self._run_off_diagonal_circuits()
