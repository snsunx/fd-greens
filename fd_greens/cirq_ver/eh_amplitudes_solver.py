"""
===================================================================================
(N±1)-Electron Transition Amplitudes Solver (:mod:`fd_greens.eh_amplitudes_solver`)
===================================================================================
"""

from typing import Optional, Sequence

import h5py
import json
import cirq

from .molecular_hamiltonian import MolecularHamiltonian
from .operators import SecondQuantizedOperators
from .circuit_constructor import CircuitConstructor
from .circuit_string_converter import CircuitStringConverter
from .transpilation import transpile_into_berkeley_gates
from .parameters import method_indices_pairs
from .utilities import histogram_to_array


class EHAmplitudesSolver:
    """Solver for transition amplitudes between ground state and (N±1)-electron states."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        qubits: Sequence[cirq.Qid],
        method: str = "exact",
        repetitions: int = 10000,
        fname: str = "lih",
        spin: str = "d",
        suffix: str = "",
    ) -> None:
        """Initializes an ``EHAmplitudesSolver`` object.

        Args:
            hamiltonian: The molecular Hamiltonian.
            qubits: Qubits in the circuit.
            method: The method for calculating the transition amplitudes. Either ``'exact'`` or ``'tomo'``.
            repetitions: Number of repetitions used in simulations.
            fname: The HDF5 file name.
            spin: The spin of the second-quantized operators. Either ``'u'`` or ``'d'``.
            suffix: The suffix for a specific experimental run.
        """
        assert method in ["exact", "tomo"]
        assert spin in ["u", "d"]

        # Basic variables.
        self.hamiltonian = hamiltonian
        self.qubits = qubits
        self.method = method
        self.repetitions = repetitions
        self.h5fname = fname + ".h5"
        self.spin = spin
        self.suffix = suffix

        self.n_orbitals = len(self.hamiltonian.active_indices)

        self.circuits = dict()
        self.circuit_string_converter = CircuitStringConverter(self.qubits)
        with h5py.File(self.h5fname, 'r') as h5file:
            qtrl_strings = json.loads(h5file["gs/ansatz"][()])
            ansatz = self.circuit_string_converter.convert_strings_to_circuit(qtrl_strings)
        self.circuit_constructor = CircuitConstructor(ansatz, self.qubits)

        # Create dictionary of the second quantized operators.
        self.second_quantized_operators = SecondQuantizedOperators(
            self.qubits, self.spin, factor=(-1) ** (self.spin == 'd'))
        self.second_quantized_operators.transform(method_indices_pairs[spin])
        # for x in self.second_quantized_operators:
        #     print(x)

    def _run_diagonal_circuits(self) -> None:
        """Runs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")

        for m in range(self.n_orbitals):
            circuit_label = f"circ{m}{self.spin}"

            # Build the diagonal circuit based on the second quantized operator.
            circuit = self.circuit_constructor.build_diagonal_circuit(
                self.second_quantized_operators[2 * m],
                self.second_quantized_operators[2 * m + 1])

            # Transpile the circuit and save to HDF5 file.
            circuit = transpile_into_berkeley_gates(circuit, self.spin)
            self.circuits[circuit_label] = circuit
            qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
            dset_transpiled = h5file.create_dataset(f"{circuit_label}/transpiled", data=json.dumps(qtrl_strings))

            # Execute the circuit and store the statevector in the HDF5 file.
            state_vector = cirq.sim.final_state_vector(circuit)
            state_vector[abs(state_vector) < 1e-8] = 0.0
            dset_transpiled.attrs[f"psi{self.suffix}"] = state_vector

            if self.method == "tomo":
                tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                    circuit, self.qubits[1:3], self.qubits[:3]) # XXX: 1:3 is hardcoded

                for tomography_label, tomography_circuit in tomography_circuits.items():
                    # Save tomography circuit to HDF5 file.
                    self.circuits[circuit_label + tomography_label] = tomography_circuit
                    qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(tomography_circuit)
                    dset_tomography = h5file.create_dataset(
                        f"{circuit_label}/{tomography_label}", data=json.dumps(qtrl_strings))

                    # Run tomography circuit and store results to HDF5 file.
                    result = cirq.Simulator().run(tomography_circuit, repetitions=self.repetitions)
                    histogram = result.multi_measurement_histogram(keys=[str(q) for q in self.qubits[:3]])
                    dset_tomography.attrs[f"counts{self.suffix}"] = histogram_to_array(histogram, n_qubits=3)

        h5file.close()

    def _run_off_diagonal_circuits(self) -> None:
        """Runs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname, "r+")

        # TODO: Take in m and n here.
        for m in range(self.n_orbitals):
            for n in range(m + 1, self.n_orbitals):
                circuit_label = f"circ{m}{n}{self.spin}"

                # Build the off-diagonal circuit based on the creation/annihilation operators.
                circuit = self.circuit_constructor.build_off_diagonal_circuit(
                    self.second_quantized_operators[2 * m],
                    self.second_quantized_operators[2 * m + 1], 
                    self.second_quantized_operators[2 * n],
                    self.second_quantized_operators[2 * n + 1])

                # Transpile the circuit and save to HDF5 file.
                circuit = transpile_into_berkeley_gates(circuit, self.spin)
                self.circuits[circuit_label] = circuit
                # print(circuit[:10])
                # print(circuit[10:20])
                # print(circuit[20:])
                qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(circuit)
                # print(qtrl_strings)
                dset_transpiled = h5file.create_dataset(f"{circuit_label}/transpiled", data=json.dumps(qtrl_strings))

                # Run simulation and save results to HDF5 file.
                state_vector = cirq.sim.final_state_vector(circuit)
                dset_transpiled.attrs[f"psi{self.suffix}"] = state_vector

                # Apply tomography and measurement gates and save to HDF5 file.
                if self.method == "tomo":
                    tomography_circuits = self.circuit_constructor.build_tomography_circuits(
                        circuit, self.qubits[2:4], self.qubits[:4]) # XXX: 2:4 is hardcoded

                    for tomography_label, tomography_circuit in tomography_circuits.items():
                        # Save tomography circuit to HDF5 file.
                        self.circuits[circuit_label + tomography_label] = tomography_circuit
                        qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(tomography_circuit)
                        dset_tomography = h5file.create_dataset(
                            f"{circuit_label}/{tomography_label}", data=json.dumps(qtrl_strings))

                        # Run simulation and save result to HDF5 file.
                        result = cirq.Simulator().run(tomography_circuit, repetitions=self.repetitions)
                        histogram = result.multi_measurement_histogram(keys=[str(q) for q in self.qubits[:4]])
                        dset_tomography.attrs[f"counts{self.suffix}"] = histogram_to_array(histogram, n_qubits=4)


        h5file.close()

    def run(self) -> None:
        """Runs all transition amplitude circuits."""
        print("Start (N±1)-electron amplitudes solver.")
        self._run_diagonal_circuits()
        self._run_off_diagonal_circuits()
        print("(N±1)-electron amplitudes solver finshed.")
