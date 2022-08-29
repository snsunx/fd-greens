"""
===================================================================================
(N±1)-Electron Transition Amplitudes Solver (:mod:`fd_greens.eh_amplitudes_solver`)
===================================================================================
"""

from typing import Sequence, Optional

import h5py
import json
import cirq

from .molecular_hamiltonian import MolecularHamiltonian
from .operators import SecondQuantizedOperators
from .circuit_constructor import CircuitConstructor
from .circuit_string_converter import CircuitStringConverter
from .transpilation import transpile_into_berkeley_gates
from .parameters import CircuitConstructionParameters, Z2TransformInstructions
from .general_utils import histogram_to_array
from .helpers import save_to_hdf5, print_circuit
from .noise_parameters import NoiseParameters


class EHAmplitudesSolver:
    """Solver for transition amplitudes between ground state and (N±1)-electron states."""

    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        qubits: Sequence[cirq.Qid],
        method: str = "exact",
        noise_fname: Optional[str] = None,
        repetitions: int = 10000,
        h5fname: str = "lih",
        suffix: str = "",
    ) -> None:
        """Initializes an ``EHAmplitudesSolver`` object.

        Args:
            hamiltonian: The molecular Hamiltonian.
            qubits: Qubits in the circuit.
            method: The method for calculating the transition amplitudes. Either ``'exact'`` or ``'tomo'``.
            repetitions: Number of repetitions used in simulations.
            h5fname: The HDF5 file name.
            suffix: The suffix for a specific experimental run.
        """
        assert method in ["exact", "tomo", "alltomo"]

        # Basic variables.
        self.hamiltonian = hamiltonian
        self.qubits = qubits
        self.method = method
        self.noise_fname = noise_fname
        self.repetitions = repetitions
        self.h5fname = h5fname
        self.suffix = suffix

        if noise_fname is not None:
            self.simulator = cirq.DensityMatrixSimulator()
            self.noise_params = NoiseParameters.from_file(self.noise_fname)
        else:
            self.simulator = cirq.Simulator()
            self.noise_params = None

        self.circuit_params = CircuitConstructionParameters()
        self.circuit_params.write(h5fname)

        self.circuits = dict()
        self.converter = CircuitStringConverter(self.qubits)

        # Create dictionary of the second quantized operators.
        self.operators = dict()
        for spin in ["u", "d"]:
            operators = SecondQuantizedOperators(self.qubits, spin)
            instructions = Z2TransformInstructions.get_instructions(spin)
            operators.transform(instructions)
            self.operators[spin] = operators

        self.n_spatial_orbitals = len(self.hamiltonian.active_indices)
        self.n_spin_orbitals = 2 * self.n_spatial_orbitals
        self.n_system_qubits = self.n_spin_orbitals - instructions.n_tapered

    def _run_diagonal_circuits(self, spin: str) -> None:
        """Runs diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname + ".h5", "r+")

        for m in range(self.n_spatial_orbitals):
            circuit_label = f"circ{m}{spin}"

            # Build the diagonal circuit based on the second quantized operator.
            circuit = self.constructor.build_diagonal_circuit(
                self.operators[spin][2 * m],
                self.operators[spin][2 * m + 1])

            # Transpile the circuit and save to HDF5 file.
            circuit = transpile_into_berkeley_gates(circuit, spin=spin, circuit_params=self.circuit_params)
            circuit = self.converter.adapt_to_hardware(circuit)
            self.circuits[circuit_label] = circuit
            dset_transpiled = self.converter.save_circuit(h5file, f"{circuit_label}/transpiled", circuit)

            # Execute the circuit and store the statevector in the HDF5 file.
            state_vector = cirq.sim.final_state_vector(circuit)
            dset_transpiled.attrs[f"psi{self.suffix}"] = state_vector

            if self.method in ["tomo", "alltomo"]:
                if self.method == "tomo":
                    tomo_qubits = self.qubits[1:self.n_system_qubits + 1]
                else:
                    tomo_qubits = self.qubits[:self.n_system_qubits + 1]
                measured_qubits = self.qubits[:self.n_system_qubits + 1]
                tomo_circuits = self.constructor.build_tomography_circuits(
                    circuit, tomographed_qubits=tomo_qubits, measured_qubits=measured_qubits)

                for tomo_label, tomo_circuit in tomo_circuits.items():
                    # Save tomography circuit to HDF5 file.
                    self.circuits[circuit_label + tomo_label] = tomo_circuit
                    dset_tomo = self.converter.save_circuit(
                        h5file, f"{circuit_label}/{tomo_label}", tomo_circuit)
                    
                    # Run tomography circuit and store results to HDF5 file.
                    if self.noise_params is not None:
                        tomo_circuit = self.noise_params.add_noise_to_circuit(tomo_circuit)
                    result = self.simulator.run(tomo_circuit, repetitions=self.repetitions)
                    histogram = result.multi_measurement_histogram(
                        keys=[str(q) for q in self.qubits[:self.n_system_qubits + 1]])
                    dset_tomo.attrs[f"counts{self.suffix}"] = histogram_to_array(
                        histogram, n_qubits=self.n_system_qubits + 1)

        h5file.close()

    def _run_off_diagonal_circuits(self, spin: str) -> None:
        """Runs off-diagonal transition amplitude circuits."""
        h5file = h5py.File(self.h5fname + ".h5", "r+")

        for m in range(self.n_spatial_orbitals):
            for n in range(m + 1, self.n_spatial_orbitals):
                circuit_label = f"circ{m}{n}{spin}"

                # Build the off-diagonal circuit based on the creation/annihilation operators.
                circuit = self.constructor.build_off_diagonal_circuit(
                    self.operators[spin][2 * m],
                    self.operators[spin][2 * m + 1], 
                    self.operators[spin][2 * n],
                    self.operators[spin][2 * n + 1])

                # Transpile the circuit and save to HDF5 file.
                circuit = transpile_into_berkeley_gates(
                    circuit, spin=spin, circuit_params=self.circuit_params)
                circuit = self.converter.adapt_to_hardware(circuit)
                self.circuits[circuit_label] = circuit
                dset_transpiled = self.converter.save_circuit(h5file, f"{circuit_label}/transpiled", circuit)

                # Run simulation and save results to HDF5 file.
                state_vector = cirq.sim.final_state_vector(circuit)
                dset_transpiled.attrs[f"psi{self.suffix}"] = state_vector
                print(f"In run_off_diagonal_circuits, {state_vector.shape = }")

                # Apply tomography and measurement gates and save to HDF5 file.
                if self.method in ["tomo", "alltomo"]:
                    if self.method == "tomo":
                        tomographed_qubits = self.qubits[2:self.n_system_qubits + 2]
                    else:
                        tomographed_qubits = self.qubits[:self.n_system_qubits + 2]
                    measured_qubits = self.qubits[:self.n_system_qubits + 2]
                    tomo_circuits = self.constructor.build_tomography_circuits(
                        circuit, tomographed_qubits=tomographed_qubits, measured_qubits=measured_qubits)

                    for tomo_label, tomo_circuit in tomo_circuits.items():
                        # Save tomography circuit to HDF5 file.
                        self.circuits[circuit_label + tomo_label] = tomo_circuit
                        dset_tomo = self.converter.save_circuit(h5file, f"{circuit_label}/{tomo_label}", tomo_circuit)

                        # Run simulation and save result to HDF5 file.
                        if self.noise_params is not None:
                            tomo_circuit = self.noise_params.add_noise_to_circuit(tomo_circuit)
                        result = self.simulator.run(tomo_circuit, repetitions=self.repetitions)
                        histogram = result.multi_measurement_histogram(
                            keys=[str(q) for q in self.qubits[:self.n_system_qubits + 2]])
                        dset_tomo.attrs[f"counts{self.suffix}"] = histogram_to_array(
                            histogram, n_qubits=self.n_system_qubits + 2)

        h5file.close()

    def run(self) -> None:
        """Runs all transition amplitude circuits."""
        print("Start (N±1)-electron amplitudes solver.")
        for spin in ["u", "d"]:
            ansatz = self.converter.load_circuit(self.h5fname, f"gs{spin}/ansatz")
            self.constructor = CircuitConstructor(ansatz, self.qubits)
            self._run_diagonal_circuits(spin)
            self._run_off_diagonal_circuits(spin)
        print("(N±1)-electron amplitudes solver finshed.")
