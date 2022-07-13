from typing import Mapping

import re
import cirq

from .transpilation import iToffoliGate
from .parameters import QUBIT_OFFSET

class NoiseParameters:
    """Noise parameters on the Berkeley device."""

    def __init__(self, fidelities: Mapping[str, float]) -> None:
        """Initializes a ``NoiseParameters`` object."""
        self.fidelities = fidelities

    @classmethod
    def from_file(cls, fname: str) -> "NoiseParameters":
        """Constructs a ``NoiseParameter`` object from a ``.yml`` file.
        
        Args:
            fname: The ``.yml`` file name.
        """
        with open(fname + '.yml', 'r') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        fidelities = dict()
        for i, line in enumerate(lines):
            if re.findall('.*: .*', line):
                gate_name, fidelity = line.split(': ')
                fidelities[gate_name] = float(fidelity)
            elif line == 's4567:':
                for j in range(4):
                    fidelity = re.findall('\d.\d+', lines[i + 1 + j])[0]
                    fidelities[f'Q{j + QUBIT_OFFSET}'] = float(fidelity)

        return cls(fidelities)
    
    def add_noise_to_circuit(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Adds depolarizing noisy moments to a circuit.
        
        Args:
            circuit: The original circuit without noise.
            
        Returns:
            circuit_noisy: The circuit with noise added.
        """
        circuit_noisy = cirq.Circuit()
        for moment in circuit:
            circuit_noisy.append(moment)    
            
            noisy_moment = []
            for operation in moment:
                gate = operation.gate
                qubits = operation.qubits

                # Extract the fidelities of X(pi/2), CZ/CS/CSD and iToffoli gates.
                # If gate is not one of the above types, set it to None.
                if isinstance(gate, cirq.XPowGate):
                    fidelity = self.fidelities[f"Q{qubits[0].x + QUBIT_OFFSET}"]
                elif isinstance(gate, cirq.CZPowGate):
                    indices = [str(q.x + QUBIT_OFFSET) for q in qubits]
                    index_string = ''.join(sorted(indices))
                    if abs(gate.exponent - 1.0) < 1e-8:
                        fidelity = self.fidelities[f"CZ{index_string}"]
                    elif abs(gate.exponent - 0.5) < 1e-8:
                        fidelity = self.fidelities[f"CS{index_string}"]
                    elif abs(gate.exponent + 0.5) < 1e-8:
                        fidelity = self.fidelities[f"CSD{index_string}"]
                elif isinstance(gate, iToffoliGate):
                    fidelity = self.fidelities["iTOF"]
                else:
                    fidelity = None

                # Append depolarizing channels to the noisy moment if fidelity is not None.
                if fidelity is not None:
                    for q in qubits:
                        noisy_moment.append(cirq.depolarize(p=1 - fidelity)(q))
            
            if noisy_moment != []:
                circuit_noisy.append(cirq.Moment(noisy_moment))

        return circuit_noisy