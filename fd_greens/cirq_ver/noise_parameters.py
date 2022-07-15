from typing import Mapping

import re
import cirq

from .transpilation import iToffoliGate
from .parameters import QUBIT_OFFSET

class NoiseParameters:
    """Noise parameters on the Berkeley device."""

    def __init__(self, noise_channels: Mapping[str, cirq.NOISE_MODEL_LIKE]) -> None:
        """Initializes a ``NoiseParameters`` object."""
        self.noise_channels = noise_channels

    @classmethod
    def from_file(cls, fname: str) -> "NoiseParameters":
        """Constructs a ``NoiseParameter`` object from a ``.yml`` file.
        
        Args:
            fname: The ``.yml`` file name.
        """
        with open(fname + '.yml', 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        noise_channels = dict()
        for i, line in enumerate(lines):
            if re.findall('^\w+:\s*0\.\d+$', line):
                gate_name, fidelity = line.split(':')
                noise_channels[gate_name] = cirq.depolarize(p=1 - float(fidelity))
            elif line == 's4567:':
                for j in range(4): # 4 is hardcoded, which comes from the number of qubits used
                    fidelity = re.findall('0\.\d+', lines[i + 1 + j])[0]
                    noise_channels[f'Q{j + QUBIT_OFFSET}'] = cirq.depolarize(p=1 - float(fidelity))

        return cls(noise_channels)
    
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
                    channel = self.noise_channels[f"Q{qubits[0].x + QUBIT_OFFSET}"]
                elif isinstance(gate, cirq.CZPowGate):
                    indices = [str(q.x + QUBIT_OFFSET) for q in qubits]
                    index_string = ''.join(sorted(indices))
                    if abs(gate.exponent - 1.0) < 1e-8:
                        channel = self.noise_channels[f"CZ{index_string}"]
                    elif abs(gate.exponent - 0.5) < 1e-8:
                        channel = self.noise_channels[f"CS{index_string}"]
                    elif abs(gate.exponent + 0.5) < 1e-8:
                        channel = self.noise_channels[f"CSD{index_string}"]
                    else:
                        channel = None
                elif isinstance(gate, iToffoliGate):
                    channel = self.noise_channels["iTOF"]
                else:
                    channel = None

                # Append depolarizing channels to the noisy moment if fidelity is not None.
                if channel is not None:
                    for q in qubits:
                        noisy_moment.append(channel(q))
            
            if noisy_moment != []:
                circuit_noisy.append(cirq.Moment(noisy_moment))

        return circuit_noisy