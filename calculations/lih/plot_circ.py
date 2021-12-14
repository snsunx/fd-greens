import itertools
import h5py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

f = h5py.File('lih.hdf5', 'r')
dset = f['eh_tomo']
circ_labels = ['circ0', 'circ1', 'circ01']
tomo_labels = [''.join(x) for x in itertools.product('xyz', repeat=2)]
labels = [''.join(x) for x in itertools.product(circ_labels, tomo_labels)]

for label in labels:
    circ = QuantumCircuit.from_qasm_str(dset.attrs[label])
    fig = circ.draw(output='mpl')
    fig.savefig(f'circs/{label}.png')
