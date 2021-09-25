from qubit_indices import *

inds_h = QubitIndices(['10', '00'], n_qubits=2)
inds_e = QubitIndices(['11', '01'], n_qubits=2)

inds_h_copy = inds_h.include_ancilla('1')

print(inds_h.int_form)
print(inds_h.list_form)
print(inds_h.str_form)
print('')
print(inds_h_copy.int_form)
print(inds_h_copy.list_form)
print(inds_h_copy.str_form)
