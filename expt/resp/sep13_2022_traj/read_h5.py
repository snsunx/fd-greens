import h5py


h5file = h5py.File('nah_resp_circ0u1u2q.h5', 'r')

circ = h5file['circ1/zzzz'][()]
print(circ)

h5file.close()
