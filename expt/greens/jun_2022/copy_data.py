import h5py




if __name__ == '__main__':
	h5file_sim = h5py.File('lih_3A_sim.h5', 'r')

	for fname in ['lih_3A_expt0.h5', 'lih_3A_expt1.h5', 'lih_3A_expt2.h5', 'lih_3A_expt3.h5']:
		h5file_expt = h5py.File(fname, 'r+')
		del h5file_expt['gs']
		del h5file_expt['es']
		h5file_expt['gs/energy'] = h5file_sim['gs/energy'][()]
		h5file_expt['es/states_e'] = h5file_sim['es/states_e'][:]
		h5file_expt['es/states_h'] = h5file_sim['es/states_h'][:]
		h5file_expt['es/energies_e'] = h5file_sim['es/energies_e'][:]
		h5file_expt['es/energies_h'] = h5file_sim['es/energies_h'][:]
	
	h5file_sim.close()
