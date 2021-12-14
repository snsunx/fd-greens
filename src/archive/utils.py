def save_eh_data(gs_solver: 'GroundStateSolver', 
                 es_solver: 'EHStatesSolver',
                 amp_solver: 'EHAmplitudesSolver',
                 fname: str = 'lih',
                 dsetname: str = 'eh') -> None:
    """Saves N+/-1 electron states data to file.
    
    Args:
        gs_solver: The ground state solver.
        es_solver: The N+/-1 electron states solver.
        amp_solver: The transition amplitudes solver.
        fname: The file name string.
        dsetname: The dataset name string.
    """
    #fname += '.hdf5'
    #f = h5py.File(fname, 'r+')
    #dset = f[dsetname]
    #dset.attrs['energy_gs'] = gs_solver.energy
    #dset.attrs['energies_e'] = es_solver.energies_e
    #dset.attrs['energies_h'] = es_solver.energies_h
    #dset.attrs['B_e'] = amp_solver.B_e
    #dset.attrs['B_h'] = amp_solver.B_h
    #e_orb = np.diag(amp_solver.h.molecule.orbital_energies)
    #act_inds = amp_solver.h.act_inds
    #dset.attrs['e_orb'] = e_orb[act_inds][:, act_inds]
    #f.close()