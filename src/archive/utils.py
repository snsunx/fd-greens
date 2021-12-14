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

def save_circuit(circ: QuantumCircuit, 
                 fname: str,
                 savetxt: bool = True,
                 savefig: bool = True) -> None:
    """Saves a circuit to disk in QASM string form and/or figure form.
    
    Args:
        fname: The file name.
        savetxt: Whether to save the QASM string of the circuit as a text file.
        savefig: Whether to save the figure of the circuit.
    """
        
    if savefig:
        fig = circ.draw(output='mpl')
        fig.savefig(fname + '.png')
    
    if savetxt:
        circ_data = []
        for inst_tup in circ.data:
            if inst_tup[0].name != 'c-unitary':
                circ_data.append(inst_tup)
        circ = data_to_circuit(circ_data, remove_barr=False)
        # for inst_tup in circ_data:
        #    print(inst_tup[0].name)
        # exit()
        f = open(fname + '.txt', 'w')
        qasm_str = circ.qasm()
        f.write(qasm_str)
        f.close()