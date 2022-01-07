def get_quantum_instance(backend,
                         noise_model_name=None,
                         optimization_level=0,
                         initial_layout=None,
                         shots=1):
    # TODO: Write the part for IBMQ backends
    if isinstance(backend, AerBackend):
        if noise_model_name is None:
            q_instance = QuantumInstance(backend=backend, shots=shots)
        else:
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q-research', group='caltech-1', project='main')
            device = provider.get_backend(noise_model_name)
            q_instance = QuantumInstance(
                backend=backend, shots=shots,
                noise_model=NoiseModel.from_backend(device.properties()),
                coupling_map=device.configuration().coupling_map,
                optimization_level=optimization_level,
                initial_layout=initial_layout)
    return q_instance

# TODO: Move these functionalities to the solvers
def save_exc_data(gs_solver: 'GroundStateSolver', 
                  es_solver: 'ExcitedStatesSolver',
                  amp_solver: 'ExcitedAmplitudesSolver',
                  fname: str = 'lih',
                  dsetname: str = 'exc') -> None:
    """Saves excited states data to file.
    
    Args:
        gs_solver: The ground state solver.
        es_solver: The excited states solver.
        amp_solver: The transition amplitudes solver.
        fname: The file name string.
        dsetname: The dataset name string.
    """
    fname += '.hdf5'
    f = h5py.File(fname, 'r+')
    dset = f[dsetname]
    dset.attrs['energy_gs'] = gs_solver.energy
    dset.attrs['energies_s'] = es_solver.energies_s
    dset.attrs['energies_t'] = es_solver.energies_t
    dset.attrs['L'] = amp_solver.L
    dset.attrs['n_states'] = amp_solver.n_states
    f.close()