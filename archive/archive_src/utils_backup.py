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

    '''
def check_ccx_inst_tups(inst_tups):
    """Checks whether the instruction tuples are equivalent to CCX up to a phase."""
    ccx_inst_tups_matrix = get_unitary(ccx_inst_tups)
    self.ccx_angle = polar(ccx_inst_tups_matrix[3, 7])[1]
    ccx_inst_tups_matrix[3, 7] /= np.exp(1j * self.ccx_angle)
    ccx_inst_tups_matrix[7, 3] /= np.exp(1j * self.ccx_angle)
    ccx_matrix = CCXGate().to_matrix()
    assert np.allclose(ccx_inst_tups_matrix, ccx_matrix)
'''

def counts_arr_to_dict(arr: np.ndarray) -> Counts:
    """Converts bitstring counts from np.ndarray form to qiskit.result.Counts form.
    
    Args:
        arr: The """
    data = {}
    for i in range(arr.shape[0]):
        data[i] = arr[i]

    counts = Counts(data)
    return counts

def block_sum(arr, n_fold=2):
    dim = arr.shape[0]
    dim_new = dim // n_fold
    arr_new = np.zeros((dim_new, dim_new))
    for i in range(dim_new):
        for j in range(dim_new):
            arr_new[i, j] = np.sum(arr[n_fold*i:n_fold*(i+1), n_fold*j:n_fold*(j+1)])
    return arr_new

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

# TODO: Deprecate this function.
def solve_energy_probabilities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.array([[1, 1], [a[0], a[1]]])
    x = np.linalg.inv(A) @ np.array([1.0, b])
    return x


# TODO: Deprecate this function.
def split_counts_on_anc(counts: Union[Counts, np.ndarray], n_anc: int = 1) -> Counts:
    """Splits the counts on ancilla qubit state."""
    if isinstance(counts, Counts):
        counts = counts_dict_to_arr(counts)
    step = 2 ** n_anc
    if n_anc == 1:
        counts0 = counts[::step]
        counts1 = counts[1::step]
        n_counts = np.sum(counts)
        counts0 = counts0 / n_counts
        counts1 = counts1 / n_counts
        # print('np.sum(counts) =', np.sum(counts))
        # print('##########################################################')
        print(counts0, np.sum(counts0))
        print(counts1, np.sum(counts1))
        return counts0, counts1
    elif n_anc == 2:
        counts00 = counts[::step]
        counts01 = counts[1::step]
        counts10 = counts[2::step]
        counts11 = counts[3::step]
        counts00 = counts00 / np.sum(counts)
        counts01 = counts01 / np.sum(counts)
        counts10 = counts10 / np.sum(counts)
        counts11 = counts11 / np.sum(counts)
        return counts00, counts01, counts10, counts11

    # Counts utility functions
def get_counts(result: Result) -> Mapping[str, int]:
    """Returns the counts from a Result object with default set to 0."""
    counts = defaultdict(lambda: 0)
    counts.update(result.get_counts())
    return counts
get_counts_default_0 = get_counts