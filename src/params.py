from qubit_indices import QubitIndices

basis_gates = ['u3', 'cz', 'swap', 'cp']
swap_direcs_round1 = {(0, 1): [['left', 'left'], ['right', 'right', 'left'], ['right', 'left'], ['right', 'right'], ['right']],
                      (1, 0): [['left'], ['right', 'right'], ['left', 'left', 'left'], ['right', 'right', 'left'], ['right']]}
swap_direcs_round2 = {(0, 1): [['left', None], [None, None, None], [None, None], [], [None]],
                      (1, 0): [[None], [], [None, None, None], [None, None, None], [None]]}

eu_inds = QubitIndices(['1101', '0111'])
ed_inds = QubitIndices(['1110', '1011'])
hu_inds = QubitIndices(['0100', '0001'])
hd_inds = QubitIndices(['1000', '0010'])