basis_gates = ['u3', 'cz', 'swap', 'cp']
swap_direcs = [['left', 'left'], ['right', 'right', 'left'], ['right', 'left'], ['right', 'right'], ['right']]
swap_direcs1 = [['left'], ['right', None], [None, None], [], [None]]
#swap_direcs = [x for y in swap_direcs for x in y]