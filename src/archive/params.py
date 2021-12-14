#iX = np.array([[0, 1j], [1j, 0]])
#ccx_data = [(SwapGate(), [1, 2]), 
#            (Barrier(4), [0, 1, 2, 3]), 
#            (UnitaryGate(iX).control(2), [0, 2, 1]), 
#            (Barrier(4), [0, 1, 2, 3]),
#            (SwapGate(), [1, 2])]