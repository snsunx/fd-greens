#iX = np.array([[0, 1j], [1j, 0]])
#ccx_data = [(SwapGate(), [1, 2]), 
#            (Barrier(4), [0, 1, 2, 3]), 
#            (UnitaryGate(iX).control(2), [0, 2, 1]), 
#            (Barrier(4), [0, 1, 2, 3]),
#            (SwapGate(), [1, 2])]

def get_ccz_inst_tups(ctrl1, ctrl2, targ):
    inst_tups = [(XGate(), [ctrl1], []), 
                 (XGate(), [ctrl2], []),
                 (HGate(), [targ], []),
                 (Barrier(4), range(4), []), 
                 (CCXGate(ctrl_state='00'), [ctrl1, ctrl2, targ], []),
                 (Barrier(4), range(4), []),
                 (XGate(), [ctrl1], []), 
                 (XGate(), [ctrl2], []),
                 (HGate(), [targ], [])]
    return inst_tups