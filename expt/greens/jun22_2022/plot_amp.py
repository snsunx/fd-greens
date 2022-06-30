import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    for subscript in ['e', 'h']:
        for spin in ['u', 'd']:
            display_fidelities('lih_3A_exact', 'lih_3A_expt1', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}')

if __name__ == '__main__':
    main()
