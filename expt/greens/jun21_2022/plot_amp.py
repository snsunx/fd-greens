import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    for subscript in ['e', 'h']:
        for spin in ['u']:
            display_fidelities('lih_3A_exact', 'lih_3A_pur', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_pur')

if __name__ == '__main__':
    main()
