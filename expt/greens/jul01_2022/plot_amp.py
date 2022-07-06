import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    for subscript in ['e', 'h']:
        for spin in ['u']:
            display_fidelities('lih_greens_exact', 'lih_greens_expt', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_expt')
            display_fidelities('lih_greens_exact', 'lih_greens_expt2q', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_expt2q')
            display_fidelities('lih_greens_exact', 'lih_greens_pur', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_pur')
            display_fidelities('lih_greens_exact', 'lih_greens_pur2q', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_pur2q')



if __name__ == '__main__':
    main()
