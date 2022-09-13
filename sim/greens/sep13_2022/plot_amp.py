import sys
sys.path.append('../../..')

from fd_greens.cirq_ver.plot_utils import display_fidelities

def main():
    for subscript in ['e', 'h']:
        for spin in ['u']:
            display_fidelities('lih_greens_exact', 'lih_greens_tomo_pur', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_tomo_pur')
            display_fidelities('lih_greens_exact', 'lih_greens_tomo2q_pur', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_tomo2q_pur')
            display_fidelities('lih_greens_exact', 'lih_greens_alltomo_pur', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_alltomo_pur')
            display_fidelities('lih_greens_exact', 'lih_greens_alltomo2q_pur', subscript=subscript, spin=spin, figname=f'fid_{subscript}{spin}_alltomo2q_pur')


if __name__ == '__main__':
    main()
