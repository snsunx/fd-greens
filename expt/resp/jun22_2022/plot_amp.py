import numpy as np
import h5py
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


def load_data(fname, **kwargs) -> None:
    if 'exact' in fname:
        load_exact_data(fname, **kwargs)
    else:
        load_expt_data(fname, **kwargs)


def load_exact_data(fname) -> None:
    """Loads exact data."""
    global psi_n, psi_np, psi_nm, N_n_exact, N_summed_n_exact, T_np_exact, T_nm_exact

    psi_n = dict()
    psi_np = dict()
    psi_nm = dict()

    probs = np.zeros((4,))

    with h5py.File(fname + '.h5', 'r') as h5file:
        N_n_exact = h5file['amp/N_n'][:]
        N_summed_n_exact = h5file['amp/N_summed_n'][:]
        T_np_exact = h5file['amp/T_np'][:]
        T_nm_exact = h5file['amp/T_nm'][:]
        for i in range(4):
            psi_n[i] = h5file[f'psi/n{i}'][:]
            probs[i] = (psi_n[i].conj() @ psi_n[i]).real
            print(f"probs[{i}] = ", probs[i])
            for j in range(i + 1, 4):
                psi_np[(i, j)] = h5file[f'psi/np{i}{j}'][:]
                psi_nm[(i, j)] = h5file[f'psi/nm{i}{j}'][:]

    # Sanity check. Both should add up to 1.
    # print(probs[0] + probs[2])
    # print(probs[1] + probs[3])

def load_expt_data(fname, suffix=None) -> None:
    """Loads experimental data."""
    global rho_n, rho_np, rho_nm, N_n_expt, N_summed_n_expt, T_np_expt, T_nm_expt

    rho_n = dict()
    rho_np = dict()
    rho_nm = dict()
    suffix = ''

    probs = np.zeros((4,))

    with h5py.File(fname + '.h5', 'r') as h5file:
        N_n_expt = h5file[f'amp{suffix}/N_n'][:]
        N_summed_n_expt = h5file[f'amp{suffix}/N_summed_n'][:]
        T_np_expt = h5file[f'amp{suffix}/T_np'][:]
        T_nm_expt = h5file[f'amp{suffix}/T_nm'][:]
        for i in range(4):
            rho_n[i] = h5file[f'rho{suffix}/n{i}'][:]
            probs[i] = np.sum(np.diag(rho_n[i])).real
            print(f"probs[{i}] = ", probs[i])
            for j in range(i + 1, 4):
                rho_np[(i, j)] = h5file[f'rho{suffix}/np{i}{j}'][:]
                rho_nm[(i, j)] = h5file[f'rho{suffix}/nm{i}{j}'][:]
                
    # Sanity check. Both should add up to 1.
    # print(probs[0] + probs[2])
    # print(probs[1] + probs[3])


def plot(amp_exact, amp_expt, amp_name='N') -> None:
    for i in range(4):
        if amp_name == 'N':
            j_start = i
        else:
            j_start = i + 1
        for j in range(j_start, 4):
            fig, ax = plt.subplots()
            ax.plot(amp_exact[i, j].real, ls='', marker='o', label='Exact')
            ax.plot(amp_expt[i, j].real, ls='', marker='o', label='Expt')
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels([0, 1, 2])
            ax.set_xlabel('State index')
            ax.set_ylabel('Value')
            ax.legend()
            fig.savefig(f'figs/amp/{amp_name}{i}{j}.png', dpi=250, bbox_inches='tight')

    
def get_fidelity(rho, psi):
    numer = psi.conj() @ rho @ psi
    denom = psi.conj() @ psi * np.sum(np.diag(rho))
    fid = numer / denom
    return fid.real

def plot_fid():

    fidelities = np.zeros((4, 4))
    probs = np.zeros((4, 4))

    for i in range(4):
        fidelities[i, i] = get_fidelity(rho_n[i], psi_n[i])
        # probs[i, i] = (psi_n[i].conj() @ psi_n[i]).real
        probs[i, i] = np.sum(np.diag(rho_n[i]))
        print(f"purity[{i}] = ", (np.trace(rho_n[i] @ rho_n[i]) / np.trace(rho_n[i]) / probs[i, i]).real)
        for j in range(i + 1, 4):
            fidelities[i, j] = get_fidelity(rho_np[(i, j)], psi_np[(i, j)])
            fidelities[j, i] = get_fidelity(rho_nm[(i, j)], psi_nm[(i, j)])
            # probs[i, j] = (psi_np[(i, j)].conj() @ psi_np[(i, j)]).real
            # probs[j, i] = (psi_nm[(i, j)].conj() @ psi_nm[(i, j)]).real
            probs[i, j] = np.sum(np.diag(rho_np[(i, j)]))
            probs[j, i] = np.sum(np.diag(rho_nm[(i, j)]))
            print(f"purity[{i}, {j}] = ", (np.trace(rho_np[(i, j)] @ rho_np[(i, j)]) / np.trace(rho_np[(i, j)]) / probs[i, j]).real)
            print(f"purity[{j}, {i}] = ", (np.trace(rho_nm[(i, j)] @ rho_nm[(i, j)]) / np.trace(rho_nm[(i, j)]) / probs[j, i]).real)




    print('fid\n', fidelities)
    print('probs\n', probs)


    plt.clf()
    plt.imshow(fidelities)
    plt.colorbar()
    plt.savefig('figs/amp/fid.png', dpi=250)

    plt.clf()
    plt.imshow(probs)
    plt.colorbar()
    plt.savefig('figs/amp/probs.png', dpi=250)


if __name__ == '__main__':
    load_data('lih_resp_exact')
    load_data('lih_resp_expt')

    # plot(N_n_exact, N_n_expt, 'N')
    # plot(T_np_exact, T_np_expt, 'T_np')
    # plot(T_nm_exact, T_nm_expt, 'T_nm')
    plot_fid()

