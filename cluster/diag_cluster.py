import sys
import pickle
import numpy as np
import time
import exact_diag.ed_spins as tvi
import exact_diag.ed_fermions as hubb
from exact_diag import ed_nlce


def diag_cluster(cluster_index,
                 cluster_data_out_fname,
                 cluster_data_in_fname):
    with open(cluster_data_in_fname, 'rb') as f:
        data = pickle.load(f)
    cluster_list = data[1]
    cluster = cluster_list[cluster_index]

    print("cluster %d" % cluster_index)
    model = tvi.spinSystem(cluster, jx=0.5, jy=0.5, jz=0.5, hx=0, hy=0, hz=0,
                           use_ryd_detunes=False)
    hamiltonian = model.createH(print_results=True)
    eig_vals, eig_vects = model.diagH(hamiltonian, print_results=True)

    temps = np.logspace(-1, 0.5, 50)
    # compute expectation values
    energies = np.zeros((1, len(temps)))
    entropies = np.zeros((1, len(temps)))
    specific_heats = np.zeros((1, len(temps)))
    szsz_corrs = np.zeros((1, len(temps)))

    t_start = time.process_time()
    for jj, T in enumerate(temps):
        # calculate properties for each temperature
        energies[0, jj] = model.get_exp_vals_thermal(eig_vects, hamiltonian, eig_vals, T, False)
        Z = np.sum(np.exp(-eig_vals / T))
        entropies[0, jj] = (np.log(Z) + energies[0, jj] / T)
        specific_heats[0, jj] = 1. / T ** 2 * \
                                 (model.get_exp_vals_thermal(eig_vects,
                                                             hamiltonian.dot(hamiltonian),
                                                             eig_vals,
                                                             T,
                                                             False) - energies[0, jj] ** 2)

        szsz_corrs[0, jj] = 0
        for aa in range(0, model.geometry.nsites):
            for bb in range(aa + 1, model.geometry.nsites):
                szsz_op = model.getTwoSiteOp(aa,
                                             bb,
                                             model.geometry.nsites,
                                             model.pauli_z,
                                             model.pauli_z,
                                             format="boson")
                szsz_exp = model.get_exp_vals_thermal(eig_vects, szsz_op, eig_vals, T)
                szsz_corrs[0, jj] = szsz_corrs[0, jj] + szsz_exp

    t_end = time.process_time()
    print("Computing %d finite temperature expectation values took %0.2fs" % (len(temps), t_end - t_start))

    data_out = [cluster, eig_vals, energies, entropies, specific_heats, szsz_corrs, temps, model]
    with open(cluster_data_out_fname, 'wb') as f:
        pickle.dump(data_out, f)


if __name__ == "__main__":
    # bash arrays start at 1, so lets keep that convention for terminal arguments
    arg1 = int(sys.argv[1]) - 1
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    diag_cluster(arg1, arg2, arg3)
