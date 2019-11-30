import sys
import glob
import cPickle as pickle
import numpy as np
import re

import ed_nlce

def aggregate_results(file_pattern, fname_cluster_dat, fname_out):

    # load diagonalized cluster files_out
    files = glob.glob(file_pattern)

    # sort these by number ... currently this is fragile because I'm assuming that there are no other
    # numbers in the file name besides for the cluster number.
    expr = '[^\d]*(\d+)[^\d]*'
    nums = np.zeros(len(files))
    for ii, file in enumerate(files):
        match = re.match(expr, file)
        nums[ii] = int(match.group(1))
    indices_sorted = np.argsort(nums)

    files = [file for _,file in sorted(zip(nums, files), key=lambda tuple: tuple[0])]
    nums = nums[indices_sorted]

    # load first cluster to get size
    with open(files[0], 'rb') as f:
        data_first = pickle.load(f)
    num_temps = data_first[2].size
    temps = data_first[6]

    # aggregated quantities
    energies_all = np.zeros((len(files), num_temps))
    entropies_all = np.zeros((len(files), num_temps))
    specific_heats_all = np.zeros((len(files), num_temps))
    szsz_corrs_all = np.zeros((len(files), num_temps))
    clusters = []
    for ii, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        clusters.append(data[0])
        energies_all[ii, :] = data[2]
        entropies_all[ii, :] = data[3]
        specific_heats_all[ii, :] = data[4]
        szsz_corrs_all[ii, :] = data[5]

    # load cluster data
    with open(fname_cluster_dat, 'rb') as f:
        cluster_dat = pickle.load(f)
    max_cluster_order = cluster_dat[0]
    cluster_multiplicities = cluster_dat[1]
    clusters_list = cluster_dat[2]
    sub_cluster_mult = cluster_dat[3]
    order_start_indices = cluster_dat[4]

    # NLCE
    energy_nlce, orders_energy, weight_energy = \
        ed_nlce.get_nlce_exp_val(energies_all[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], 1)

    energy_nlce_euler_resum, energy_euler_orders = ed_nlce.euler_resum(orders_energy, 1)

    entropy_nlce, orders_entropy, weight_entropy = \
        ed_nlce.get_nlce_exp_val(entropies_all[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], 1)

    entropy_nlce_euler_resum, entropy_euler_orders = ed_nlce.euler_resum(orders_entropy, 1)

    specific_heat_nlce, orders_specific_heat, weight_specific_heat = \
        ed_nlce.get_nlce_exp_val(specific_heats_all[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], 1)

    spheat_nlce_euler_resum, spheat_euler_orders = ed_nlce.euler_resum(orders_specific_heat, 1)

    szsz_nlce, orders_szsz, weight_szsz = \
        ed_nlce.get_nlce_exp_val(szsz_corrs_all[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], 1)

    szsz_nlce_euler_resum, szsz_euler_orders = ed_nlce.euler_resum(orders_szsz, 1)

    data_out = [clusters,
                energies_all, energy_nlce, orders_energy, weight_energy, energy_nlce_euler_resum,
                entropies_all, entropy_nlce, orders_entropy, weight_entropy, entropy_nlce_euler_resum,
                specific_heats_all, specific_heat_nlce, orders_specific_heat, weight_specific_heat, spheat_nlce_euler_resum,
                szsz_corrs_all, szsz_nlce, orders_szsz, weight_szsz, szsz_nlce_euler_resum, temps]

    with open(fname_out, 'wb') as f:
        pickle.dump(data_out, f)

    return data_out


if __name__ == "__main__":
    aggregate_results(sys.argv[1], sys.argv[2], sys.argv[3])