import time
import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ed_geometry as geom
import ed_spins as tvi
from ed_nlce import *

########################################
# general settings
########################################

max_cluster_order = 9
do_thermodynamic_limit = False
max_cluster_order_thermo_limit = 7
diag_full_cluster = False
display_results = True
temps = np.logspace(-1, 0.5, 50)
nts = len(temps)

save_dir = "data"

########################################
# create geometry
########################################
parent_geometry = geom.Geometry.createSquareGeometry(4, 4, 0, 0, bc1_open=True, bc2_open=True)
parent_geometry.permute_sites(parent_geometry.get_sorting_permutation())
clusters_list, sub_cluster_mult, order_start_indices = get_reduced_subclusters(parent_geometry)
ncs = len(clusters_list)

jx = 0.5
jy = 0.5
jz = 0.5
hx = 0.0
hy = 0.0
hz = 0.0

########################################
# diagonalization of each cluster and computation of properties for each cluster
########################################
# initialize variables which will store expectation values
energies = np.zeros((ncs, nts))
entropies = np.zeros(energies.shape)
specific_heats = np.zeros(energies.shape)

for ii, cluster in enumerate(clusters_list):
    if cluster.nsites > max_cluster_order:
        continue

    print("cluster %d/%d" % (ii + 1, ncs))
    model = tvi.spinSystem(cluster, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)
    H = model.createH(print_results=True)
    eig_vals, eig_vects = model.diagH(H, print_results=True)

    t_start = time.process_time()
    for jj, T in enumerate(temps):
        # calculate properties for each temperature
        energies[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, T, 0)
        Z = np.sum(np.exp(-eig_vals / T))
        entropies[ii, jj] = (np.log(Z) + energies[ii, jj] / T)
        specific_heats[ii, jj] = 1. / T ** 2 * \
                                 (model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, T, 0) - energies[ii, jj] ** 2)
    t_end = time.process_time()
    print("Computing %d finite temperature expectation values took %0.2fs" % (len(temps), t_end - t_start))

# nlce computation
parent_cluster_mult_vector = sub_cluster_mult[-1, :]
if max_cluster_order == parent_geometry.nsites:
    parent_cluster_mult_vector[0, -1] = 1

energy_nlce, orders_energy, weight_energy = \
    get_nlce_exp_val(energies[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order],
                     0:order_start_indices[max_cluster_order]],
                     parent_cluster_mult_vector[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], parent_geometry.nsites)

entropy_nlce, order_entropy, weight_entropy = \
    get_nlce_exp_val(entropies[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order],
                     0:order_start_indices[max_cluster_order]],
                     parent_cluster_mult_vector[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], parent_geometry.nsites)

specific_heat_nlce, orders_specific_heat, weight_specific_heat = \
    get_nlce_exp_val(specific_heats[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order],
                     0:order_start_indices[max_cluster_order]],
                     parent_cluster_mult_vector[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], parent_geometry.nsites)

########################################
# diagonalize full cluster
########################################
if diag_full_cluster:
    model = tvi.spinSystem(parent_geometry, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)

    cx, cy = model.geometry.get_center_of_mass()
    # rot-fn
    rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
    rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, model.geometry)
    rot_op = model.get_xform_op(rot_cycles)

    # reflection about y-axis
    refl_fn = symm.getReflFn(np.array([0, 1]), cx=cx, cy=cy)
    refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, model.geometry)
    refl_op = model.get_xform_op(refl_cycles)

    # symmetry projectors
    # symm_projs = symm.getC4VProjectors(rot_op, refl_op)
    symm_projs, _ = symm.getZnProjectors(rot_op, 4, print_results=True)

    # TODO calculate energy eigs for each...
    eig_vals_sectors = []
    energy_exp_sectors = np.zeros((len(symm_projs), len(temps)))
    energy_sqr_exp_sectors = np.zeros((len(symm_projs), len(temps)))
    for ii, proj in enumerate(symm_projs):
        H = model.createH(projector=proj, print_results=True)
        eig_vals, eig_vects = model.diagH(H, print_results=True)
        eig_vals_sectors.append(eig_vals)
        for jj in range(0, len(temps)):
            energy_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, temps[jj], print_results=False)
            energy_sqr_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, temps[jj],
                                                                        print_results=False)
    eigs_all = np.sort(np.concatenate(eig_vals_sectors))

    energies_full = np.zeros(len(temps))
    entropies_full = np.zeros(len(temps))
    specific_heat_full = np.zeros(len(temps))
    for jj, temp in enumerate(temps):
        energies_full[jj] = model.thermal_avg_combine_sectors(energy_exp_sectors[:, jj], eig_vals_sectors,
                                                              temp) / model.geometry.nsites
        Z = np.sum(np.exp(- eigs_all / temp))
        # for entropy calculation, need full energy, so must multiply energy by number of sites again
        entropies_full[jj] = 1. / model.geometry.nsites * (
            np.log(Z) + energies_full[jj] * model.geometry.nsites / temp)
        # for specific heat, need full energy instead of energy per site
        ham_sqr = model.thermal_avg_combine_sectors(energy_sqr_exp_sectors[:, jj], eig_vals_sectors, temp)
        specific_heat_full[jj] = 1. / (temp ** 2 * model.geometry.nsites) * (
            ham_sqr - (energies_full[jj] * model.geometry.nsites) ** 2)

########################################
# generate all clusters up to certain order on the infinite lattice
########################################

if do_thermodynamic_limit:
    # clusters_list, sub_cluster_mult, order_start_indices = get_reduced_subclusters(parent_geometry)
    # TODO: want cluster order indices from this function also
    clusters_list_tl, cluster_multiplicities_tl, sub_cluster_mult_tl = \
        get_all_clusters_with_subclusters(max_cluster_order_thermo_limit)

    # initialize variables which will store expectation values
    energies_tl = np.zeros((len(clusters_list_tl), len(temps)))
    entropies_tl = np.zeros((len(clusters_list_tl), len(temps)))
    specific_heats_tl = np.zeros((len(clusters_list_tl), len(temps)))

    for ii, cluster in enumerate(clusters_list_tl):
        print("%d/%d" % (ii + 1, len(clusters_list_tl)))
        model = tvi.spinSystem(cluster, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)
        H = model.createH(print_results=True)
        eig_vals, eig_vects = model.diagH(H, print_results=True)

        t_start = time.process_time()
        for jj, T in enumerate(temps):
            # calculate properties for each temperature
            energies_tl[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, T, 0)
            Z_tl = np.sum(np.exp(-eig_vals / T))
            entropies_tl[ii, jj] = (np.log(Z_tl) + energies_tl[ii, jj] / T)
            specific_heats_tl[ii, jj] = 1. / T ** 2 * \
                                        (model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, T, 0) - energies_tl[
                                            ii, jj] ** 2)
        t_end = time.process_time()
        print("Computing %d finite temperature expectation values took %0.2fs" % (len(temps), t_end - t_start))

    # nlce computation
    energy_nlce_tl, weight_energy_tl = \
        get_nlce_exp_val(energies_tl, sub_cluster_mult_tl, cluster_multiplicities_tl, 1)
    entropy_nlce_tl, weight_entropy_tl = \
        get_nlce_exp_val(entropies_tl, sub_cluster_mult_tl, cluster_multiplicities_tl, 1)
    specific_heat_nlce_tl, weight_specific_heat_tl = \
        get_nlce_exp_val(specific_heats_tl, sub_cluster_mult_tl, cluster_multiplicities_tl, 1)

########################################
# plot results
########################################
if display_results:
    fig_handle = plt.figure()
    nrows = 2
    ncols = 2

    plt.subplot(nrows, ncols, 1)
    plt.semilogx(temps, energies[-1, :] / parent_geometry.nsites)
    plt.semilogx(temps, energy_nlce)
    for ii in range(6, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_energy[0:ii, ...], 0))

    if diag_full_cluster:
        plt.semilogx(temps, energies_full)
    if do_thermodynamic_limit:
        plt.semilogx(temps, energy_nlce_tl)

    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Energy/site (J)')
    plt.title('Energy/site')
    plt.ylim([-2, 0.25])
    plt.legend(['last cluster', 'finite nlce', 'full cluster',
                'thermo limit nlce order = %d' % max_cluster_order_thermo_limit])

    plt.subplot(nrows, ncols, 2)
    plt.semilogx(temps, entropies[-1, :] / parent_geometry.nsites)
    plt.semilogx(temps, entropy_nlce)
    for ii in range(6, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(order_entropy[0:ii, ...], 0))
    if diag_full_cluster:
        plt.semilogx(temps, entropies_full)
    if do_thermodynamic_limit:
        plt.semilogx(temps, entropy_nlce_tl)
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Entropy/site ()')
    plt.ylim([0, 2])
    # plt.title('Entropy/site')

    plt.subplot(nrows, ncols, 3)
    plt.semilogx(temps, specific_heats[-1, :] / parent_geometry.nsites)
    plt.semilogx(temps, specific_heat_nlce)
    for ii in range(6, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_specific_heat[0:ii, ...], 0))
    if diag_full_cluster:
        plt.semilogx(temps, specific_heat_full)
    if do_thermodynamic_limit:
        plt.semilogx(temps, specific_heat_nlce_tl)
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Specific heat/site ()')
    plt.ylim([0, 2])
    # plt.title('Specific heat / site')

########################################
# save data
########################################

data = {"cluster_list": clusters_list, "sub_cluster_mult": sub_cluster_mult,
        "order_start_indices": order_start_indices, "temps": temps,
        "energies": energies, "weight_energy": weight_energy,
        "entropies": entropies, "weight_entropy": weight_entropy,
        "specific_heats": specific_heats, "weight_specific_heat": weight_specific_heat,
        "energy_nlce": energy_nlce, "entropy_nlce": entropy_nlce, "specific_heat_nlce": specific_heat_nlce}

if diag_full_cluster:
    data.update({"energies_full": energies_full, "entropies_full": entropies_full,
                 "specific_heat_full": specific_heat_full})
if do_thermodynamic_limit:
    data.update({"energies_tl": energies_tl, "energy_nlce_tl": energy_nlce_tl, "weight_energy_tl": weight_energy_tl,
                 "entropies_tl": entropies_tl, "entropy_ncle_tl": entropy_nlce_tl, "weight_entropy_tl": weight_entropy_tl,
                 "specific_heats_tl": specific_heats_tl, "specific_heat_nlce_tl": specific_heat_nlce_tl,
                 "weight_specific_heat_tl": weight_specific_heat_tl})

today_str = datetime.datetime.today().strftime('%Y-%m-%d_%H;%M;%S')

if display_results:
    fig_name = "%s_nlce_results.png" % today_str
    fig_handle.savefig(fig_name)
    plt.show()

fname = "%s_nlce_test.dat" % today_str
with open(fname, 'wb') as f:
    pickle.dump(data, f)

