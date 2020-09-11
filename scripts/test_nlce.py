import os
import time
import datetime
import numpy as np
import pickle
import ed_spins as tvi
import ed_geometry as geom
import ed_symmetry as symm
import ed_nlce as nlce

########################################
# general settings
########################################

now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
max_cluster_order = 8
display_results = True
save_dir = "../data"
fname_full_cluster_ed = os.path.join(save_dir, 'four_by_four_heisenberg_ed.pkl')
fname_clusters = os.path.join(save_dir, 'cluster_data_order=%d.dat' % max_cluster_order)

# create directory for saving results
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if display_results:
    import matplotlib.pyplot as plt

jx = 0.5
jy = 0.5
jz = 0.5
hx = 0.0
hy = 0.0
hz = 0.0
temps = np.logspace(-1, 0.5, 50)

########################################
# diagonalize full cluster
########################################
if not os.path.isfile(fname_full_cluster_ed):
    print("did not find file %s, diagonalizing hamiltonian" % fname_full_cluster_ed)
    parent_geometry = geom.Geometry.createSquareGeometry(4, 4, 0, 0, bc1_open=False, bc2_open=False)
    parent_geometry.permute_sites(parent_geometry.get_sorting_permutation())

    model = tvi.spinSystem(parent_geometry, jx, jy, jz, hx, hy, hz, use_ryd_detunes=False)

    # x-translation
    tx_fn = symm.getTranslFn(np.array([[1], [0]]))
    tx_cycles, ntx = symm.findSiteCycles(tx_fn, model.geometry)
    tx_op = model.get_xform_op(tx_cycles)

    # y-translations
    ty_fn = symm.getTranslFn(np.array([[0], [1]]))
    ty_cycles, nty = symm.findSiteCycles(ty_fn, model.geometry)
    ty_op = model.get_xform_op(ty_cycles)

    # symmetry projectors
    symm_projs, kxs, kys = symm.get2DTranslationProjectors(tx_op, ntx, ty_op, nty, print_results=True)

    # Calculate eigenvalues and expectation values for each symmetry sector
    eig_vals_sectors = []
    energy_exp_sectors = np.zeros((len(symm_projs), len(temps)))
    energy_sqr_exp_sectors = np.zeros((len(symm_projs), len(temps)))
    szsz_exp_sectors = np.zeros((len(symm_projs), len(temps)))
    szsz_op_full = model.get_two_site_op(0, 0, 1, 0, model.pauli_z, model.pauli_z, format="boson")
    for ii, proj in enumerate(symm_projs):
        print("symmetry subspace %d/%d" % (ii + 1, len(symm_projs)))
        H = model.createH(projector=proj, print_results=True)
        eig_vals, eig_vects = model.diagH(H, print_results=True)
        eig_vals_sectors.append(eig_vals)
        szsz_op_sector = proj.dot(szsz_op_full.dot(proj.conj().transpose()))

        t_start = time.process_time()
        for jj in range(0, len(temps)):
            print("temp %d/%d" % (jj+1, len(temps)))
            energy_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, temps[jj], print_results=False)
            energy_sqr_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, temps[jj],
                                                                        print_results=False)
            szsz_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, szsz_op_sector, eig_vals,
                                                                  temps[jj], print_results=False)
        t_end = time.process_time()
        print("Computed %d finite temperature expectation values in %0.2fs" % (len(temps), t_end - t_start))
    eigs_all = np.sort(np.concatenate(eig_vals_sectors))

    # Calculate full eigenvalues and expectation values, combining results from sectors
    energies_full = np.zeros(len(temps))
    entropies_full = np.zeros(len(temps))
    specific_heat_full = np.zeros(len(temps))
    szsz_full = np.zeros(len(temps))
    for jj, temp in enumerate(temps):
        energies_full[jj] = \
            model.thermal_avg_combine_sectors(energy_exp_sectors[:, jj], eig_vals_sectors, temp) / model.geometry.nsites
        Z = np.sum(np.exp(- eigs_all / temp))
        # for entropy calculation, need full energy, so must multiply energy by number of sites again
        entropies_full[jj] = 1. / model.geometry.nsites * (np.log(Z) + energies_full[jj] * model.geometry.nsites / temp)
        # for specific heat, need full energy instead of energy per site
        ham_sqr = model.thermal_avg_combine_sectors(energy_sqr_exp_sectors[:, jj], eig_vals_sectors, temp)
        specific_heat_full[jj] = 1. / (temp ** 2 * model.geometry.nsites) * (
            ham_sqr - (energies_full[jj] * model.geometry.nsites) ** 2)
        # assuming symmetry for whichever sites we choose
        szsz_full[jj] = model.thermal_avg_combine_sectors(szsz_exp_sectors[:, jj], eig_vals_sectors, temp)

    data = {"model": model, "eigs_all": eigs_all, "energies_full": energies_full,
            "entropies_full": entropies_full, "specific_heat_full": specific_heat_full,
            "szsz_full": szsz_full, "temps": temps, "run_date": now_str}
    with open(fname_full_cluster_ed, 'wb') as f:
        pickle.dump(data, f)
else:
    print("found and loaded file %s" % fname_full_cluster_ed)
    with open(fname_full_cluster_ed, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    eigs_all = data['eigs_all']
    energies_full = data['energies_full']
    entropies_full = data['entropies_full']
    specific_heat_full = data['specific_heat_full']
    szsz_full = data['szsz_full']

########################################
# generate all clusters up to certain order on the infinite lattice
########################################

# TODO: make it so can load smaller order data too ...
if os.path.isfile(fname_clusters):
    print("found a loaded cluster data from file %s" % fname_clusters)
    with open(fname_clusters, 'rb') as f:
        data_clusters = pickle.load(f)
    cluster_multiplicities = data_clusters["cluster_multiplicities"]
    clusters_list = data_clusters["clusters_list"]
    sub_cluster_mult = data_clusters["sub_cluster_mult"]
    order_start_indices = data_clusters["order_start_indices"]
else:
    print("cluster data file %s does not exist. Generating clusters." % fname_clusters)
    clusters_list, cluster_multiplicities, sub_cluster_mult, order_start_indices = \
        nlce.get_all_clusters_with_subclusters(max_cluster_order)
    cluster_multiplicities = cluster_multiplicities[None, :]

    data_clusters = {"max_cluster_order": max_cluster_order, "cluster_multiplicities": cluster_multiplicities,
                     "clusters_list": clusters_list, "sub_cluster_mult": sub_cluster_mult,
                     "order_start_indices": order_start_indices}
    # save cluster data
    with open(fname_clusters, 'wb') as f:
        pickle.dump(data_clusters, f)

# initialize variables which will store expectation values
energies = np.zeros((len(clusters_list), len(temps)))
entropies = np.zeros((len(clusters_list), len(temps)))
specific_heats = np.zeros((len(clusters_list), len(temps)))
szsz_corr = np.zeros((len(clusters_list), len(temps)))

for ii, cluster in enumerate(clusters_list):
    print("%d/%d" % (ii + 1, len(clusters_list)))
    model = tvi.spinSystem(cluster, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)
    H = model.createH(print_results=True)
    eig_vals, eig_vects = model.diagH(H, print_results=True)

    t_start = time.process_time()
    for jj, T in enumerate(temps):
        # calculate properties for each temperature
        energies[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, T, 0)
        Z = np.sum(np.exp(-eig_vals / T))
        entropies[ii, jj] = (np.log(Z) + energies[ii, jj] / T)
        specific_heats[ii, jj] = \
            1./T**2 * (model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, T, 0) - energies[ii, jj] ** 2)

        # sum over pairs
        szsz_corr[ii, jj] = 0
        for aa in range(0, model.geometry.nsites):
            for bb in range(aa + 1, model.geometry.nsites):
                szsz_op = model.get_two_site_op(aa, 0, bb, 0, model.pauli_z, model.pauli_z, format="boson")
                szsz_exp = model.get_exp_vals_thermal(eig_vects, szsz_op, eig_vals, T)
                szsz_corr[ii, jj] = szsz_corr[ii, jj] + szsz_exp

    t_end = time.process_time()
    print("Computing %d finite temperature expectation values took %0.2fs" % (len(temps), t_end - t_start))

# nlce computation
# TODO: this in a nicer way ...
energy_nlce, orders_energy, weight_energy = \
    nlce.get_nlce_exp_val(energies[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order], 0:order_start_indices[max_cluster_order]],
                     cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], 1)

energy_nlce_euler_resum, energy_euler_orders = nlce.euler_resum(orders_energy, 1)

entropy_nlce, orders_entropy, weight_entropy = \
    nlce.get_nlce_exp_val(entropies[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order], 0:order_start_indices[max_cluster_order]],
                     cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], 1)

entropy_nlce_euler_resum, entropy_euler_orders = nlce.euler_resum(orders_entropy, 1)

specific_heat_nlce, orders_specific_heat, weight_specific_heat = \
    nlce.get_nlce_exp_val(specific_heats[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order], 0:order_start_indices[max_cluster_order]],
                     cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], 1)

spheat_nlce_euler_resum, spheat_euler_orders = nlce.euler_resum(orders_specific_heat, 1)

szsz_nlce, orders_szsz, weight_szsz = \
    nlce.get_nlce_exp_val(szsz_corr[0:order_start_indices[max_cluster_order], :],
                     sub_cluster_mult[0:order_start_indices[max_cluster_order], 0:order_start_indices[max_cluster_order]],
                     cluster_multiplicities[0, 0:order_start_indices[max_cluster_order]],
                     order_start_indices[0:max_cluster_order + 1], 1)

szsz_nlce_euler_resum, szsz_euler_orders = nlce.euler_resum(orders_szsz, 1)

data_nlce = {"cluster_list": clusters_list, "sub_cluster_mult": sub_cluster_mult,
             "order_start_indices": order_start_indices,
             "energies": energies, "orders_energy": orders_energy, "weight_energy": weight_energy,
             "energy_nlce_euler_resum": energy_nlce_euler_resum, "energy_euler_order": energy_euler_orders,
             "entropies": entropies, "weight_entropy": weight_entropy,
             "specific_heats": specific_heats, "weight_specific_heat": weight_specific_heat,
             "entropy_nlce_euler_resum": entropy_nlce_euler_resum, "entropy_euler_orders": entropy_euler_orders,
             "energy_nlce": energy_nlce, "entropy_nlce": entropy_nlce, "specific_heat_nlce": specific_heat_nlce,
             "spheat_nlce_euler_resum": spheat_nlce_euler_resum, "spheat_euler_orders": spheat_euler_orders,
             "szsz_nlce": szsz_nlce, "orders_szsz": orders_szsz, "weight_szsz": weight_szsz,
             "szsz_nlce_euler_resum": szsz_nlce_euler_resum, "szsz_euler_orders": szsz_euler_orders,
             "temps": temps}

fname_nlce = os.path.join(save_dir, "%s_nlce_results_order_to=%d.pkl" % (now_str, max_cluster_order))
with open(fname_nlce, 'wb') as f:
    pickle.dump(data_nlce, f)

########################################
# plot results
########################################
if display_results:
    figh = plt.figure(figsize=(12, 8))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.2)

    plt.subplot(grid[0, 0])
    plt.semilogx(temps, energies_full)
    leg = ['4x4 cluster pbc']
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_energy[0:ii + 1, ...], axis=0), '.')
        order_str = 'nlce order = %d' % (ii + 1)
        leg.append(order_str)

    plt.semilogx(temps, energy_nlce_euler_resum, '.')
    leg.append('euler resum')


    plt.grid()
    # plt.xlabel('Temperature (J)')
    plt.ylabel('Energy/site (J)')
    # plt.title('Energy/site')
    plt.ylim([-2, 0.25])
    plt.legend(leg)

    plt.subplot(grid[0, 1])
    plt.semilogx(temps, entropies_full)
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_entropy[0 : ii + 1, ...], 0), '.')
    plt.semilogx(temps, entropy_nlce_euler_resum, '.')
    plt.grid()
    # plt.xlabel('Temperature (J)')
    plt.ylabel('Entropy/site')
    plt.ylim([0, 2])
    # plt.title('Entropy/site')

    plt.subplot(grid[1, 0])
    plt.semilogx(temps, specific_heat_full)
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_specific_heat[0 : ii + 1, ...], 0), '.')
    plt.semilogx(temps, spheat_nlce_euler_resum, '.')
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Specific heat/site ()')
    plt.ylim([0, 2])
    # plt.title('Specific heat / site')

    plt.subplot(grid[1, 1])
    plt.semilogx(temps, szsz_full)
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_szsz[0: ii + 1, ...], 0), '.')
    plt.semilogx(temps, szsz_nlce_euler_resum, '.')
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('SzSz mean ()')
    plt.ylim([-1, 1])

    fig_name = os.path.join(save_dir, "%s_nlce_results.png" % now_str)
    figh.savefig(fig_name)
    plt.show()
