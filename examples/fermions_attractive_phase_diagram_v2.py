"""
Spinless Fermions with nearest-neighbor interactions. Using symmetries to reduce size of matrices we are diagonalizing.
"""

import datetime
import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from exact_diag import ed_fermions
import exact_diag.ed_geometry as geom
import exact_diag.ed_symmetry as symm
import exact_diag.fermi_gas as fg

now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
nsites = 16

save_dir = "../data"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

tstart = time.process_time()

# set up parameters to scan
t = 1
# temps = np.linspace(0, 3, 10) * t
temps = np.linspace(0, 3, 2) * t

with np.errstate(divide="ignore"):
    betas = np.divide(1, temps)
    betas[temps == 0] = np.inf

# ints = np.linspace(0, 5, 60) * t
ints = np.linspace(0, 5, 5) * t
ints = np.concatenate((-np.flip(ints, axis=0), ints[1:]))

print("spinless fermions with nearest neighbor interactions")
print("nsites = %d, n interactions = %d, ntemps = %d" % (nsites, len(ints), len(temps)))

gm = geom.Geometry.createSquareGeometry(nsites, 1, 0, 0, bc1_open=False, bc2_open=True)

# translational symmetry
xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)

# get all symmetry operators at the beginning
n_list = np.array([])
k_list = np.array([])
projector_list = []
sf = ed_fermions.fermions(gm, 0, t, us_same_species=0, potentials=0, nspecies=1)

# number subspace projectors
n_species_op = sf.get_sum_op(sf.n_op, 0, format="boson")
n_projs, ns = sf.get_subspace_projs(n_species_op, print_results=False)

# translation op
xtransl_op_full = sf.get_xform_op(xtransl_cycles)

# get all projectors
for n, n_proj in zip(ns, n_projs):
    xtransl_op = n_proj * xtransl_op_full * n_proj.conj().transpose()
    symm_projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx)

    for kx, symm_proj in zip(kxs, symm_projs):
        curr_proj = symm_proj * n_proj
        if curr_proj.size > 0:
            projector_list.append(curr_proj)
            k_list = np.concatenate((k_list, np.array([kx])))
            n_list = np.concatenate((n_list, np.array([n])))

max_proj_size = np.max([p.shape[0] for p in projector_list])
mean_proj_size = np.mean([p.shape[0] for p in projector_list])

print("Subspace projectors: max size=%d, mean size=%.1f, number=%d" %
      (max_proj_size, mean_proj_size, len(projector_list)))

# loop over interactions_4 and solve for each symmetry sector
corrs_sectors = np.zeros((len(projector_list), temps.size, ints.size, gm.nsites))
corrs = np.zeros((temps.size, ints.size, gm.nsites))
corrs_c = np.zeros((temps.size, ints.size, gm.nsites))
struct_fact = np.zeros((temps.size, ints.size, gm.nsites))

dens_sectors = np.zeros((len(projector_list), temps.size, ints.size, gm.nsites))
dens = np.zeros((temps.size, ints.size, gm.nsites))

ns = np.array([])
ks = np.array([])
for ii, U in enumerate(ints):
    u_tstart = time.process_time()

    eigs = []
    # spinless fermions
    sf = ed_fermions.fermions(gm, 0, t, mus=U, us_same_species=U, potentials=0, nspecies=1)

    # for each interaction, solve number and symmetry subspaces independently
    # TODO: could still speed this up by looping over n and k vals. Then would be able to first use one of the
    # projectors and keep that hamiltonian around. Then apply other projectors to it.
    ham = sf.createH(print_results=True)
    for jj, (n, k, proj) in enumerate(zip(n_list, k_list, projector_list)):
        # ham = sf.createH(print_results=1, projector=proj)
        ham_proj = proj * ham * proj.conj().transpose()
        if ham_proj.shape[0] > 1000:
            pr = True
        else:
            pr = False
        eig_vals, eig_vects = sf.diagH(ham_proj, print_results=pr)

        eigs.append(eig_vals)

        dens_curr_sector, _ = sf.get_thermal_exp_sites(eig_vects, eig_vals, sf.n_op, 0, temps,
                                                       projector=proj, format="boson")
        dens_sectors[jj, :, ii, :] = dens_curr_sector.transpose()

        corrs_curr_sector, _, _ = sf.get_thermal_corr_sites(eig_vects, eig_vals, 0, 0, sf.n_op, sf.n_op,
                                                            temps, sites1=np.zeros(gm.nsites),
                                                            sites2=np.arange(0, gm.nsites), projector=proj, format="boson")
        corrs_sectors[jj, :, ii, :] = corrs_curr_sector.transpose()

        # only need to do these once
        if ii == 0 and n == 0:
            ns = np.concatenate((ns, np.ones(eig_vals.size) * n))
            ks = np.concatenate((ks, np.ones(eig_vals.size) * kx))

    dens[..., ii, :] = sf.thermal_avg_combine_sectors(dens_sectors[:, :, ii, :], eigs, temps)
    corrs[..., ii, :] = sf.thermal_avg_combine_sectors(corrs_sectors[:, :, ii, :], eigs, temps)

    u_tend = time.process_time()
    print("interaction %d/%d ran in %.2fs for %d temps" % (ii + 1, ints.size, u_tend - u_tstart, temps.size))

corrs_c = corrs - dens ** 2
sfact = np.fft.fft(corrs_c, axis=-1) / gm.nsites

tend = time.process_time()
print("total time was %fs" % (tend - tstart))

# ###############################
# plot results
# ###############################
# save results
data = {"nn_interactions": ints, "temps": temps, "geometry": gm, "density": dens,
        "correlators": corrs, "correlators_c": corrs_c, "structure_factor": sfact, "datetime": now_str}
fname = os.path.join(save_dir, "%s_spinless_fermion_chain_nsites=%d_pickle.dat" % (now_str, gm.nsites))
with open(fname, 'wb') as f:
        pickle.dump(data, f)

# plot results
n_distinct_corrs = int(np.ceil(0.5 * gm.nsites)) + 1
nrows = int(np.floor(np.sqrt(n_distinct_corrs)))
ncols = int(np.ceil(n_distinct_corrs / float(nrows)))

# correlators
figh_corr = plt.figure(figsize=(14, 9))

dt = temps[1] - temps[0]
du = ints[1] - ints[0]
extent = [temps[0] - 0.5 * dt, temps[-1] + 0.5 * dt, ints[-1] + 0.5 * du, ints[0] - 0.5 * du]

for ii in range(0, n_distinct_corrs):
    plt.subplot(nrows, ncols, ii+1)
    plt.imshow(4 * corrs_c[:, :, ii].transpose(), interpolation=None, cmap='RdGy', vmin=-1, vmax=1,
               extent=extent, aspect='auto')
    plt.colorbar()
    if ii == 0:
        plt.xlabel('Temperature (t)')
        plt.ylabel('Nearest Neighbor U (t)')
    plt.title('d = %d' % ii)

plt.suptitle('correlators, spinless fermions at half-filling\n vs. interaction and temperature, nsite = %d chain' % gm.nsites)

fname = os.path.join(save_dir, "%s_spinless_fermion_chain_nsites=%d_correlators.png" % (now_str, gm.nsites))
figh_corr.savefig(fname)

# structure factor
figh_sfact = plt.figure(figsize=(14, 9))

for ii in range(0, n_distinct_corrs):
    plt.subplot(nrows, ncols, ii+1)
    plt.imshow(4 * np.real(sfact[:, :, ii].transpose()), interpolation=None, cmap='RdGy',
               vmin=-1, vmax=1, extent=extent, aspect='auto')
    plt.xlabel('Temperature (t)')
    plt.ylabel('NN U/t')
    plt.title('k/pi = %0.2f' % (kxs[ii]/np.pi))

plt.suptitle('structure factor, spinless fermions at half-filling\n '
             'vs. interaction and temperature, nsite = %d chain' % gm.nsites)

fname = os.path.join(save_dir, "%s_spinless_fermion_chain_nsites=%d_structure_factor.png" %
                     (now_str, gm.nsites))
figh_sfact.savefig(fname)