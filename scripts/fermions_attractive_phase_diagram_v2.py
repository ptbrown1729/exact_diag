from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm
import fermi_gas as fg
import datetime
import time
#import cPickle as pickle
import pickle

# TODO: use translation and number to get smaller sectors to solve larger systems faster.

tstart = time.clock()

# set up parameters to scan
t = 1
temps = np.linspace(0, 3, 8) * t
betas = np.divide(1, temps)
betas[temps == 0] = np.inf

ints = np.linspace(0, 10, 15) * t
ints = np.concatenate(( -np.flip(ints, axis=0), ints[1:]))
# ints = np.array([0])

gm = geom.Geometry.createSquareGeometry(8, 1, 0, 0, bc1_open=0, bc2_open=1)

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
n_projs, ns = sf.get_subspace_projs(n_species_op, print_results=0)

# translation op
xtransl_op_full = sf.get_xform_op(xtransl_cycles)

# get all projectors
for n, n_proj in zip(ns, n_projs):
    xtransl_op = n_proj * xtransl_op_full * n_proj.conj().transpose()
    symm_projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx)

    for kx, symm_proj in zip(kxs, symm_projs):
        curr_proj = symm_proj*n_proj
        if curr_proj.size > 0:
            projector_list.append(curr_proj)
            k_list = np.concatenate((k_list, np.array([kx])))
            n_list = np.concatenate((n_list, np.array([n])))



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
    u_tstart = time.clock()

    eigs = []
    # spinless fermions
    sf = ed_fermions.fermions(gm, 0, t, mus=U, us_same_species=U, potentials=0, nspecies=1)

    # for each interaction, solve number and symmetry subspaces independently
    # TODO: could still speed this up by looping over n and k vals. Then would be able to first use one of the
    # projectors and keep that hamiltonian around. Then apply other projectors to it.
    ham = sf.createH(print_results=1)
    for jj, (n, k, proj) in enumerate(zip(n_list, k_list, projector_list)):
            #ham = sf.createH(print_results=1, projector=proj)
            ham_proj = proj * ham * proj.conj().transpose()
            if ham_proj.shape[0] > 1000:
                pr = 1
            else:
                pr = 0
            eig_vals, eig_vects = sf.diagH(ham_proj, print_results=pr)

            eigs.append(eig_vals)

            #dens_curr, _ = sf.get_exp_vals_sites(eig_vects, sf.n_op, species=0, sites=[0], projector=proj, format="boson")
            #dens.append(dens_curr)
            dens_curr_sector, _ = sf.get_thermal_exp_sites(eig_vects, eig_vals, sf.n_op, 0, temps, projector=proj, format="boson")
            dens_sectors[jj, :, ii, :] = dens_curr_sector.transpose()

            # corr_curr, _, _ = sf.get_corr_sites(eig_vects, 0, 0, sf.n_op, sf.n_op, sites1=[0], sites2=[1], projector=proj, format="boson")
            # corrs.append(corr_curr)
            corrs_curr_sector, _, _ = sf.get_thermal_corr_sites(eig_vects, eig_vals, 0, 0, sf.n_op, sf.n_op,
                                                                          temps, sites1=np.zeros(gm.nsites), sites2=np.arange(0, gm.nsites), projector=proj, format="boson")
            corrs_sectors[jj, :, ii, :] = corrs_curr_sector.transpose()

            # only need to do these once
            if ii == 0 and n == 0:
                ns = np.concatenate((ns, np.ones(eig_vals.size) * n))
                ks = np.concatenate((ks, np.ones(eig_vals.size) * kx))

    dens[..., ii, :] = sf.thermal_avg_combine_sectors(dens_sectors[:, :, ii, :], eigs, temps)
    corrs[..., ii, :] = sf.thermal_avg_combine_sectors(corrs_sectors[:, :, ii, :], eigs, temps)

    u_tend = time.clock()
    print("interaction %d/%d ran in %fs for %d temps" % (ii + 1, ints.size, u_tend - u_tstart, temps.size))

corrs_c = corrs - dens ** 2
sfact = np.fft.fft(corrs_c, axis=-1) / gm.nsites

tend = time.clock()
print("total time was %fs" % (tend - tstart))

# ###############################
# plot results
# ###############################
now = datetime.datetime.now()
now_str = "%04d;%02d;%02d_%02dh_%02dm" % (now.year, now.month, now.day, now.hour, now.minute)

# save results
data = [ints, temps, gm, dens, corrs, corrs_c, sfact]
fname = "%s_spinless_fermion_chain_nsites=%d_pickle.dat" % (now_str, gm.nsites)
with open(fname, 'wb') as f:
        pickle.dump(data, f)

n_distinct_corrs = int(np.ceil(0.5 * gm.nsites)) + 1
nrows = np.floor( np.sqrt(n_distinct_corrs) )
ncols = np.ceil(n_distinct_corrs / float(nrows))

# correlators
figh_corr = plt.figure(figsize=(14,9))

for ii in range(0, n_distinct_corrs):
    plt.subplot(nrows, ncols, ii+1)
    plt.imshow(4 * corrs_c[:, :, ii].transpose(), interpolation=None, cmap='RdGy', vmin=-1, vmax=1, extent=[temps[0], temps[-1], ints[-1], ints[0]], aspect='auto')
    plt.colorbar()
    plt.xlabel('Temperature (t)')
    plt.ylabel('NN U/t')
    plt.title('d = %d' % ii)

plt.suptitle('correlators, spinless fermions at half-filling\n vs. interaction and temperature, nsite = %d chain' % gm.nsites)

fname = "%s_spinless_fermion_chain_nsites=%d_correlators.png" % (now_str, gm.nsites)
figh_corr.savefig(fname)

# structure factor
figh_sfact = plt.figure(figsize=(14,9))

for ii in range(0, n_distinct_corrs):
    plt.subplot(nrows, ncols, ii+1)
    plt.imshow(4 * np.real(sfact[:, :, ii].transpose()), interpolation=None, cmap='RdGy', vmin=-1, vmax=1, extent=[temps[0], temps[-1], ints[-1], ints[0]], aspect='auto')
    plt.xlabel('Temperature (t)')
    plt.ylabel('NN U/t')
    plt.title('k/pi = %0.2f' % (kxs[ii]/np.pi))

plt.suptitle('structure factor, spinless fermions at half-filling\n vs. interaction and temperature, nsite = %d chain' % gm.nsites)

fname = "%s_spinless_fermion_chain_nsites=%d_structure_factor.png" % (now_str, gm.nsites)
figh_sfact.savefig(fname)