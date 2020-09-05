"""
Spinless Fermions with nearest-neighbor interactions. Not using symmetries.
"""

import datetime
import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm
import fermi_gas as fg


# TODO: use translation and number to get smaller sectors to solve larger systems faster.

now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
nsites = 12

save_dir = "../data"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

tstart = time.process_time()

# set up parameters to scan
t = 1
# temps = np.linspace(0, 3, 10) * t
temps = np.linspace(0, 3, 2) * t
betas = np.divide(1, temps)
betas[temps == 0] = np.inf

# ints = np.linspace(0, 5, 60) * t
ints = np.linspace(0, 5, 5) * t
ints = np.concatenate((-np.flip(ints), ints[1:]))

gm = geom.Geometry.createSquareGeometry(nsites, 1, 0, 0, bc1_open=False, bc2_open=True)

# translational symmetry
# xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
# xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)

# ###############################
# constant mu
# ###############################
exps = np.zeros((ints.size, gm.nsites, temps.size))
# dens = np.zeros((ints.size, gm.nsites, temps.size))
corrs_all = np.zeros((ints.size, gm.nsites, temps.size))
corrs_allc = np.zeros((ints.size, gm.nsites, temps.size))
# Using particle-hole symmetry can show half-filling (one particle per spin per every other site) happens at mu = U
for ii, U in enumerate(ints):
    sf = ed_fermions.fermions(gm, 0, t, mus=U, us_same_species=U, potentials=0, nspecies=1)
    ham = sf.createH(print_results=True)
    eig_vals, eig_vects = sf.diagH(ham, print_results=True)

    # get single-site expectation values for ground state
    exps[ii, :, :], _ = sf.get_thermal_exp_sites(eig_vects, eig_vals, sf.n_op, 0, temps, format="boson")
    # get correlations
    corrs, _, _ = sf.get_thermal_corr_sites(eig_vects, eig_vals, 0, 0, sf.n_op, sf.n_op, temps, sites1=np.zeros(gm.nsites),
                                                      sites2=np.arange(0, gm.nsites, 1), format="boson")

    corrs_all[ii, :, :] = corrs

    # connected correlators
    for jj in range(0, gm.nsites):
        corrs_allc[ii, jj, :] = corrs_all[ii, jj, :] - exps[ii, 0, :] * exps[ii, jj, :]
    print("%d/%d" % (ii + 1, ints.size))

corrs_nnc = corrs_allc[:, 1, :]
corrs_nnnc = corrs_allc[:, 2, :]

data = {"ints": ints, "temps": temps, "gm": gm, "corrs": corrs, "corrs_c": corrs_allc, "datetime": now_str}
fname = os.path.join(save_dir, "%s_spinless_fermion_chain_nsites=%d_pickle.dat" % (now_str, gm.nsites))
with open(fname, 'wb') as f:
    pickle.dump(data, f)

tend = time.process_time()
print("total time was %fs" % (tend - tstart))

# ###############################
# fermi gas
# ###############################
# CURRENTLY THIS HAS SOME ISSUES FOR T=0 IF MU ~ EIGENVALUE. Need to correct this
fg_dens = fg.fg_density(betas, 0, nsites=gm.nsites, dim='1d')
fg_corr = fg.fg_corr(betas, 0, [1, 0], nsites=gm.nsites, dim='1d')

# ###############################
# plot results
# ###############################

dt = temps[1] - temps[0]
du = ints[1] - ints[0]
extent = [temps[0] - 0.5 * dt, temps[-1] + 0.5 * dt, ints[-1] + 0.5 * du, ints[0] - 0.5 * du]

# correlators
figh_nn = plt.figure()

n_distinct_corrs = int(np.ceil(0.5 * gm.nsites))
nrows = int(np.floor(np.sqrt(n_distinct_corrs)))
ncols = int(np.ceil(n_distinct_corrs / float(nrows)))

for ii in range(0, n_distinct_corrs):
    plt.subplot(nrows, ncols, ii+1)
    plt.imshow(4 * corrs_allc[:, ii, :], interpolation=None, cmap='RdGy', vmin=-1, vmax=1,
               extent=extent, aspect='auto')
    plt.colorbar()
    if ii == 0:
        plt.xlabel('Temperature (t)')
        plt.ylabel('NN U/t')
    plt.title('d = %d' % ii)

plt.suptitle('correlators, spinless fermions at half-filling\n vs. interaction and temperature, nsite = %d chain' % gm.nsites)

fname = os.path.join(save_dir, "%s_spinless_fermion_chain_nsites=%d_correlators.png" % (now_str, gm.nsites))
figh_nn.savefig(fname)

