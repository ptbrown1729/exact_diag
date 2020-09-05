import numpy as np
import matplotlib.pyplot as plt
import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm
import fermi_gas as fg
import datetime

# TODO: use translation and number to get smaller sectors to solve larger systems faster.

# set up parameters to scan
t = 1
temps = np.linspace(0, 4, 10) * t
betas = np.divide(1, temps)
betas[temps == 0] = np.inf

ints = np.linspace(0, 5, 60) * t
ints = np.concatenate(( -np.flip(ints), ints[1:]))

gm = geom.Geometry.createSquareGeometry(10, 1, 0, 0, bc1_open=False, bc2_open=True)

# translational symmetry
# xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
# xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)

# ###############################
# constant mu
# ###############################
exps = np.zeros((ints.size, gm.nsites, temps.size))
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

# loop over interactions_4
# corrs_all = [] # nstates x nints x nsectors ... do temperatures afterwords
# eigs = [] # nstates
# for ii, U in enumerate(ints):
#
#     # for each interaction, solve number and symmetry subspaces independently
#     for jj in range(0, gm.nsites):
#         sf = fermions.fermions(gm, 0, t, ns=jj, us_same_species=U, potentials=0, nspecies=1)
#         xtransl_op = sf.get_xform_op(xtransl_cycles)
#         xtransl_op = sf.n_projector * xtransl_op * sf.n_projector.conj().transpose()
#         symm_projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx)
#
#         for symm_proj in symm_projs:
#             ham = sf.createH(print_results=1, projector=symm_proj * sf.n_projector)
#             eig_vals, eig_vects = sf.diagH(ham, print_results=1)


# ###############################
# fermi gas
# ###############################
# CURRENTLY THIS HAS SOME ISSUES FOR T=0 IF MU ~ EIGENVALUE. Need to correct this
fg_dens = fg.fg_density(betas, 0, nsites=gm.nsites, dim='1d')
fg_corr = fg.fg_corr(betas, 0, [1, 0], nsites=gm.nsites, dim='1d')

# ###############################
# plot results
# ###############################
now = datetime.datetime.now()
now_str = "%04d;%02d;%02d_%02dh_%02dm" % (now.year, now.month, now.day, now.hour, now.minute)

# correlators
figh_nn = plt.figure()
n_distinct_corrs = int(np.ceil(0.5 * gm.nsites))
nrows = np.floor( np.sqrt(n_distinct_corrs) )
ncols = np.ceil(n_distinct_corrs / float(nrows))

for ii in range(0, n_distinct_corrs):
    plt.subplot(nrows, ncols, ii+1)
    plt.imshow(4 * corrs_allc[:, ii, :], interpolation=None, cmap='RdGy', vmin=-1, vmax=1, extent=[temps[0], temps[-1], ints[-1], ints[0]], aspect='auto')
    plt.colorbar()
    plt.xlabel('Temperature (t)')
    plt.ylabel('NN U/t')
    plt.title('d = %d' % ii)

plt.suptitle('correlators, spinless fermions at half-filling\n vs. interaction and temperature, nsite = %d chain' % gm.nsites)

fname = "%s_spinless_fermion_chain_nsites=%d_correlators.png" % (now_str, gm.nsites)
figh_nn.savefig(fname)

