import numpy as np
import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm
import matplotlib.pyplot as plt

nsites = 16
nfermions = 8
interaction = 1.
F = -4.
alpha = 0.5
t = -1
potentials = F * (np.arange(0, nsites) - 0.5 * (nsites - 1) ) + alpha * np.arange(0, nsites) ** 2 / nsites ** 2

gm = geom.Geometry.createSquareGeometry(nsites, t, 0, 0, bc1_open=False, bc2_open=True)
sf = ed_fermions.fermions(gm, 0, 1, ns=nfermions, mus=0, us_same_species=interaction, potentials=potentials, nspecies=1)
ham = sf.createH(print_results=True, projector=sf.n_projector)

# diagonalize
eig_vals, eig_vects = sf.diagH(ham, print_results=True)
# get level statistics
diffs = eig_vals[1:] - eig_vals[0:-1]
gap_ratios = np.divide(diffs[1:], diffs[:-1])
rn = np.min( np.concatenate((gap_ratios[:, None],  np.divide(1, gap_ratios[:, None])),axis=1), axis=1)
rn[np.isnan(rn)] = 0

r_mean = np.mean(rn)
num_in_bin, bin_edges = np.histogram(rn, bins=30)
pr = num_in_bin / float(rn.size)
bin_mean = 0.5 * (bin_edges[1:] + bin_edges[:-1])

plt.figure()
plt.plot(bin_mean, pr)
plt.xlabel('bin mean energy')
plt.ylabel('P(r)')

#
gstate = eig_vects[:, 0]

n_exp, sites = sf.get_exp_vals_sites(eig_vects[:, 0:9], sf.n_op, 0, projector=sf.n_projector, format="boson")
ns = np.sum(n_exp, 0)


# plot density of eigenstates
for ii in range(0, n_exp.shape[1]):
    plt.subplot(1, n_exp.shape[1], ii)
    plt.plot(sites, n_exp[:, ii])
