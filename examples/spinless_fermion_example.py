"""
Simple example of exact digaonalization of a spinless fermion system
"""

import numpy as np
import matplotlib.pyplot as plt
import ed_fermions
import ed_geometry as geom
import fermi_gas as fg

# set up system
t = 1
temps = np.array([0, 0.5, 1, 2, 10]) * t
betas = np.divide(1, temps)
betas[temps == 0] = np.inf
mu = 0

# ###############################
# constant number
# ###############################
gm = geom.Geometry.createSquareGeometry(8, 1, 0, 0, bc1_open=False, bc2_open=True)
sf = ed_fermions.fermions(gm, 0, t, ns=np.array([4]), us_same_species=0, potentials=0, nspecies=1)

# diagonalize hamiltonian
ham = sf.createH(projector=sf.n_projector, print_results=True)
eig_vals, eig_vects = sf.diagH(ham, print_results=True)

# get single-site expectation values for ground state
exps, _ = sf.get_thermal_exp_sites(eig_vects, eig_vals, sf.n_op, 0, temps, sites=[0], projector=sf.n_projector, format="boson")
corrs, _, _ = sf.get_thermal_corr_sites(eig_vects, eig_vals, 0, 0, sf.n_op, sf.n_op, temps, projector=sf.n_projector, sites1=np.array([0]),
                                                  sites2=np.array([1]), format="boson")
corrs_c = corrs - exps[:, 0] * exps[:, 1]


# ###############################
# constant mu
# ###############################
sf_2 = ed_fermions.fermions(gm, 0, t, mus=mu, us_same_species=0, potentials=0, nspecies=1)
ham2 = sf_2.createH(print_results=True)
eig_vals2, eig_vects2 = sf_2.diagH(ham2, print_results=True)

# get single-site expectation values for ground state
exps2, _ = sf_2.get_thermal_exp_sites(eig_vects2, eig_vals2, sf_2.n_op, 0, temps, sites=[0], format="boson")
corrs2, _, _ = sf_2.get_thermal_corr_sites(eig_vects2, eig_vals2, 0, 0, sf_2.n_op, sf_2.n_op, temps, sites1=np.array([0]),
                                                  sites2=np.array([1]), format="boson")
corrs2_c = corrs2 - exps2[:, 0] * exps2[:, 1]


# ###############################
# fermi gas
# ###############################
# CURRENTLY THIS HAS SOME ISSUES FOR T=0 IF MU ~ EIGENVALUE. Need to correct this
fg_dens = fg.fg_density(betas, mu, nsites=gm.nsites, dim='1d')
fg_corr = fg.fg_corr(betas, mu, [1, 0], nsites=gm.nsites, dim='1d')