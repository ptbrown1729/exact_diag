import time
import datetime
import os.path
import numpy as np
import scipy.integrate
import scipy.sparse as sp
import pickle
import matplotlib.pyplot as plt

import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm

save_results = 1
temps = np.linspace(0.1, 10, 100)
today_str = datetime.datetime.today().strftime('%Y-%m-%d_%H;%M;%S')
fname_results = "_hubbard_meas_results" + today_str + ".dat"

nx = 7
ny = 1
gm = geom.Geometry.createSquareGeometry(nx, ny, 0, 0, bc1_open=0, bc2_open=1)
#gm = geom.Geometry.createTiltedSquareGeometry(3, 1, 0, 0, bc1_open=0, bc2_open=0)

U = -6
nup = np.floor(gm.nsites / 2)
ndn = gm.nsites - nup
model = ed_fermions.fermions(gm, U, 1, nup, ndn)

# translational symmetry projectors
xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)
xtransl_op = model.n_projector * model.get_xform_op(xtransl_cycles) * model.n_projector.conj().transpose()
if not model.geometry.lattice.bc2_open:
    ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
    ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, gm)
    ytransl_op = model.n_projector * model.get_xform_op(ytransl_cycles) * model.n_projector.conj().transpose()

# projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx,
#                                                   ytransl_op, max_cycle_len_transly)

# get projectors
if not model.geometry.lattice.bc1_open and not model.geometry.lattice.bc2_open:
    projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                      max_cycle_len_transly)
elif not model.geometry.lattice.bc2_open:
    projs, kys = symm.getZnProjectors(ytransl_op, max_cycle_len_transly)
    kxs = np.zeros(kys.shape)
elif not model.geometry.lattice.bc1_open:
    projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx)
    kys = np.zeros(kxs.shape)
else:
    raise Exception

# for each symmetry sector
# logical_site_up = model.spinful2spinlessIndex(0, model.geometry.nsites, 0)
# logical_site_dn = model.spinful2spinlessIndex(0, model.geometry.nsites, 1)
# logical_site_two_up = model.spinful2spinlessIndex(1, model.geometry.nsites, 0)
# logical_site_two_dn = model.spinful2spinlessIndex(1, model.geometry.nsites, 1)

sz_one_op = model.n_projector * (model.get_single_site_op(0, 0, model.n_op, format="boson")
                                 - model.get_single_site_op(0, 1, model.n_op, format="boson")) * \
            model.n_projector.conj().transpose()

sz_two_op = model.n_projector * (model.get_single_site_op(1, 0, model.n_op, format="boson")
                                 - model.get_single_site_op(1, 1, model.n_op, format="boson")) * \
            model.n_projector.conj().transpose()

szsz_op = sz_one_op.dot(sz_two_op)

nup_one_op = model.n_projector * model.get_single_site_op(0, 0, model.n_op, format="boson") * model.n_projector.conj().transpose()
nup_two_op = model.n_projector * model.get_single_site_op(1, 0, model.n_op, format="boson") * model.n_projector.conj().transpose()
nupnup_op = nup_one_op.dot(nup_two_op)

ndn_one_op = model.n_projector * model.get_single_site_op(0, 1, model.n_op, format="boson") * model.n_projector.conj().transpose()
ndn_two_op = model.n_projector * model.get_single_site_op(1, 1, model.n_op, format="boson") * model.n_projector.conj().transpose()
ndnndn_op = ndn_one_op.dot(ndn_two_op)

nupndn_op = nup_one_op.dot(ndn_two_op)

d_one_op = model.n_projector * (model.get_single_site_op(0, 0, model.n_op, format="boson").
                                dot(model.get_single_site_op(0, 1, model.n_op, format="boson"))) * model.n_projector.conj().transpose()
d_two_op = model.n_projector * (model.get_single_site_op(1, 0, model.n_op, format="boson").
                                dot(model.get_single_site_op(1, 1, model.n_op, format="boson"))) * model.n_projector.conj().transpose()
dd_op = d_one_op.dot(d_two_op)

nsingles_one_op = nup_one_op + ndn_one_op - 2 * d_one_op
nsingles_two_op = nup_two_op + ndn_two_op - 2 * d_two_op
nsns_op = nsingles_one_op.dot(nsingles_two_op)

# quantities
energy_exp_sec = np.zeros((len(projs), len(temps)))
entropy_exp_sec = np.zeros((len(projs), len(temps)))
spheat_exp_sec = np.zeros((len(projs), len(temps)))
# correlators
sz_exp_sector = np.zeros((len(projs), len(temps)))
szsz_exp_sector = np.zeros((len(projs), len(temps)))
nup_exp_sector = np.zeros((len(projs), len(temps)))
nupnup_exp_sector = np.zeros((len(projs), len(temps)))
ndn_exp_sector = np.zeros((len(projs), len(temps)))
ndnndn_exp_sector = np.zeros((len(projs), len(temps)))
nupndn_exp_sector = np.zeros((len(projs), len(temps)))
d_exp_sector = np.zeros((len(projs), len(temps)))
dd_exp_sector = np.zeros((len(projs), len(temps)))
ns_exp_sector =  np.zeros((len(projs), len(temps)))
nsns_exp_sector = np.zeros((len(projs), len(temps)))

eigs_sector = []
for ii, proj in enumerate(projs):
    print("started sector %d/%d" % (ii + 1,len(projs)))
    H = model.createH(projector =proj * model.n_projector, print_results=True)
    eigs, eigvects = model.diagH(H, print_results=True)
    eigs_sector.append(eigs)

    t_start = time.process_time()
    for jj, temp in enumerate(temps):
        energy_exp_sec[ii, jj] = model.get_exp_vals_thermal(eigvects, H, eigs, temp)

        Z_sect = np.sum(np.exp(- eigs / temp))
        entropy_exp_sec[ii, jj] = (np.log(Z_sect) + energy_exp_sec[ii, jj] / temp)

        ham_squared = model.get_exp_vals_thermal(eigvects, H.dot(H), eigs, temp)
        spheat_exp_sec[ii, jj] = 1. / (temp ** 2) * (ham_squared - (energy_exp_sec[ii, jj]) ** 2)


        sz_proj_op = proj * sz_one_op * proj.conj().transpose()
        szsz_proj_op = proj * szsz_op * proj.conj().transpose()
        sz_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, sz_proj_op, eigs, temp)
        szsz_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, szsz_proj_op, eigs, temp)

        nup_proj_op = proj * nup_one_op * proj.conj().transpose()
        nupnup_proj_op = proj * nupnup_op * proj.conj().transpose()
        nup_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, nup_proj_op, eigs, temp)
        nupnup_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, nupnup_proj_op, eigs, temp)

        ndn_proj_op = proj * ndn_one_op * proj.conj().transpose()
        ndnndn_proj_op = proj * ndnndn_op * proj.conj().transpose()
        ndn_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, ndn_proj_op, eigs, temp)
        ndnndn_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, ndnndn_proj_op, eigs, temp)

        nupndn_proj_op = proj * nupndn_op * proj.conj().transpose()
        nupndn_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, nupndn_proj_op, eigs, temp)

        d_proj_op = proj * d_one_op * proj.conj().transpose()
        dd_proj_op = proj * dd_op * proj.conj().transpose()
        d_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, d_proj_op, eigs, temp)
        dd_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, dd_proj_op, eigs, temp)

        ns_proj_op = proj * nsingles_one_op * proj.conj().transpose()
        nsns_proj_op = proj * nsns_op * proj.conj().transpose()
        ns_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, ns_proj_op, eigs, temp)
        nsns_exp_sector[ii, jj] = model.get_exp_vals_thermal(eigvects, nsns_proj_op, eigs, temp)
    t_end = time.process_time()
    print("Computing expectation values for %d temperatures took %0.2f" % (len(temps), t_end - t_start))

energy_exp = np.zeros(len(temps))
entropy_exp = np.zeros(len(temps))
spheat_exp = np.zeros(len(temps))

sz_exp = np.zeros(len(temps))
szsz_exp = np.zeros(len(temps))
szsz_c = np.zeros(len(temps))
nup_exp = np.zeros(len(temps))
nupnup_exp = np.zeros(len(temps))
nupnup_c = np.zeros(len(temps))
ndn_exp = np.zeros(len(temps))
ndnndn_exp = np.zeros(len(temps))
ndnndn_c = np.zeros(len(temps))
nupndn_exp = np.zeros(len(temps))
nupndn_c = np.zeros(len(temps))
d_exp = np.zeros(len(temps))
dd_exp = np.zeros(len(temps))
dd_c = np.zeros(len(temps))
ns_exp = np.zeros(len(temps))
nsns_exp = np.zeros(len(temps))
nsns_c = np.zeros(len(temps))
for jj, temp in enumerate(temps):
    energy_exp[jj] = model.thermal_avg_combine_sectors(energy_exp_sec[:, jj], eigs_sector, temp)
    entropy_exp[jj] = model.thermal_avg_combine_sectors(entropy_exp_sec[:, jj], eigs_sector, temp)
    spheat_exp[jj] = model.thermal_avg_combine_sectors(spheat_exp_sec[:, jj], eigs_sector, temp)

    sz_exp[jj] = model.thermal_avg_combine_sectors(sz_exp_sector[:, jj], eigs_sector, temp)
    szsz_exp[jj] = model.thermal_avg_combine_sectors(szsz_exp_sector[:, jj], eigs_sector, temp)
    szsz_c[jj] = szsz_exp[jj] - sz_exp[jj] ** 2

    nup_exp[jj] = model.thermal_avg_combine_sectors(nup_exp_sector[:, jj], eigs_sector, temp)
    nupnup_exp[jj] = model.thermal_avg_combine_sectors(nupnup_exp_sector[:, jj], eigs_sector, temp)
    nupnup_c[jj] = nupnup_exp[jj] - nup_exp[jj] ** 2

    ndn_exp[jj] = model.thermal_avg_combine_sectors(ndn_exp_sector[:, jj], eigs_sector, temp)
    ndnndn_exp[jj] = model.thermal_avg_combine_sectors(ndnndn_exp_sector[:, jj], eigs_sector, temp)
    ndnndn_c[jj] = ndnndn_exp[jj] - ndn_exp[jj] ** 2

    nupndn_exp[jj] = model.thermal_avg_combine_sectors(nupndn_exp_sector[:, jj], eigs_sector, temp)
    # assuming translational invariance.
    nupndn_c[jj] = nupndn_exp[jj] - nup_exp[jj] * ndn_exp[jj]

    d_exp[jj] = model.thermal_avg_combine_sectors(d_exp_sector[:, jj], eigs_sector, temp)
    dd_exp[jj] = model.thermal_avg_combine_sectors(dd_exp_sector[:, jj], eigs_sector, temp)
    dd_c[jj] = dd_exp[jj] - d_exp[jj] ** 2

    ns_exp[jj] = model.thermal_avg_combine_sectors(ns_exp_sector[:, jj], eigs_sector, temp)
    nsns_exp[jj] = model.thermal_avg_combine_sectors(nsns_exp_sector[:, jj], eigs_sector, temp)
    nsns_c[jj] = nsns_exp[jj] - ns_exp[jj] ** 2

# print "period_start/t = %0.2f" % temp
# print "4*<Sz(0) Sz(1)>_c = %0.3f" % szsz_c
# print "<nup(0)nup(1)>_c = %0.3f" % nupnup_c
# print "<ndn(0)ndn(1)>_c = %0.3f" % ndnndn_c
# print "<nup(0)ndn(1)>_c = %0.3f" % nupndn_c
# print "<ns(0)ns(1)>_c = %0.3f" % nsns_c
# print "<d(0) d(1)>_c = %0.3f" % dd_c

fig_handle_quantities = plt.figure()
nrows = 2
ncols = 3

plt.subplot(nrows, ncols, 1)
plt.plot(temps, energy_exp / model.geometry.nsites)
plt.grid()
plt.xlabel('Temp (t)')
plt.ylabel('Energy / site (t)')

plt.subplot(nrows, ncols, 2)
plt.plot(temps, entropy_exp / model.geometry.nsites)
plt.grid()
plt.xlabel('Temp (t)')
plt.ylabel('Entropy / site')

plt.subplot(nrows, ncols, 3)
plt.plot(temps, spheat_exp / model.geometry.nsites)
plt.grid()
plt.xlabel('Temp (t)')
plt.ylabel('Sp Heat / site')

plt.subplot(nrows, ncols, 4)
plt.plot(temps, ns_exp)
plt.grid()
plt.ylim([0, 1])
plt.xlabel('Temp (t)')
plt.ylabel('Singles density')

plt.subplot(nrows, ncols, 5)
plt.plot(temps, d_exp)
plt.grid()
plt.ylim([0, 1])
plt.xlabel('Temp (t)')
plt.ylabel('doubles density')

nrows = 2
ncols = 3

fig_handle_corrs = plt.figure()

plt.subplot(nrows, ncols, 1)
plt.plot(temps, szsz_c)
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.ylabel('<szsz>_c')

plt.subplot(nrows, ncols, 2)
plt.plot(temps, nupnup_c)
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.ylabel('<nup nup>_c')

plt.subplot(nrows, ncols, 3)
plt.plot(temps, ndnndn_c)
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.ylabel('<ndn ndn>_c')

plt.subplot(nrows, ncols, 4)
plt.plot(temps, nupndn_c)
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.ylabel('<nup ndn>_c')

plt.subplot(nrows, ncols, 5)
plt.plot(temps, nsns_c)
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.ylabel('<ns ns>_c')

plt.subplot(nrows, ncols, 6)
plt.plot(temps, dd_c)
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.ylabel('<d d>_c')

data = [model, temps, energy_exp, entropy_exp, spheat_exp,
        sz_exp, szsz_c, nup_exp, nupnup_c, ndn_exp, ndnndn_c, nupndn_c,
        ns_exp, nsns_c, d_exp, dd_c]

if save_results:
    with open(fname_results, 'wb') as f:
        pickle.dump(data, f)

    fig_name = "attractive_hubb_ed_energy" + today_str + ".png"
    fig_handle_quantities.savefig(fig_name)

    fig_name = "attractive_hubb_ed_corr" + today_str + ".png"
    fig_handle_corrs.savefig(fig_name)

plt.show()