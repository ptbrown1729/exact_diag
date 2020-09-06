"""
Diagonalize Hubbard system and record useful quantities, such as density expectation values,
correlators, etc versus temperature.
"""

import time
import datetime
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm
import fermi_gas as fg

# ############################
# settings
# ############################
save_results = True
save_dir = "../data"

nx = 4
ny = 3
U = 8
mu_up = U/2
mu_dn = U/2
temps = np.linspace(0.1, 3, 10)

today_str = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
identifier_str = "hubbard_nx=%d_ny=%d_u=%0.2f" % (nx, ny, U)
fname_results = os.path.join(save_dir, "%s_%s.pkl" % (today_str, identifier_str))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ############################
# geometry
# ############################
tstart = time.process_time()

bc1_open = False
bc2_open = False
gm = geom.Geometry.createSquareGeometry(nx, ny, 0, 0, bc1_open=bc1_open, bc2_open=bc2_open)
# gm = geom.Geometry.createTiltedSquareGeometry(3, 1, 0, 0, bc1_open=False, bc2_open=False)

model = ed_fermions.fermions(gm, us_interspecies=U, ts=1, mus=(mu_up, mu_dn), nspecies=2)

# get nup projectors
nup_op = model.get_sum_op(model.n_op, 0, format="boson")
nup_projs, nups = model.get_subspace_projs(nup_op, print_results=False)

# get ndn op
ndn_op = model.get_sum_op(model.n_op, 1, format="boson")

# get translation operators
# translational symmetry projectors
if not bc1_open:
    xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
    xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)
    xtransl_op = model.n_projector * model.get_xform_op(xtransl_cycles) * model.n_projector.conj().transpose()

if not bc2_open:
    ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
    ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, gm)
    ytransl_op = model.n_projector * model.get_xform_op(ytransl_cycles) * model.n_projector.conj().transpose()

# get all projectors
projector_list = []
nups_list = np.array([])
ndns_list = np.array([])
kxs_list = np.array([])
kys_list = np.array([])
for nup, nup_proj in zip(nups, nup_projs):
    ndn_sector_op = nup_proj * ndn_op * nup_proj.conj().transpose()
    ndn_projs, ndns = model.get_subspace_projs(ndn_sector_op)

    for ndn, ndn_proj in zip(ndns, ndn_projs):
        n_proj = ndn_proj * nup_proj

        tx_sector = n_proj * xtransl_op * n_proj.conj().transpose()
        ty_sector = n_proj * ytransl_op * n_proj.conj().transpose()

        # get projectors
        if not bc1_open and not bc2_open:
            projs, kxs, kys = symm.get2DTranslationProjectors(tx_sector, max_cycle_len_translx,
                                                              ty_sector, max_cycle_len_transly)
        elif bc1_open and not bc2_open:
            projs, kys = symm.getZnProjectors(ty_sector, max_cycle_len_transly)
            kxs = np.zeros(kys.shape)
        elif not bc1_open and bc2_open:
            projs, kxs = symm.getZnProjectors(tx_sector, max_cycle_len_translx)
            kys = np.zeros(kxs.shape)
        else:
            projs = [1]
            kxs = [0]
            kys = [0]

        for kx, ky, symm_proj in zip(kxs, kys, projs):
            curr_proj = symm_proj * n_proj
            if curr_proj.size > 0:
                projector_list.append(curr_proj)
                kxs_list = np.concatenate((kxs_list, np.array([kx])))
                kys_list = np.concatenate((kys_list, np.array([ky])))
                nups_list = np.concatenate((nups_list, np.array([nup])))
                ndns_list = np.concatenate((ndns_list, np.array([ndn])))

# ############################
# build operators to measure
# ############################

ops = {"nup1": model.get_single_site_op(0, 0, model.n_op, format="boson"),
       "nup2": model.get_single_site_op(1, 0, model.n_op, format="boson"),
       "ndn1": model.get_single_site_op(0, 1, model.n_op, format="boson"),
       "ndn2": model.get_single_site_op(1, 1, model.n_op, format="boson"),
       }

ops.update({"nup_nup": ops["nup1"].dot(ops["nup2"]),
            "ndn_ndn": ops["ndn1"].dot(ops["ndn2"]),
            "nup_ndn": ops["nup1"].dot(ops["ndn2"]),
            "d1": ops["nup1"].dot(ops["ndn1"]),
            "d2": ops["nup2"].dot(ops["ndn2"]),
            "sz1": ops["nup1"] - ops['ndn1'],
            "sz2": ops["nup2"] - ops['ndn2'],
            })

ops.update({"ns1": ops["nup1"] + ops["ndn1"] - 2 * ops["d1"],
            "ns2": ops["nup2"] + ops["ndn2"] - 2 * ops["d2"]})

ops.update({"sz_sz": ops["sz1"].dot(ops["sz2"]),
            "dd": ops["d1"].dot(ops["d2"]),
            "ns_ns": ops["ns1"].dot(ops["ns2"]),
            })


# ############################
# expectation values for each symmetry/number sector
# ############################
nprojs = len(projector_list)
nt = len(temps)
nu = 1

exp_vals_sectors = {"energy": [np.zeros(nt) for _ in projector_list],
            "entropy": [np.zeros(nt) for _ in projector_list],
            "specific_heat": [np.zeros(nt) for _ in projector_list]}

for k in ops.keys():
    exp_vals_sectors.update({k: [np.zeros(nt) for _ in projector_list]})

eigs_sector = []
H = model.createH(print_results=True)
for ii, proj in enumerate(projector_list):
    ndim = proj.shape[0]
    print("started sector %d/%d of size %dx%d" % (ii + 1, len(projector_list), ndim, ndim))
    h_sector = proj * H * proj.conj().transpose()
    eigs, eigvects = model.diagH(h_sector)
    eigs_sector.append(eigs)

    t_start = time.process_time()
    for jj, temp in enumerate(temps):
        exp_vals_sectors["energy"][ii][jj] = model.get_exp_vals_thermal(eigvects, h_sector, eigs, temp)

        partition_fn_sector = np.sum(np.exp(-eigs / temp))
        exp_vals_sectors["entropy"][ii][jj] = \
            (np.log(partition_fn_sector) + exp_vals_sectors["energy"][ii][jj] / temp)

        ham_squared = model.get_exp_vals_thermal(eigvects, h_sector.dot(h_sector), eigs, temp)
        exp_vals_sectors["specific_heat"][ii][jj] = \
            1. / (temp ** 2) * (ham_squared - exp_vals_sectors["energy"][ii][jj] ** 2)

        for k, op in ops.items():
            op_sector = proj * op * proj.conj().transpose()
            exp_vals_sectors[k][ii][jj] = model.get_exp_vals_thermal(eigvects, op_sector, eigs, temp)

    t_end = time.process_time()
    print("Computing expectation values for %d temperatures took %0.2f" % (len(temps), t_end - t_start))

# final experimental values
exp_vals = {}
for k in exp_vals_sectors.keys():
    exp_vals.update({k: np.zeros(nt)})

for jj, temp in enumerate(temps):
    for k, evs in exp_vals_sectors.items():
        ev_at_temp = np.array([e[jj] for e in evs])
        exp_vals[k][jj] = model.thermal_avg_combine_sectors(ev_at_temp, eigs_sector, temp)

# non-interacting Fermi gas correlators for comparison
mu_up_fg = 0
mu_dn_fg = 0
exp_vals_fg = {
        "ns": fg.fg_singles(1 / temps, mu_up_fg, mu_dn_fg, dim="2d"),
        "d": fg.fg_doubles(1 / temps, mu_up_fg, mu_dn_fg, dim="2d"),
        "nup_nup": fg.fg_corr(1 / temps, mu_up_fg, corr_index=(1, 0), dim="2d"),
        "ndn_ndn": fg.fg_corr(1 / temps, mu_dn_fg, corr_index=(1, 0), dim="2d"),
        "sz_sz": fg.fg_sz_corr(1 / temps, mu_up_fg, mu_dn_fg, corr_index=(0, 1), dim="2d"),
        "dd": fg.fg_doubles_corr(1 / temps, mu_up_fg, mu_dn_fg, corr_index=(0, 1), dim="2d"),
        "ns_ns": fg.fg_singles_corr(1 / temps, mu_up_fg, mu_dn_fg, corr_index=(0, 1), dim="2d"),
        }


# store output data in dictionary
tend = time.process_time()

exp_vals.update({"model": model, "temps": temps, "U": U, "mu_up": mu_up, "mu_dn": mu_dn,
                 "exp_vals_non_int_fg": exp_vals_fg,
                 "run_time_secs": (tend - tstart)})

if save_results:
    with open(fname_results, "wb") as f:
        pickle.dump(exp_vals, f)

# ############################
# plot densities and energy
# ############################
fig_handle_quantities = plt.figure(figsize=(10, 8))
nrows = 2
ncols = 3

plt.subplot(nrows, ncols, 1)
plt.plot(temps, exp_vals["energy"] / model.geometry.nsites, 'b.-')
plt.grid()
plt.xlabel('Temp (t)')
plt.title('Energy / site (t)')

plt.subplot(nrows, ncols, 2)
plt.plot(temps, exp_vals["entropy"] / model.geometry.nsites, 'b.-')
plt.grid()
plt.xlabel('Temp (t)')
plt.title('Entropy / site')

plt.subplot(nrows, ncols, 3)
plt.plot(temps, exp_vals["specific_heat"] / model.geometry.nsites, 'b.-')
plt.grid()
plt.xlabel('Temp (t)')
plt.title('Sp Heat / site')

plt.subplot(nrows, ncols, 4)
plt.plot(temps, exp_vals["nup1"], 'b.-')
plt.plot(temps, exp_vals["ndn1"], 'k.-')
plt.grid()
plt.ylim([-0.05, 1.1])
plt.legend(['nup', 'ndn'])
plt.xlabel('Temp (t)')
plt.title('Densities')

plt.subplot(nrows, ncols, 5)
plt.plot(temps, exp_vals["ns1"], 'b.-')
plt.plot(temps, exp_vals["d1"], 'k.-')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["ns"], 'b--')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["d"], 'k--')
plt.grid()
plt.ylim([-0.05, 1.1])
plt.legend(['ns', 'nd'])
plt.xlabel('Temp (t)')
plt.title('Densities')

if save_results:
    fig_name = os.path.join(save_dir, "%s_%s_density_plot.png" % (today_str, identifier_str))
    fig_handle_quantities.savefig(fig_name)

# ############################
# plot correlators
# ############################
nrows = 2
ncols = 3

fig_handle_corrs = plt.figure(figsize=(10, 8))

plt.subplot(nrows, ncols, 1)
plt.plot(temps, exp_vals["sz_sz"] - exp_vals["sz1"] * exp_vals["sz2"], 'b.-')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["sz_sz"], 'b--')
plt.grid()
plt.ylim([-0.3, 0.1])
plt.xlabel('Temp (t)')
plt.title('<szsz>_c')

plt.subplot(nrows, ncols, 2)
plt.plot(temps, exp_vals["nup_nup"] - exp_vals["nup1"] * exp_vals["nup2"], 'b.-')
plt.plot(temps, exp_vals["ndn_ndn"] - exp_vals["ndn1"] * exp_vals["ndn2"], 'k.-')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["nup_nup"], 'b--')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["ndn_ndn"], 'k--')
plt.grid()
plt.legend(['nup', 'ndn'])
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.title('<n_sigma n_sigma>_c')


plt.subplot(nrows, ncols, 3)
plt.plot(temps, exp_vals["nup_ndn"] - exp_vals["nup1"] * exp_vals["ndn1"], 'b.-')
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.title('<nup ndn>_c')

plt.subplot(nrows, ncols, 4)
plt.plot(temps, exp_vals["ns_ns"] - exp_vals["ns1"] * exp_vals["ns2"], 'b.-')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["ns_ns"], 'b--')
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.title('<ns ns>_c')

plt.subplot(nrows, ncols, 5)
plt.plot(temps, exp_vals["dd"] - exp_vals["d1"] * exp_vals["d2"], 'b.-')
plt.plot(temps, exp_vals["exp_vals_non_int_fg"]["dd"], 'b--')
plt.grid()
plt.ylim([-0.1, 0.1])
plt.xlabel('Temp (t)')
plt.title('<d d>_c')

if save_results:
    fig_name = os.path.join(save_dir, "%s_%s_corr_plot.png" % (today_str, identifier_str))
    fig_handle_corrs.savefig(fig_name)

plt.show()
