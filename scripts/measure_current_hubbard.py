import time
import datetime
import pickle
import os.path

import numpy as np
import scipy.integrate
import scipy.sparse as sp
import ed_fermions
import ed_geometry as geom
import ed_symmetry as symm

# parameters
save_results = 0
plot_results = 1
print_all = 1
t_script_start = time.process_time()
print_all = 1

# define geometry
bc1_open = 0
bc2_open = 1
phi1 = 0
phi2 = 0

nx = 6
ny = 1
gm = geom.Geometry.createSquareGeometry(nx, ny, phi1, phi2, bc1_open, bc2_open)
# nr = 3
# nv = 1
# gm = geom.Geometry.createTiltedSquareGeometry(nr, nv, phi1, phi2, bc1_open, bc2_open)

# define fermions parameters
U = 6
tx = 1
ty = tx
# n_ups = np.ceil(gm.nsites / 2)
# n_dns = gm.nsites - n_ups
n_ups = 3.0
n_dns = 3.0
# thermal averaging
temps = np.array([0, 0.1, 1, 10])

h = ed_fermions.fermions(gm, U, tx, ty, n_ups, n_dns)

# build translational symmetries
# x-translation
xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)
xtransl_op = h.n_projector * h.get_xform_op(xtransl_cycles) * h.n_projector.conj().transpose()

# y-translations
if not bc2_open:
    ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
    ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, gm)
    ytransl_op = h.n_projector * h.get_xform_op(ytransl_cycles) * h.n_projector.conj().transpose()

# get projectors
if not bc1_open and not bc2_open:
    projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op, max_cycle_len_transly, print_all)
elif not bc2_open:
    projs, kys = symm.getZnProjectors(ytransl_op, max_cycle_len_transly, print_all)
    kxs = np.zeros(kys.shape)
elif not bc1_open:
    projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx, print_all)
    kys = np.zeros(kxs.shape)
else:
    projs = [sp.identity(h.geometry.nsites)]

# produce operators and structures for storing operators
current_op_x = h.n_projector * h.get_current_op(np.array([[1], [0]])) * h.n_projector.conj().transpose()
eigvals_sectors = []
kx_ops_sectors = []
current_ops_x_sector = []
# creation op, site 0
# creation_op_up = h.n_projector * h.getSingleSiteOp(0, h.geometry.nsites * h.nspecies, h.cdag_op, format="fermion") * h.n_projector.conj().transpose()
# annihilation_op_up = h.n_projector * h.getSingleSiteOp(0, h.geometry.nsites * h.nspecies, h.c_op, format="fermion") * h.n_projector.conj().transpose()
creation_op_up = h.n_projector * h.get_single_site_op(0, 0, h.cdag_op, format="fermion") * h.n_projector.conj().transpose()
annihilation_op_up = h.n_projector * h.get_single_site_op(0, 0, h.c_op, format="fermion") * h.n_projector.conj().transpose()

# temps = np.array([0, 4, 6, 8, 10])
omegas = np.linspace(-2*U, 2*U, 500)
# etas = np.array([0.025, 0.05, 0.1, 0.3])
etas = np.array([0.15])
kxe_sectors_temps = np.zeros([len(projs), len(temps)])
opt_cond_integral_sectors_temps = np.zeros([len(projs), len(temps)])
opt_cond_sectors_temps = np.zeros([len(projs), len(temps), len(omegas), len(etas)])
# trying to calculate spectral function in this way does not work, because we have imposed number symmetry
#spectral_fn_sectors_temps = np.zeros([len(projs), len(temps), len(omegas), len(etas)])

# diagonalize all symmetry sectors
for ii,proj in enumerate(projs):
    print "sector %d/%d" % (ii + 1, len(projs))
    H = h.createH(projector = proj * h.n_projector, print_results=print_all)
    eigs, eigvects = h.diagH(H, print_results=print_all)

    eigvals_sectors.append(eigs)

    # kinetic energy operator commutes with translation operators
    # TODO: revise this, as get_hopping_mat signature has changed
    #kx_op_sector = h.get_kinetic_op(h.get_hopping_mat(gm, tunn, 0), projector = symm_proj * h.n_projector)
    kx_op_sector = h.get_kinetic_op(projector = proj * h.n_projector, direction_vect=np.array([1, 0]))
    kx_ops_sectors.append(kx_op_sector)

    # current operator commutes with translation operators
    jx_sector = proj * current_op_x * proj.conj().transpose()
    current_ops_x_sector.append(jx_sector)
    jx_matrix_elems = h.get_matrix_elems(eigvects, jx_sector, print_results=True)

    # also get creation/annihilation operators
    # creation_op_upk = symm_proj * creation_op_up * symm_proj.conj().transpose()
    # creation_op_upk_matrixel = h.get_matrix_elems(eigvects, creation_op_upk, print_results=1)
    # annihilation_op_upk = symm_proj * annihilation_op_up * symm_proj.conj().transpose()
    # annihilation_op_upk_matrixel = h.get_matrix_elems(eigvects, annihilation_op_upk, print_results=1)

    # loop over temperatures, get kinetic energy and optical conductivity
    for jj, temp in enumerate(temps):
        tstart = time.process_time()
        kxe_sectors_temps[ii, jj] = h.get_exp_vals_thermal(eigvects, kx_op_sector, eigs, temp, print_results=False)
        opt_cond_integral_sectors_temps[ii, jj] = h.integrate_conductivity(jx_matrix_elems, eigs, temp, print_results=False)
        opt_cond_fn = h.get_optical_cond_fn(jx_matrix_elems, eigs, temp, print_results=False)

        # loop over omegas for optical conductivity
        for kk in range(0, len(omegas)):
            # loop over broadening parameters
            for aa, eta in enumerate(etas):
                print "eta %d/%d" % (aa + 1, len(etas))
                opt_cond_sectors_temps[ii, jj, kk, aa] = opt_cond_fn(omegas[kk], eta)
        tend = time.process_time()
        print "temperature %d/%d in sector %d took %0.2f s" % (jj + 1, len(temps), ii, tend-tstart)

kxe_temps = np.zeros(len(temps))
opt_cond_integrals = np.zeros(len(temps))
drude_weights = np.zeros(len(temps))
opt_cond = np.zeros([len(temps), len(omegas), len(etas)])
for ii, temp in enumerate(temps):
    kxe_temps[ii] = h.thermal_avg_combine_sectors(kxe_sectors_temps[:, ii], eigvals_sectors, temp)
    opt_cond_integrals[ii] = h.thermal_avg_combine_sectors(opt_cond_integral_sectors_temps[:, ii], eigvals_sectors, temp)
    opt_cond[ii, :, :] = h.thermal_avg_combine_sectors(opt_cond_sectors_temps[:, ii, :, :], eigvals_sectors, temp)

drude_weights = -np.pi * kxe_temps / gm.nsites - opt_cond_integrals

# test optical conductivity by numerical integrating. Does it match the more direct integration technique that does not
# require broadening?
integrated_oc = np.zeros([len(temps), len(etas)])
for ii in range(0, len(temps)):
    for jj in range(0, len(etas)):
        interp_fn = lambda w: np.interp(w, omegas, opt_cond[ii, :, jj])
        integrated_oc[ii, jj], tol = scipy.integrate.quad(interp_fn, omegas.min(), omegas.max())

# full solution for testing small systems
# full problem
# h_full = h.createH(projector = h.n_projector)
# eig_vals, eig_vects = h.diagH(h_full)
# jx_mat_elem_full = h.get_matrix_elems(eig_vects, current_op_x)
# kx_op_full = h.get_kinetic_op(h.get_hopping_mat(gm, tunn, 0), projector = h.n_projector)
# z = np.sum(np.exp(-eig_vals))

t_script_end = time.process_time()
print "runtime = %0.2f s" % (t_script_end - t_script_start)

# save results
h.runtime = t_script_end - t_script_start
h.temps = temps
h.etas = etas
h.omegas = omegas

h.kxe_temps = kxe_temps
h.opt_cond_ints = opt_cond_integrals
h.drude_weights = drude_weights

h.eigvals_sectors = eigvals_sectors
h.opt_cond = opt_cond

if save_results:
    now = datetime.datetime.now()
    #fpath = "Z://Data analysis//PeterB//hubbard_cond_ed_data"
    fpath = ""
    fname = "%04d_%02d_%02d_%02d_%02d_nsites=%d_U=%0.1f_nup=%d_ndn=%d_phi1=%0.3f_phi2=%0.3f" % (now.year, now.month, now.day, now.hour, now.minute, gm.nsites, U, n_ups, n_dns, gm.phase1, gm.phase2)
    h.save(os.path.join(fpath, fname + '.pkl'))
    h.save(os.path.join(fpath, fname + '.mat'))
    print "saved results to %s" % os.path.join(fpath, fname + '.pkl')

if plot_results:
    import matplotlib.pyplot as plt

    today_str = datetime.datetime.today().strftime('%Y-%m-%d_%H;%M;%S')

    # save data results
    if not os.path.isdir("data"):
        os.mkdir("data")

    data = [t_script_end - t_script_start, temps, etas, omegas, kxe_temps, opt_cond_integrals, integrated_oc,
            drude_weights, eigvals_sectors, opt_cond]
    fname = os.path.join("data", "measure_current_" + today_str + "_pickle.dat")
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    # plot conductivity for different parameters
    for ii in range(0, len(temps)):
        for jj in range(0, len(etas)):
            fig_handle = plt.figure()
            plt.plot(omegas, opt_cond[ii, :, jj])
            plt.grid()
            plt.xlabel('omega_start (t)')
            plt.ylabel('Re{sigma}(w)')
            plt.title('nsites = %d, nup = %d, ndn = %d, U = %0.2f, period_start = %0.2f, eta = %0.2f' % (
            gm.nsites, n_ups, n_dns, U, temps[ii], etas[jj]))

            # save figure
            fname_fig = os.path.join("data", "measure_current_" + today_str +
                                     "_nsites=%d_nup=%d_ndn=%d_U=%0.2f_T=%0.2f_eta=%0.2f.png" %
                                     (gm.nsites, n_ups, n_dns, U, temps[ii], etas[jj]))
            fig_handle.savefig(fname_fig)

            plt.draw()


    leg = ['sum rule', 'integral conductivity']
    fig_handle = plt.figure()
    plt.plot(temps, np.pi * -kxe_temps / gm.nsites, 'bo')
    plt.plot(temps, opt_cond_integrals, 'ro')
    for ii in range(0, len(etas)):
        plt.plot(temps, integrated_oc[:, ii], 'o')
        leg.append('manual integration, eta = %0.1f' % etas[ii])

    plt.grid()
    plt.xlabel('period_start (t)')
    plt.legend(tuple(leg))
    plt.title('nsites = %d, nups = %d, ndns = %d, U = %0.2f, bc1 open = %d, bc2 open = %d' % (
    gm.nsites, n_ups, n_dns, U, bc1_open, bc2_open))

    # save figure
    fname_fig = os.path.join("data", "measure_current_" + today_str + "_figure.png")
    fig_handle.savefig(fname_fig)

    plt.draw()
    plt.show()


