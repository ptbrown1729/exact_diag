"""
Test twisted boundary conditions with an Ising system
"""

import numpy as np
import ed_spins
import ed_geometry as geom
import ed_symmetry as symm


n_phases = 4

phis = np.linspace(0, 2*np.pi, n_phases)
phi1s, phi2s = np.meshgrid(phis, phis)
phi1s = phi1s.flatten()
phi2s = phi2s.flatten()

n_twisted_bc = phi1s.size
gstate_es = np.zeros(n_twisted_bc)

# define paramsn
detune = 0
rabi = 0
J = -2.166
print_all = 1

nx = 3
ny = 4

# define geometry
bc1_open = False
bc2_open = True
phi1 = 0
phi2 = 0

if nx > 1:
    bc1_open = False
else:
    bc1_open = True

if ny > 1:
    bc2_open = False
else:
    bc2_open = True

for ii, (phi1, phi2) in enumerate(zip(phi1s, phi2s)):
    gm = geom.Geometry.createSquareGeometry(nx, ny, phi1, phi2, bc1_open, bc2_open)
    ss = ed_spins.spinSystem()

    jsmat = ss.get_interaction_mat_heisenberg(gm, J)

    # build all possible symmetries
    # x-translation
    xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
    xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)
    xtransl_op = ss.get_xform_op(xtransl_cycles)
    xtransl_op = xtransl_op

    # y-translations
    ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
    ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, gm)
    ytransl_op = ss.get_xform_op(ytransl_cycles)
    ytransl_op = ytransl_op

    # get projectors
    if not bc1_open and not bc2_open:
        projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                          max_cycle_len_transly, print_all)
    elif not bc1_open:
        projs, kys = symm.getZnProjectors(ytransl_op, max_cycle_len_transly, print_all)
        kxs = np.zeros(kys.shape)
    elif not bc2_open:
        projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx, print_all)
        kys = np.zeros(kxs.shape)
    else:
        projs = None

    # diagonalize all symmetry sectors
    eigvals_sectors = []
    for proj in projs:
        H = ss.createH(gm.nsites, detune, rabi, jsmat, projector=proj, print_results=print_all)
        eigs_temp, _ = ss.diagH(H, print_results=print_all)

        eigvals_sectors.append(eigs_temp)

    #eigs_all_sectors = np.sort(np.concatenate(eigvals_sectors))
    gstate_es[ii] = np.min(np.concatenate(eigvals_sectors))
