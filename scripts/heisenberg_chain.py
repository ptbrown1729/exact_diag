"""
exact digaonalization of Heisenberg system
"""

import numpy as np
import exact_diag.ed_geometry as geom
import exact_diag.ed_spins as spins
import exact_diag.ed_symmetry as symm

# set up system
J = 1
gm = geom.Geometry.createSquareGeometry(10, 1, 0, 0, bc1_open=False, bc2_open=True)
ss = spins.spinSystem(gm, jx=J, jy=J, jz=J)

# total sz and spin flip op
sz_op = ss.get_sum_op(ss.sz)
sz_flip_op = ss.get_swap_up_down_op()

# translation operators
xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
xtransl_cycles, ntranslations = symm.findSiteCycles(xtransl_fn, ss.geometry)
tx_op = ss.get_xform_op(xtransl_cycles)

# inversion op
cx, cy = ss.geometry.get_center_of_mass()
inv_fn = symm.getInversionFn(cx, cy)
inv_cycles, _ = symm.findSiteCycles(inv_fn, ss.geometry)
inv_op = ss.get_xform_op(inv_cycles)

# get projectors on mz subspace
mz_projs, mzs = ss.get_subspace_projs(sz_op.tocsr())

# sequentially get projectors on mz the x-translation (then spin flip for mz=0 sector only)
projs = []
mzs_all = []
kxs_all = []
spin_parity_all = []
space_parity_all = []
for p, mz in zip(mz_projs, mzs):
    tx_op_sub = p * tx_op * p.conj().transpose()
    tx_projs, kxs = symm.getZnProjectors(tx_op_sub, ss.geometry.nsites)

    if mz == 0:
        for tp, kx in zip(tx_projs, kxs):
            sz_flip_op_sub = tp * p * sz_flip_op * p.conj().transpose() * tp.conj().transpose()
            sz_projs, spin_parities = symm.getZnProjectors(sz_flip_op_sub, 2)
            spin_parities = np.round(np.exp(1j * spin_parities).real)

            for pp, parity in zip(sz_projs, spin_parities):
                if (pp * tp * p).shape[0] != 0:

                    if kx == 0 or kx == np.pi:
                        inv_op_sub = pp * tp * p * inv_op * \
                                     p.conj().transpose() * tp.conj().transpose() * pp.conj().transpose()
                        inv_projs, space_parities = symm.getZnProjectors(inv_op_sub, 2)
                        space_parities = np.round(np.exp(1j * space_parities).real)

                        for pps, parity_s in zip(inv_projs, space_parities):
                            proj_full = pps * pp * tp * p
                            if proj_full.shape[0] != 0:
                                projs.append(proj_full)
                                mzs_all.append(mz)
                                kxs_all.append(kx)
                                spin_parity_all.append(parity)
                                space_parity_all.append(parity_s)
                    else:
                        projs.append(pp * tp * p)
                        mzs_all.append(mz)
                        kxs_all.append(kx)
                        spin_parity_all.append(parity)
                        space_parity_all.append(0)

    else:
        for tp, kx in zip(tx_projs, kxs):
            if (tp*p).shape[0] != 0:
                if kx == 0 or kx == np.pi:
                    inv_op_sub = tp * p * inv_op * p.conj().transpose() * tp.conj().transpose()
                    inv_projs, space_parities = symm.getZnProjectors(inv_op_sub, 2)
                    space_parities = np.round(np.exp(1j * space_parities).real)

                    for pps, parity_s in zip(inv_projs, space_parities):
                        proj_full = pps * tp * p
                        if proj_full.shape[0] != 0:
                            projs.append(proj_full)
                            mzs_all.append(mz)
                            kxs_all.append(kx)
                            spin_parity_all.append(0)
                            space_parity_all.append(parity_s)
                else:
                    projs.append(tp * p)
                    mzs_all.append(mz)
                    kxs_all.append(kx)
                    spin_parity_all.append(0)
                    space_parity_all.append(0)

# create hamiltonian
ham = ss.createH(print_results=True)

eig_vals_all, _ = ss.diagH(ham, print_results=True)

eigs_sectors = []
for proj in projs:
    ev, _ = ss.diagH(proj * ham * proj.conj().transpose(), print_results=True)
    eigs_sectors.append(ev)

max_diff = np.max(np.abs(np.sort(np.concatenate(eigs_sectors)) - eig_vals_all))
print("max difference between eigenvalues from full diagonalization and symmetry sectors = %0.3g" % max_diff)

