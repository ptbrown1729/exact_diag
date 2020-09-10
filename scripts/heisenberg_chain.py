"""
exact digaonalization of Heisenberg system
"""

import numpy as np
import ed_geometry as geom
import ed_spins as spins
import ed_symmetry as symm

# set up system
J = 1
temps = np.array([0, 0.5, 1, 2, 10]) * J
betas = np.divide(1, temps)
betas[temps == 0] = np.inf

# ###############################
# constant number
# ###############################
gm = geom.Geometry.createSquareGeometry(10, 1, 0, 0, bc1_open=False, bc2_open=True)
ss = spins.spinSystem(gm, jx=J, jy=J, jz=J)

# total sz and spin flip op
sz_op = ss.get_sum_op(ss.sz)
sz_flip_op = ss.get_swap_up_down_op()

# translation operators
xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
xtransl_cycles, ntranslations = symm.findSiteCycles(xtransl_fn, ss.geometry)
tx_op = ss.get_xform_op(xtransl_cycles)

# get projectors on mz subspace
mz_projs, mzs = ss.get_subspace_projs(sz_op.tocsr())

# sequentially get projectors on mz the x-translation (then spin flip for mz=0 sector only)
projs = []
mzs_all = []
kxs_all = []
parity_all = []
for p, mz in zip(mz_projs, mzs):
    tx_op_sub = p * tx_op * p.conj().transpose()
    tx_projs, kxs = symm.getZnProjectors(tx_op_sub, ss.geometry.nsites)

    if mz == 0:
        for tp, kx in zip(tx_projs, kxs):
            sz_flip_op_sub = tp * p * sz_flip_op * p.conj().transpose() * tp.conj().transpose()
            sz_projs, parities = symm.getZnProjectors(sz_flip_op_sub, 2)
            parities = np.round(np.exp(1j * parities).real)

            for pp, parity in zip(sz_projs, parities):
                proj_full = pp * tp * p
                if proj_full.shape[0] != 0:
                    projs.append(proj_full)
                    mzs_all.append(mz)
                    kxs_all.append(kx)
                    parity_all.append(parity)

    else:
        for tp, kx in zip(tx_projs, kxs):
            proj_full = tp * p
            if proj_full.shape[0] != 0:
                projs.append(proj_full)
                mzs_all.append(mz)
                kxs_all.append(kx)
                parity_all.append(0)

# create hamiltonian
ham = ss.createH(print_results=True)

eig_vals_all, _ = ss.diagH(ham, print_results=True)

eigs_sectors = []
for proj in projs:
    ev, _ = ss.diagH(proj * ham * proj.conj().transpose(), print_results=True)
    eigs_sectors.append(ev)

max_diff = np.max(np.abs(np.sort(np.concatenate(eigs_sectors)) - eig_vals_all))
print("max difference between eigenvalues from full diagonalization and symmetry sectors = %0.3g" % max_diff)

