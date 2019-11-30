import numpy as np
import matplotlib.pyplot as plt

import ed_geometry as geom
import ed_symmetry as symm
import ed_fermions as hubbard

bc1_open = 0
bc2_open = 1
gm = geom.Geometry.createSquareGeometry(2, 1, 0, 0, bc1_open, bc2_open)
U = 0.0
tunn = 1.0
temperature = 1.0
eta = 0.1
model = hubbard.fermions(gm, U, tunn, mu_up=0., mu_dn=0.)

# translational symmetry projectors
xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)
xtransl_op = model.n_projector * model.get_xform_op(xtransl_cycles) * model.n_projector.conj().transpose()
if not bc2_open:
    ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
    ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, gm)
    ytransl_op = model.n_projector * model.get_xform_op(ytransl_cycles) * model.n_projector.conj().transpose()

# get projectors
if not bc1_open and not bc2_open:
    projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                      max_cycle_len_transly)
elif not bc2_open:
    projs, kys = symm.getZnProjectors(ytransl_op, max_cycle_len_transly)
    kxs = np.zeros(kys.shape)
elif not bc1_open:
    projs, kxs = symm.getZnProjectors(xtransl_op, max_cycle_len_translx)
    kys = np.zeros(kxs.shape)
else:
    raise Exception

# c_up = model.getSingleSiteOp(0, model.geometry.nsites * model.nspecies, model.c_op, 'fermion')
c_up = model.get_single_site_op(0, 0, model.c_op, 'fermion')

c_ups = []
for ii in range(0, model.geometry.nsites):
    #index = model.spinful2spinlessIndex(ii, model.geometry.nsites, 1)
    #ci_up = model.getSingleSiteOp(index, model.geometry.nsites * model.nspecies, model.c_op, 'fermion')
    ci_up = model.get_single_site_op(ii, 1, model.c_op, 'fermion')
    c_ups.append(ci_up)

omegas_interp = np.linspace(-10, 10, 100)
spectral_fn_samples = np.zeros((len(omegas_interp), len(kxs)))
for ii, proj in enumerate(projs):
    #ck_up_full = symm_proj.conj().transpose().dot(symm_proj).dot(c_ups[0])

    ck_up = 0
    for jj, c in enumerate(c_ups):
        ck_up = ck_up + c * np.exp(1j * kxs[ii] * model.geometry.xlocs[jj])
    ck_up = proj.dot(ck_up.dot(proj.conj().transpose()))

    ham_k = model.createH(projector=proj)
    eig_vals, eig_vects = model.diagH(ham_k)

    ck_up_matrixelms = model.get_matrix_elems(eig_vects, ck_up)

    spectral_fn = model.get_response_fn_retarded(ck_up_matrixelms, ck_up_matrixelms.conj().transpose(), eig_vals, temperature, format="fermion")
    for jj in range(0, len(omegas_interp)):
        spectral_fn_samples[jj, ii] = spectral_fn(omegas_interp[jj], eta)

leg = ['k = %0.2f' % kx for kx in kxs]

plt.figure()
plt.plot(omegas_interp, spectral_fn_samples)
plt.xlabel('Frequency')
plt.ylabel('A(w)')
plt.legend(leg)
plt.grid()
plt.show()