
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# define params
print_all = 1

Detune = 0.33
Rabi = 1.022
J = -2.166

#####################################
#  do v4 symmetries
#####################################
import ed_spins
import ed_geometry as geom
import ed_symmetry as symm

ti6 = ed_spins.spinSystem()

# create geometry
bc1_open = 1
bc2_open = 1
phase1 = 0
phase2 = 0
nsites6 = 3
xlocs6 = np.array([0.5, 1, 0])
ylocs6 = np.array([np.sqrt(3)/2, 0, 0])
periodicity_vect1 = np.zeros([2, 1])
periodicity_vect2 = np.zeros([2, 1])
connectmat6 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
xlocs6, ylocs6 = geom.getCenteredSiteLocs(xlocs6, ylocs6)

geom.dispGeometry(xlocs6, ylocs6, connectmat6)
plt.show()

# symmetries on sites
sites6 = range(0, nsites6)
# pi/2 rotation function
rot_fn6 = geom.getRotFn(3)
rot_cycles6, max_cycle_len_rot6 = geom.findSiteCycles(rot_fn6, xlocs6, ylocs6, periodicity_vect1, periodicity_vect2, print_all)
rot_op6 = ti6.get_xform_op(rot_cycles6, print_all)

# reflection about y-axis
refl_fn6 = geom.getReflFn(np.array([0, 1]))
refl_cycles6, max_cycle_len_refl6 = geom.findSiteCycles(refl_fn6, xlocs6, ylocs6, periodicity_vect1, periodicity_vect2, print_all)
refl_op6 = ti6.get_xform_op(refl_cycles6, print_all)

projs6 = symm.getD3Projectors(rot_op6, refl_op6, print_all)
#projs6 = symm.getCnProjectors(rot_op6, 3, print_all)

# set up Hamiltonian
jsmat6 = ti6.getInteractionMat(xlocs6, ylocs6, connectmat6, J)
ryd_detunes6 = ti6.get_rydberg_detunes(nsites6, jsmat6)


eigvals6 = []
for proj in projs6:
    H = ti6.createH(nsites6, Detune + ryd_detunes6, Rabi, jsmat6, projector= proj, print_results = print_all)
    eigs_temp, _ = ti6.diagH(H, print_results = print_all)

    eigvals6.append(eigs_temp)

eigs_symms6 = np.sort(np.concatenate(eigvals6))

# also do full Hamiltonian
H = ti6.createH(nsites6, Detune + ryd_detunes6, Rabi, jsmat6, print_results = print_all)
eigvals_full6, _ = ti6.diagH(H, print_results = print_all)

# compare eigenvalues using symmetries and not
print "For v6, Eigenvalues calculated with and without symmetries differed by at most %0.3e" % (eigvals_full6 - eigs_symms6).max()
