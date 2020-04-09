"""
Compare exact diagonalization of 4-spin Heisenberg model with couplings  J1 = J12; J2 = J13 = J23; J3 = J14 - J24 = J34
to handle calculated result.
"""

import numpy as np
import ed_geometry as geom
import ed_spins

nx = 4
ny = 1
phi1 = 0
phi2 = 0
bc_open1 = 1
bc_open2 = 1
gm = geom.Geometry.createSquareGeometry(nx, ny, phi1, phi2, bc_open1, bc_open2)
gm.dispGeometry()

j1 = 0.3333
j2 = 0.23623627
j3 = 0.8434783478
# -1 to account for difference in definitions
js = -np.array([[0, j1, j2, j3], [j1, 0, j2, j3], [j2, j2, 0, j3], [j3, j3, j3, 0]])
ss = ed_spins.spinSystem(gm, jx=js, jy=js, jz=js, spin=1.5)
hamiltonian = ss.createH()

eig_vals, eig_vects = ss.diagH(hamiltonian)

# verify these are equal to expected result
# E = j1 * [s1*(s1+1) + s2*(s2+1)] + j2 * s3*(s3+1) + j3 * s4*(s4+1) + (j2-j1)*s12*(s12+1) + (j3-j2)*s123*(s123+1) - j3 * s1234 * (s1234 + 1)
eigs_analytic = []
s1 = 1.5
s2 = 1.5
s3 = 1.5
s4 = 1.5
s12s = np.arange(np.abs(s1 - s2), s1 + s2 + 1)

for s12 in s12s:
    for s123 in np.arange(np.abs(s3 - s12), s3 + s12 + 1):
        for s1234 in np.arange(np.abs(s4 - s123), s4 + s123 + 1):
            eig = j1 * (s1 * (s1 + 1) + s2 * (s2 + 1)) + \
                  j2 * s3 * (s3 + 1) + \
                  j3 * s4 * (s4 + 1) + \
                  (j2 - j1) * s12 * (s12 + 1) + \
                  (j3 - j2) * s123 * (s123 + 1) - \
                  j3 * s1234 * (s1234 + 1)
            multiplicity = int(2 * s1234 + 1)
            eigs_analytic += [eig] * multiplicity
eigs_analytic.sort()

assert np.max(eigs_analytic - eig_vals) < 1e-12