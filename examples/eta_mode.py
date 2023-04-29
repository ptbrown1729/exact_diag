"""
Eta mode calculations for attractive Hubbard model
 todo: does this script work?
"""

import time
import datetime
import os.path
import numpy as np
import scipy.linalg
import scipy.integrate
import scipy.sparse as sp
import matplotlib.pyplot as plt
from exact_diag import ed_spins
import exact_diag.ed_geometry as geom
import exact_diag.ed_symmetry as symm

# parameters
save_results = False
plot_results = True
t_script_start = time.process_time()
print_all = False

# define geometry
bc1_open = False
bc2_open = True
phi1 = 0
phi2 = 0

nx = 10
ny = 1
gm = geom.Geometry.createSquareGeometry(nx, ny, phi1, phi2, bc1_open, bc2_open)

# at first confused because setting one j direction to one and the others to zero gave different results
# for different choices of directions. For example, jz = 1 has perfect staggered magnetization, but jx = 1 has
# small staggered magnetization, and jy = 1 has no staggered magnetization. But the problem is there are two
# degenerate anti-ferromagnetic states, and the way the eigenvalue solver chooses these is unclear. So in the
# jz case it picked a state with perfect staggered magnetization but no correlations. In the jx case it picked
# a state with some correlations and some staggered magnetization. In the jy = 1case it picked a state with
# no staggered magnetization and all correlations.
U = -4
jx = 0.5 * 4 / np.abs(U) # 4t^2/|U|
jy = jx
jz = jx
hx = 0.
hy = 0.
hz = 2*(-0.4) # 2*mu - U
larmor_period = 2 * np.pi / hz
print("U = %0.2f" % U)
print("J = %0.2f" % jx)
print("hz = %0.2f" % hz)
print("larmor period_exp = %0.2f" % larmor_period)
# factor of two based on my convention for hamiltonian = 0.5 *\sum sigma*sigma * 2 * \sum spin*spin

ss = ed_spins.spinSystem(gm, jx, jy, jz, hx, hy, hz)

hamiltonian = ss.createH(projector=None, print_results=print_all)
eigvals, eigvects = ss.diagH(hamiltonian)

# useful operators
sigma_z_op = ss.get_sum_op(ss.pauli_z)
sigma_y_op = ss.get_sum_op(ss.pauli_y)
sigma_x_op = ss.get_sum_op(ss.pauli_x)
# ops on each site
sigma_z_sites = []
sigma_y_sites = []
sigma_x_sites = []
for ii in range(0, ss.geometry.nsites):
    sigma_z_sites.append(ss.getSingleSiteOp(ii, ss.geometry.nsites, ss.pauli_z, format="boson"))
    sigma_y_sites.append(ss.getSingleSiteOp(ii, ss.geometry.nsites, ss.pauli_y, format="boson"))
    sigma_x_sites.append(ss.getSingleSiteOp(ii, ss.geometry.nsites, ss.pauli_x, format="boson"))

# extract ground state properties for diagnostics
# correlations between sites 0 and 1
sigma_z_corr_op = ss.getTwoSiteOp(0, 1, ss.geometry.nsites, ss.pauli_z, ss.pauli_z, format="boson")
sigma_y_corr_op = ss.getTwoSiteOp(0, 1, ss.geometry.nsites, ss.pauli_y, ss.pauli_y, format="boson")
sigma_x_corr_op = ss.getTwoSiteOp(0, 1, ss.geometry.nsites, ss.pauli_x, ss.pauli_x, format="boson")

# expectation values
# net values
sigma_z_exp_gs = ss.get_exp_vals(eigvects[:, 0], sigma_z_op)
sigma_y_exp_gs = ss.get_exp_vals(eigvects[:, 0], sigma_y_op)
sigma_x_exp_gs = ss.get_exp_vals(eigvects[:, 0], sigma_x_op)
# magnetization in each direction on each site
sigma_z_site_exp_gs = np.zeros(ss.geometry.nsites)
sigma_y_site_exp_gs = np.zeros(ss.geometry.nsites)
sigma_x_site_exp_gs = np.zeros(ss.geometry.nsites)
for ii in range(0, ss.geometry.nsites):
    sigma_z_site_exp_gs[ii] = ss.get_exp_vals(eigvects[:, 0], sigma_z_sites[ii])
    sigma_y_site_exp_gs[ii] = ss.get_exp_vals(eigvects[:, 0], sigma_y_sites[ii])
    sigma_x_site_exp_gs[ii] = ss.get_exp_vals(eigvects[:, 0], sigma_x_sites[ii])
# correlations between sites 0 and 1
sigma_z_corr = ss.get_exp_vals(eigvects[:, 0], sigma_z_corr_op)
sigma_y_corr = ss.get_exp_vals(eigvects[:, 0], sigma_y_corr_op)
sigma_x_corr = ss.get_exp_vals(eigvects[:, 0], sigma_x_corr_op)

print("min eig = %0.2f" % eigvals[0])
print("x net mag = %0.2f" % sigma_x_exp_gs)
print("y net mag = %0.2f" % sigma_y_exp_gs)
print("z net mag = %0.2f" % sigma_z_exp_gs)
print("x corr = %0.2f" % sigma_x_corr)
print("y corr = %0.2f" % sigma_y_corr)
print("z corr = %0.2f" % sigma_z_corr)
print(sigma_x_site_exp_gs)
print(sigma_y_site_exp_gs)
print(sigma_z_site_exp_gs)

# operators related to SU(2) symmetry
Q = np.array([np.pi, np.pi])
# Q = np.array([0, 0])
s_plus_all_op = ss.get_sum_op_q(0, [0,0], ss.pauli_plus, format="boson", print_results=print_all)
s_plus_q_op = ss.get_sum_op_q(0, Q, ss.pauli_plus, format="boson", print_results=print_all)
s_plus_mq_op = ss.get_sum_op_q(0, -Q, ss.pauli_plus, format="boson", print_results=print_all)
s_minus_all_op = ss.get_sum_op_q(0, [0,0], ss.pauli_minus, format="boson", print_results=print_all)
s_minus_q_op = ss.get_sum_op_q(0, Q, ss.pauli_minus, format="boson", print_results=print_all)
s_minus_mq_op = ss.get_sum_op_q(0, -Q, ss.pauli_minus, format="boson", print_results=print_all)
sz_q_op = ss.get_sum_op_q(0, Q, ss.pauli_z, format="boson", print_results=print_all)
sz_mq_op = ss.get_sum_op_q(0, -Q, ss.pauli_z, format="boson", print_results=print_all)
sy_q_op = ss.get_sum_op_q(0, Q, ss.pauli_y, format="boson", print_results=print_all)
sy_mq_op = ss.get_sum_op_q(0, -Q, ss.pauli_y, format="boson", print_results=print_all)
sx_q_op = ss.get_sum_op_q(0, Q, ss.pauli_x, format="boson", print_results=print_all)
sx_mq_op = ss.get_sum_op_q(0, -Q, ss.pauli_x, format="boson", print_results=print_all)

print("Gap expectation value, ground state = %0.2f" % ss.get_exp_vals(eigvects[:, 0], s_minus_q_op))

# matrix elements
s_plus_matrixel = ss.get_matrix_elems(eigvects, s_plus_all_op, print_results=print_all)
s_minus_matrixel = ss.get_matrix_elems(eigvects, s_minus_all_op, print_results=print_all)
sz_q_matrixel = ss.get_matrix_elems(eigvects, sz_q_op, print_results=print_all)
sz_mq_matrixel = ss.get_matrix_elems(eigvects, sz_mq_op, print_results=print_all)
sy_q_matrixel = ss.get_matrix_elems(eigvects, sy_q_op, print_results=print_all)
sy_mq_matrixel = ss.get_matrix_elems(eigvects, sy_mq_op, print_results=print_all)
sx_q_matrixel = ss.get_matrix_elems(eigvects, sx_q_op, print_results=print_all)
sx_mq_matrixel = ss.get_matrix_elems(eigvects, sx_mq_op, print_results=print_all)
# response functions
temperature = 0
sp_sm_resp_fn = ss.get_response_fn_retarded(s_plus_matrixel, s_minus_matrixel, eigvals, temperature, print_results=True)
sm_sp_resp_fn = ss.get_response_fn_retarded(s_minus_matrixel, s_plus_matrixel, eigvals, temperature, print_results=True)
# eta_sz_resp_fn = ss.get_response_fn(s_plus_matrixel, sz_mq_matrixel, eigvals, temperature, print_results=print_all)
sz_sp_resp_fn = ss.get_response_fn_retarded(sz_q_matrixel, s_plus_matrixel, eigvals, temperature, print_results=True)
sz_sm_resp_fn = ss.get_response_fn_retarded(sz_q_matrixel, s_minus_matrixel, eigvals, temperature, print_results=True)
sz_sz_resp_fn = ss.get_response_fn_retarded(sz_q_matrixel, sz_mq_matrixel, eigvals, temperature, print_results=True)
sy_sz_resp_fn = ss.get_response_fn_retarded(sy_q_matrixel, sz_mq_matrixel, eigvals, temperature, print_results=True)
sx_sz_resp_fn = ss.get_response_fn_retarded(sx_q_matrixel, sz_mq_matrixel, eigvals, temperature, print_results=True)

broadening = 0.01
omegas = np.linspace(-5 * np.abs(hz), 5 * np.abs(hz), 100)
sp_sm_resp = np.zeros(omegas.size)
sm_sp_resp = np.zeros(omegas.size)
sz_sp_resp = np.zeros(omegas.size)
sz_sm_resp = np.zeros(omegas.size)
sz_sz_resp = np.zeros(omegas.size)
sx_sz_resp = np.zeros(omegas.size)
for ii in range(0, omegas.size):
    sp_sm_resp[ii] = sp_sm_resp_fn(omegas[ii], broadening)
    sm_sp_resp[ii] = sm_sp_resp_fn(omegas[ii], broadening)
    sz_sp_resp[ii] = sz_sp_resp_fn(omegas[ii], broadening)
    sz_sm_resp[ii] = sz_sm_resp_fn(omegas[ii], broadening)
    sz_sz_resp[ii] = sz_sz_resp_fn(omegas[ii], broadening)
    sx_sz_resp[ii] = sx_sz_resp_fn(omegas[ii], broadening)


plt.figure(1)
nrows = 2
ncols = 3
plt.subplot(nrows, ncols, 1)
plt.plot(omegas, sz_sz_resp)
plt.ylabel('frequency')
plt.title('Sz(Q) Sz(-Q) resp fn, imaginary part')

plt.subplot(nrows, ncols, 3)
plt.plot(omegas, sm_sp_resp)
plt.ylabel('frequency')
plt.title('S^- S^+ resp fn, imaginary part')

plt.subplot(nrows, ncols, 6)
plt.plot(omegas, sp_sm_resp)
plt.ylabel('frequency')
plt.title('S^+ S^- resp fn, imaginary part')

plt.subplot(nrows, ncols, 2)
plt.plot(omegas, sz_sp_resp)
plt.ylabel('frequency')
plt.title('sz(Q) s^+ resp fn, imaginary part')

plt.subplot(nrows, ncols, 5)
plt.plot(omegas, sz_sm_resp)
plt.ylabel('frequency')
plt.title('sz(Q) s^- resp fn, imaginary part')



plt.draw()

#### interesting time dependent test to see what eta mode is doing
# start in ground state and rotate sites alternating directions out of x-y plane (this also rotates the net magnetization)
# Net magnetization starts larmor precession, and spins on alternating sites nutate out of phase. The larmor precession
# is the superfluid gap phase changing, and the nutation is the fluctuating charge density wave order
angle = np.pi/ 2.
rot_axis_op = ss.pauli_y
spin_rot_op_even = scipy.linalg.expm(1j * rot_axis_op * angle)
spin_rot_op_odd = scipy.linalg.expm(1j * rot_axis_op * angle)
s_alt_rot_op = ss.getSingleSiteOp(0, ss.geometry.nsites, spin_rot_op_even, format="boson")
for ii in range(1, ss.geometry.nsites):
    if np.mod(ii, 2) == 0:
        op = spin_rot_op_even #np.exp(-1j * rot_axis_op * angle)
    else:
        # op = np.exp(1j * rot_axis_op * angle)
        op = spin_rot_op_odd
    s_alt_rot_op = s_alt_rot_op.dot(ss.getSingleSiteOp(ii, ss.geometry.nsites, op, format="boson"))

#excitation_state = ss.getSingleSiteOp(0, ss.geometry.nsites, ss.pauli_plus, format="boson").dot(eigvects[:, 0][:, None])
excitation_state = s_alt_rot_op.dot(eigvects[:, 0][:, None])
print("norm of initial state = %0.2f" % ss.get_norms(excitation_state))
# have to be careful this is not zero...
excitation_state = excitation_state / np.sqrt(ss.get_norms(excitation_state))
sigma_z_exp_es = ss.get_exp_vals(excitation_state, sigma_z_op)
print("initial time evolution state had z magnetization = %0.2f" % sigma_z_exp_es)

# time evolve eta + state
times = np.linspace(0, 2 * larmor_period, 1000)
_, evolved_states = ss.quench_time_evolve(excitation_state, eigvects, eigvals, times, print_results=print_all)

sz_time_evolved = ss.get_exp_vals(evolved_states, sigma_z_op)
sy_time_evolved = ss.get_exp_vals(evolved_states, sigma_y_op)
sx_time_evolved = ss.get_exp_vals(evolved_states, sigma_x_op)

sz_time_evolved_sites = np.zeros([len(times), ss.geometry.nsites])
sy_time_evolved_sites = np.zeros([len(times), ss.geometry.nsites])
sx_time_evolved_sites = np.zeros([len(times), ss.geometry.nsites])
for ii in range(0, ss.geometry.nsites):
    sz_time_evolved_sites[:, ii] = ss.get_exp_vals(evolved_states, sigma_z_sites[ii])
    sy_time_evolved_sites[:, ii] = ss.get_exp_vals(evolved_states, sigma_y_sites[ii])
    sx_time_evolved_sites[:, ii] = ss.get_exp_vals(evolved_states, sigma_x_sites[ii])

# plot results on each site in time
leg = []
for ii in range(0, ss.geometry.nsites):
    leg.append("site %d" % ii)

plt.figure(2)
plt.subplot(2, 3, 1)
plt.plot(times, sz_time_evolved_sites)
plt.xlabel('time')
plt.ylabel('<Sz>')
plt.legend(leg)

plt.subplot(2, 3, 4)
plt.plot(times, sz_time_evolved)
plt.xlabel('time')
plt.ylabel('<Sz>_all')

plt.subplot(2, 3, 2)
plt.plot(times, sy_time_evolved_sites)
plt.xlabel('time')
plt.ylabel('<Sy>')

plt.subplot(2, 3, 5)
plt.plot(times, sy_time_evolved)
plt.xlabel('time')
plt.ylabel('<Sy>_all')

plt.subplot(2, 3, 3)
plt.plot(times, sx_time_evolved_sites)
plt.xlabel('time')
plt.ylabel('<Sx>')

plt.subplot(2, 3, 6)
plt.plot(times, sx_time_evolved)
plt.xlabel('time')
plt.ylabel('<Sx>_all')

plt.show()
