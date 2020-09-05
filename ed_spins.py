import time
import numpy as np
import scipy.sparse as sp
import ed_geometry as geom
import ed_symmetry as symm
import ed_base

class spinSystem(ed_base.ed_base):

    #nbasis = 2
    nspecies = 1
    # index that "looks" the same as the spatial index from the perspective of constructing operators. E.g. for
    # fermions we have spatial index + spin index, but it is convenient to treat this like an enlarged set of
    # spatial indices. This is the reason it is called "nspin"
    ng = np.array([[0, 0], [0, 1]])
    nr = np.array([[1, 0], [0, 0]])
    nplus = 0.5 * np.array([[1, 1], [1, 1]])
    nminus = 0.5 * np.array([[1, -1], [-1, 1]])
    swap_op = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def __init__(self, geometry, jx=0.0, jy=0.0, jz=0.0, hx=0.0, hy=0.0, hz=0.0, use_ryd_detunes=0, spin=0.5):
        """
        This class implements a spin system with the following Hamiltonian:
        H = \sum_<i, j, \alpha=x,y,z> 0.5 * j_alpha * \sigma_i^\alpha * \sigma_j^\alpha -
            \sum_{i, \alpha=x,y,z} 0.5 * h_\alpha * \sigma_i^\alpha
        If write this in terms of spin operators instead of Pauli matrices, S_i^\alpha = 0.5 * \sigma_i^\alpha
        H = \sum_<i, j, \alpha=x,y,z> 2 * j_\alpha * S_i^\alpha * \S_j^\alpha -
             \sum_{i, \alpha=x,y,z} h_\alpha * S_i^\alpha
        :param geometry:
        :param jx:
        :param jy:
        :param jz:
        :param hx:
        :param hy:
        :param hz:
        :param use_ryd_detunes:
        """
        # TODO: update this to allow easy use of Heisenberg or Rydberg
        # TODO: finish updating class to work appropriately with this...
        # TODO: update base class to assume we have the geometry field, which will make the signature of fns such as get_xform_op the same for the derived classes.
        self.nbasis = int(2 * spin + 1)
        self.splus = self.get_splus(spin)
        self.sminus = self.get_sminus(spin)
        self.sz = self.get_sz(spin)
        self.sy = self.get_sy(spin)
        self.sx = self.get_sx(spin)

        ed_base.ed_base.__init__(self, geometry)
        self.jx = self.get_interaction_mat(jx)
        self.jy = self.get_interaction_mat(jy)
        self.jz = self.get_interaction_mat(jz)
        self.hx = self.get_field_mat(hx)
        self.hy = self.get_field_mat(hy)
        self.hz = self.get_field_mat(hz)

        if use_ryd_detunes:
            self.rydberg_detunings = self.get_rydberg_detunes(self.jz)
            self.hz = self.hz + self.rydberg_detunings

    def get_splus(self, spin):
        """
        S^+ |s m> = sqrt( (s-m) * (s+m+1) ) |s (m+1)>
        :param spin:
        :return:
        """
        ms = np.arange(spin - 1, -spin - 1, -1)
        return sp.diags(np.sqrt((spin - ms) * (spin + ms + 1)), -1)

    def get_sminus(self, spin):
        """
        S^- |s m> = sqrt( (s+m) * (s_m+1) ) |s (m-1)>
        :param spin:
        :return:
        """
        ms = np.arange(spin, -spin, -1)
        return sp.diags(np.sqrt((spin + ms) * (spin - ms + 1)), 1)

    def get_sx(self, spin):
        """
        s^x = 0.5 * (s^+ + s^-)
        :param spin:
        :return:
        """
        return 0.5 * (self.get_splus(spin) + self.get_sminus(spin))

    def get_sy(self, spin):
        """
        s^y = 0.5 * (s^+ - s^-)
        :param spin:
        :return:
        """
        return 0.5 * (self.get_splus(spin) - self.get_sminus(spin)) / 1j

    def get_sz(self, spin):
        """
        S^z |s m> = m |s m>
        :param spin:
        :return:
        """
        return sp.diags(np.arange(spin, -spin - 1, -1))

    def get_interaction_mat_c6(self, C6, cutoff_dist=1.5):
        """
        Get interaction matrix by using real space distances. This allows, e.g. anisotropic interactions_4.
        :param xlocs:
        :param ylocs:
        :param C6:
        :param xscale:
        :param yscale:
        :param cutoff_dist:
        :return:
        """
        distmat = np.sqrt(self.geometry.site_distances_reduced_x ** 2 + self.geometry.site_distances_reduced_y ** 2)
        jmat = np.zeros(distmat.shape)
        jmat[distmat != 0] = C6*np.power(np.reciprocal(distmat[distmat != 0]), 6)
        return jmat

    def get_interaction_mat(self, j):
        """
        Get interaction matrix, which is a matrix of size nsites x nsites where m[ii, jj] gives the interaction
        strength between sites ii and jj
        :param j: integer giving uniform interaction strength for all sites
        :return:
        """
        if isinstance(j, (int, float)):
            j_mat = j * self.geometry.adjacency_mat * self.geometry.phase_mat
        elif isinstance(j, np.ndarray):
            if j.shape == (self.geometry.nsites, self.geometry.nsites):
                j_mat = j
            elif j.size == self.geometry.nsites:
                j_mat = np.diag(j[:-1], 1) + np.diag(j[-1], -1)
                j_mat[0, self.geometry.nsites - 1] = j[-1]
            else:
                raise Exception('j should be nsites x nsites matrix or a list of size nsites')
        else:
            raise Exception('j was not integer, float, or numpy array.')

        return j_mat

    def get_field_mat(self, h):
        if isinstance(h, (int, float)):
            h_mat = h * np.ones(self.geometry.nsites)
        elif isinstance(h, np.ndarray):
            if h.size != self.geometry.nsites:
                raise Exception('j was a numpy array, but size did not match geometry.')
            h_mat = h
        else:
            raise Exception('j was not integer, float, or numpy array.')

        return h_mat

    def get_state_vects(self, print_results=False):
        """
        Generate a description of the basis states in the full tensor product of spins space
        :param nsites: Int, total number of sites in the system
        :param print_results:
        :return: NumPy array of size 2 ** nsites x nsites describing each basis state in the tensor product spin space.
        Each row represents the spins for a given state according to |up> = 1, |down> = 0 on the site corresponding to
        the column index.
        """
        if print_results:
            tstart = time.time()

        nsites = self.geometry.nsites
        StateSpinLabels = sp.csc_matrix((2 ** nsites, nsites))  # csc 0.7s vs. csr 9s. As expected for vertical slicing.
        RootMat = sp.csc_matrix([[1], [0]])
        for ii in range(0, nsites):
            # simple pattern to create the columns of the matrix. It goes like this: the last column alternates as
            # 1,0,1,0,... the second to last column goes 1,1,0,0,1,1,..., all the way to the first, which goes,
            # 1,1,...,1,0,...0 (is first half ones, second half zeros).
            StateSpinLabels[:, ii] = sp.kron(np.ones([2 ** ii, 1]),
                                             sp.kron(RootMat, np.ones([2 ** (nsites - ii - 1), 1])))
        # self.StateSpinLabels = StateSpinLabels
        StateSpinLabels.eliminate_zeros()
        if print_results:
            tend = time.time()
            print("Took %0.2f s to generate state vector labels" % (tend - tstart))
        return StateSpinLabels

    def get_state_parity(self, print_results=False):
        """
        Get parity of basis states
        :param nsites:
        :param nstates:
        :param print_results:
        :return:
        """
        if print_results:
            tstart = time.time()
        StateParity = np.mod(np.array(self.get_state_vects(self.geometry.nsites, 2 ** self.geometry.nsites).sum(1)), 2)
        if print_results:
            tend = time.time()
            print("get_state_parity took %0.2fs" % (tend - tstart))
        return StateParity

    # ########################
    # Build and diagonalize H
    # ########################

    def get_rydberg_detunes(self, jsmat):
        """
        Generate rydberg detuning terms for each site.
        :param nsites: Int, number of sites
        :param jsmat: nsites x nsites NumPy array
        :return: rydberg_detunings, an nsites NumPy array
        """

        nsites = self.geometry.nsites
        rydberg_detunings = np.zeros(nsites, dtype = np.complex)
        for ii in range(0, nsites):
            rydberg_detunings[ii] = 0.5 * np.sum(jsmat[ii, :])
        rydberg_detunings = -2.0*rydberg_detunings  # need this to keep same term in Hamiltonian had before.

        if not (rydberg_detunings.imag > 10 ** -self._round_decimals).any():
            rydberg_detunings = rydberg_detunings.real

        return rydberg_detunings

    def createH(self, projector=None, print_results=False):
        """

        :param nsites: Int, total number of sites
        :param detunes: Int or NumPy array of length nsites, specifying detuning (longitudinal field) at each site
        :param rabis: Int or NumPy array of length nsites, specifying rabi frequency (transverse field) at each site
        :param jsmat: NumPy array of size nsites x nsites x 3. Interaction term between sites ii, jj is jsmat[ii,jj,kk]*sigma^kk_i*sigma^kk_j
        where kk = x, y, or z.
        :param projector:
        :param print_results:
        :return:
        """
        #TODO: if I do twisted boundary conditions can I still only sum over ii > jj?
        nsites = self.geometry.nsites
        nstates = self.nbasis ** nsites

        if print_results:
            tstart = time.process_time()

        if projector is None:
            projector = sp.eye(nstates)
        nstates = projector.shape[0]

        # transverse and longitudinal field terms
        H = sp.csr_matrix((nstates, nstates))
        for ii in range(0, nsites):
            if self.hx[ii] != 0:
                # THESE factors of two related to the fact I've written things in terms of the pauli matrices, instead
                # of spin matrix = 0.5 * pauli_matrices
                # H = H - 0.5 * self.hx[ii] * projector * self.get_single_site_op(ii, 0, self.pauli_x, format="boson") * \
                #     projector.conj().transpose()
                H = H - self.hx[ii] * projector * self.get_single_site_op(ii, 0, self.sx, format="boson") * \
                    projector.conj().transpose()
            if self.hy[ii] != 0:
                # H = H - 0.5 * self.hy[ii] * projector * self.get_single_site_op(ii, 0, self.pauli_y, format="boson") * \
                #     projector.conj().transpose()
                H = H - self.hy[ii] * projector * self.get_single_site_op(ii, 0, self.sy, format="boson") * \
                    projector.conj().transpose()
            if self.hz[ii] != 0:
                # H = H - 0.5 * self.hz[ii] * projector * self.get_single_site_op(ii, 0, self.pauli_z, format="boson") * \
                #     projector.conj().transpose()
                H = H - self.hz[ii] * projector * self.get_single_site_op(ii, 0, self.sz, format="boson") * \
                    projector.conj().transpose()

        # interaction terms
        if nsites > 1:
            for ii in range(0, nsites):
                for jj in range(0, nsites):
                    if ii > jj:  # to avoid repeating elements
                        jx = self.jx[ii, jj]
                        jy = self.jy[ii, jj]
                        jz = self.jz[ii, jj]

                        # factors of 2 =>  0.5 * \sum sigma*sigma = 2 * \sum s*s
                        if jx != 0:
                            # H = H + 0.5 * jx * projector * \
                            #     self.get_two_site_op(ii, 0, jj, 0, self.pauli_x, self.pauli_x, format="boson") * \
                            #     projector.conj().transpose()
                            H = H + 2 * jx * projector * \
                                self.get_two_site_op(ii, 0, jj, 0, self.sx, self.sx, format="boson") * \
                                projector.conj().transpose()
                        if jy != 0:
                            # H = H + 0.5 * jy * projector * \
                            #     self.get_two_site_op(ii, 0, jj, 0, self.pauli_y, self.pauli_y, format="boson") * \
                            #     projector.conj().transpose()
                            H = H + 2 * jy * projector * \
                                self.get_two_site_op(ii, 0, jj, 0, self.sy, self.sy, format="boson") * \
                                projector.conj().transpose()
                        if jz != 0:
                            # H = H + 0.5 * jz * projector * \
                            #     self.get_two_site_op(ii, 0, jj, 0, self.pauli_z, self.pauli_z, format="boson") * \
                            #     projector.conj().transpose()
                            H = H + 2 * jz * projector * \
                                self.get_two_site_op(ii, 0, jj, 0, self.sz, self.sz, format="boson") * \
                                projector.conj().transpose()

        if print_results:
            tend = time.process_time()
            print("Constructing H of size %d x %d took %0.2f s" % (H.shape[0], H.shape[0], tend - tstart))

        return H

    def get_interaction_op(self, projector=None):
        pass

    def get_field_op(self, projector=None):
        pass

    # ########################
    # miscellanoues functions
    # ########################

    def find_special_states(self, XSites, YSites, number_left2right_top2bottom=0):
        """
        Generate state vectors for special states. Currently, the ferromagnetic states, anti-ferromagnetic states,
        and plus and minus product states
        :param XSites:
        :param YSites:
        :param number_left2right_top2bottom:
        :return:
        """
        NSites = self.geometry.nsites#XSites * YSites
        NStates = 2 ** NSites

        AllUpStateInd = 0
        # AllUpState = np.zeros([NStates,1])
        # AllUpState[AllUpStateInd] = 1
        AllUpState = sp.csr_matrix((np.array([1]), (np.array([AllUpStateInd]), np.array([0]))), shape=(NStates, 1))

        AllDnStateInd = NStates - 1
        # AllDnState = np.zeros([NStates,1])
        # AllDnState[AllDnStateInd] = 1
        AllDnState = sp.csr_matrix((np.array([1]), (np.array([AllDnStateInd]), np.array([0]))), shape=(NStates, 1))

        if ((XSites % 2) or not number_left2right_top2bottom) and (NSites > 1):
            # if Xsites is odd
            if NSites % 2:
                # if nsites is odd
                AFMState1Ind = (1 + NStates) / 3 - 1
            else:
                # if nsites is even
                AFMState1Ind = (2 + NStates) / 3 - 1
            AFMState2Ind = NStates - AFMState1Ind
        else:
            print('AFM State finder not implemented for even number of sites in X-direction when using ' \
                      'conventional number, or not implemented for only a single site. Will return FM states instead')
            AFMState1Ind = 0
            AFMState2Ind = NStates - 1

        AFMState1 = sp.csr_matrix((np.array([1]), (np.array([AFMState1Ind]), np.array([0]))), shape=(NStates, 1))
        AFMState2 = sp.csr_matrix((np.array([1]), (np.array([AFMState2Ind]), np.array([0]))), shape=(NStates, 1))

        AllPlusState = np.ones([NStates, 1]) / np.sqrt(NStates)
        AllMinusState = sp.csr_matrix.dot(self.get_allsite_op(self.pauli_z), AllPlusState)

        SymmExcState = np.zeros([NStates, 1])
        for ii in range(0, NSites):
            SymmExcState[2 ** NSites - 2 ** ii - 1] = 1 / np.sqrt(NSites)

        Rows = 2 ** NSites - np.power(2, np.arange(0, NSites)) - 1
        Cols = np.zeros([NSites])
        Vals = np.ones([NSites]) / np.sqrt(NSites)
        SymmExcState = sp.csr_matrix((Vals, (Rows, Cols)), shape=(NStates, 1))

        states = sp.hstack((AllUpState, AllDnState, AFMState1, AFMState2, AllPlusState, AllMinusState, SymmExcState))
        Descriptions = ['AllUp', 'AllDn', 'AFM1', 'AFM2', 'AllPlus', 'AllMinus', 'SymmExc']
        return states.tocsr(), Descriptions

    # ########################
    # Calculate operators
    # ########################

    def get_npairs_op(self, corr_sites_conn):
        """
        Get operator that counts number of pairs of rydbergs. (i.e. spin-ups on NN sites)
        :param corr_sites_conn:
        :param nsites:
        :return:
        """
        nsites = self.geometry.nsites
        op = 0
        xs, ys = np.meshgrid(range(0, nsites), range(0, nsites))
        i_corr = xs[xs > ys]
        j_corr = ys[xs > ys]
        i_corr = i_corr[corr_sites_conn == 1]
        j_corr = j_corr[corr_sites_conn == 1]
        for ii in range(0, len(i_corr)):
            op = op + self.get_two_site_op(i_corr[ii], 0, j_corr[ii], 0, self.nr, self.nr, format="boson")
        return op

    def get_allsite_op(self, op_onsite):
        """
        operators which are product of nsites copies of a single operator.
        :param nsites: Int, total number of sites
        :param op_onsite:
        :return: op_full, sparse COO matrix
        """
        op_full = sp.coo_matrix(1)
        op_onsite = sp.coo_matrix(op_onsite)
        for ff in range(0, self.geometry.nsites):
            op_full = sp.kron(op_full, op_onsite, 'coo')
        return op_full

    def get_sum_op(self, op):
        """
        Get operator that counts the number of spins ups/rydberg excitations
        :param nsites: Int, total number of sites
        :param op: 2x2 operator acting on each site
        :return: OpMat, sparse matrix
        """
        species_index = 0
        return ed_base.ed_base.get_sum_op(self, op, species_index, format="boson", print_results=False)

    def get_swap_op(self, site1, site2, species=0):
        """
        Construct an operator that swaps the states of two sites. This version does not require any recursion.
        :param site1: Integer value specifying first site
        :param site2: Integer value specifying second site
        :param species: Integer specifying species (should always be zero in this case)
        :return: Sparse matrix
        """
        return self.get_two_site_op(site1, species, site2, species, self.pauli_plus, self.pauli_minus, format="boson") + \
               self.get_two_site_op(site1, species, site2, species, self.pauli_minus, self.pauli_plus, format="boson") + \
               0.5 * sp.eye(self.nstates) + \
               0.5 * self.get_two_site_op(site1, species, site2, species, self.pauli_z, self.pauli_z, format="boson")


if __name__ == "__main__":
    pass
