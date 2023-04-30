"""
Base class that other exact diagonalization classes inherit from
"""
import time
import datetime
import os
import ctypes
import pickle
import numpy as np
from scipy.io import savemat, loadmat
import scipy.sparse.linalg
import scipy.linalg
import scipy.sparse as sp
import exact_diag.ed_geometry as geom


class ed_base:
    # these must be overriden in derived classes
    nspecies = 0
    nbasis = 0

    # Pauli matrices
    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_plus = np.array([[0, 1], [0, 0]])
    pauli_minus = np.array([[0, 0], [1, 0]])
    parity_op = np.array([[1, 0], [0, -1]])

    def __init__(self, geometry: geom.Geometry):
        """
        Base class for exact diagonalizations. Classes for Bosons and Fermions will inherit from this class.

        In the case of spins, the basis states are tensor product states. We can write these as vectors, denoting
        up spins by "1" and down spins by "0". Therefore, the vector [0, 0, 0, 1] represents down spins on the first
        three logical sites, and an up spin on the fourth site. The tensor product gives a natural order to the basis
        states, and the first few states are
        [0, 0, 0, 0]
        [0, 0, 0, 1]
        [0, 0, 1, 0]
        [0, 0, 1, 1]
        ...

        In the case of bosons, we need only expand the allowed numbers in the vector. However, in this case it will
        be necessary to truncate each site to only allow a finite number of bosons. If we restrict each site to have
        no more than two, then the basis states would be

        .. math::

          (0, 0, 0, 0) &= \\left | 0 \\right \\rangle

          (0, 0, 0, 1) &= a_4^\\dagger \\left | 0 \\right \\rangle

          (0, 0, 0, 2) &= a_4^\\dagger a_4^\\dagger \\left | 0 \\right \\rangle

          (0, 0, 1, 0) &= a_3^\\dagger \\left | 0 \\right \\rangle

          (0, 0, 2, 0) &= a_3^\\dagger a_3^\\dagger \\left | 0 \\right \\rangle

          (0, 0, 2, 1) &= a_4^\\dagger a_3^\\dagger a_3^\\dagger \\left | 0 \\right \\rangle

        and so on

        For Fermions (and bosons) basis states are combinations of creation operators acting on the vacuum
        state. For Fermions, the order of these creation operators matters. In that case, creation operators are
        organized first by spin index, with higher spin indexes further to the left, and then by site indexing, with
        higher site index further to the left within each spin index.

        Having established the order of creation operators within a basis state, we can now write our fermion
        occupations like spins, using a vector like [0, 0, 0, 1] to mean that there are no fermions on the
        first three logical sites, and one fermion on the fourth. The order of these vectors will mimic the
        order of the tensor product state We take the leftmost element of the vector to be the 'smallest' index.
        We say that the 'smallest' index has the lowest site number and the smallest spin (i.e. most down spin).
        Note that there is no necessary connection between the ordering of the basis states and the
        ordering of creation operators within a basis state.

        For example, consider a fermions model on two sites. The spin-like vectors specifying the basis states are
        [site 0 spin down, site 1 spin down, site 0 spin up, site 1 spin up], and the first few basis states in order are

        .. math ::

          (0, 0, 0, 0) &= |0>

          (0, 0, 0, 1) &= c_{1,\\uparrow}^\\dagger |0>

          (0, 0, 1, 0) &= c_{0,\\uparrow}^\\dagger |0>

          (0, 0, 1, 1) &= c_{1,\\uparrow}^\\dagger c_{0,\\uparrow}^\\dagger |0>

        """

        self.check_version()
        self.geometry = geometry
        self.nstates = self.nbasis ** (self.geometry.nsites * self.nspecies)
        self.eye_onsite = np.eye(self.nbasis)

    def check_version(self):
        """
        Check which version of python is being used.

        :return:
        """

        if ctypes.sizeof(ctypes.c_voidp) == 4:
            print("Warning: using 32-bit python. Array size is limited. Recommend switching to 64-bit")

    def get_state_vects(self, print_results: bool = False):
        pass

    # ########################
    # Funtions to construct transformation operators
    # ########################

    def get_xform_op(self, cycles, print_results: bool = False):
        """
        Get transformation operator acting on our space (in same basis as Hamiltonian). Construct this by
        decomposing the transformation into cycles, and decomposing cycles into swap operations.

        :param cycles: A list of lists. Each sub-list is an ordered collection of sites, where one site transforms
          into the next under the actaion of whichever symmetry transformation we are using. Determine cycles using the
          ed_geometry package function findSiteCycles
        :param print_results: Whether to print results
        :return: Sparse matrix representing transformation operator in Hilbert space.
        """
        if print_results:
            tstart = time.perf_counter()

        trans_op = 1
        for cycle in cycles:
            if len(cycle) > 1:
                # loop over species
                for ii in range(0, self.nspecies):
                    cycle_op = 1
                    for jj in range(len(cycle) - 1, 0, -1):
                        cycle_op = self.get_swap_op(cycle[jj - 1], cycle[jj], ii).tocsr() * cycle_op
                    trans_op = cycle_op * trans_op

        if print_results:
            tend = time.perf_counter()
            print("get_xform_op op took %0.2f s" % (tend - tstart))  # TODO find way to print a name

        return trans_op

    # ########################
    # Build and diagonalize H
    # ########################

    def createH(self):
        """
        Create the Hamiltonian of the system
        :return:
        """
        pass

    def diagH(self,
              haml,
              neigs: int = 200,
              use_sparse: bool = False,
              print_results: bool = False):
        """
        Diagonalize Hamiltonian, or, if using spares matrices, get neigs eigenvalues and eigenvectors

        :param haml: Hamiltonian of the system as a sparse matrix.
        :param neigs: Optional, number of eigenvalues to compute.
        :param use_sparse:
        :param print_results:
        :return: (eigvals, eigvects)
        """
        if print_results:
            tstart = time.perf_counter()
        if use_sparse:
            # from scipy.sparse.linalg import eigsh
            # look for eigenvalues nearest the all ground state.
            # 'LM' is by far the fastest...but adding number to runner_offset seems to slow things back down.
            offset = 100
            # runner_offset = -Energy_AllGnd
            # eigvals_temp,eigvects_temp = eigsh(self.H+runner_offset*sp.eye(self.NStates).tocsr(),neigs,which='SM')
            eigvals_temp, eigvects_temp = scipy.sparse.linalg.eigsh(haml + offset * sp.eye(haml.shape[0]).tocsr(), neigs, which='LM')
            eigvals_temp = eigvals_temp - offset
            eigvals = eigvals_temp
            eigvects = eigvects_temp
        else:
            # from numpy.linalg import eigh
            eigvals, eigvects = np.linalg.eigh(haml.toarray())

        sorted_is = np.argsort(eigvals)
        eigvals = eigvals[sorted_is].real
        eigvects = eigvects[:, sorted_is]

        if print_results:
            tend = time.perf_counter()
            print("Diagonalizing H of size %d x %d took %0.2f s" % (haml.shape[0], haml.shape[1], tend - tstart))

        return eigvals, eigvects

    def quench_time_evolve(self,
                           initial_states,
                           eig_vects,
                           eig_energies,
                           times,
                           print_results: bool = False):
        """
        Do quench time evolution in subspace we've diagonalized. Take initial state and times to compute quench at
        as arguments. Return Times and resulting states, which is an NStates x NTimes array.

        :param initial_states:
        :param eig_vects:
        :param eig_energies:
        :param times:
        :param print_results:
        :return: Times, evolved_states
        """

        # TODO: implement this with scipy.sparse.linalg.expm??? Maybe not in this function, because that is mostly
        # useful if changing hamiltonian each timestep
        if print_results:
            tstart = time.perf_counter()
        if initial_states.ndim == 1:
            # initial_states = initial_states[:, None]
            initial_states = np.expand_dims(initial_states, axis=1)

        num_init_states = initial_states.shape[1]
        if isinstance(times, (int, float)):
            times = np.array([times])

        # first project into the subspace we've diagonalized...
        # note that P = EigVects.transpose() projects to the eigenbasis
        # init_states_eig_basis = EigVects.transpose().dot(InitialState)
        init_states_eig_basis = eig_vects.transpose().conj().dot(initial_states)
        # can also see what is left over after projeccting and then unprojecting.
        init_states_projected_back = eig_vects.dot(init_states_eig_basis)

        # TODO make this part work for multiple states!
        init_state_fraction_in_basis = self.get_norms(init_states_projected_back)
        # init_state_fraction_in_basis = np.sum(np.multiply(init_states_projected_back.conj(),init_states_projected_back).real,0)

        if print_results:
            print("Projection of initial states onto diagonalized subspace had minimum norm %0.3f" %
                  float(np.min(init_state_fraction_in_basis)))
        # normalize state...but of course doesn't make much sense if the state isn't nearly all in the subspace.
        # init_states_projected_back = init_states_projected_back/np.sqrt(Norm) #can add this back in with some manipulation,
        # but maybe not necessary?

        # do time evolution
        # this loop can probably be removed, to speed this up.
        # should also think about adding ability to process multiple states simultaneously
        expanded_times, expanded_energies = np.meshgrid(times, eig_energies)
        # expanded_times = np.tile(expanded_times[:, :, None], (1, 1, num_init_states))
        expanded_times = np.tile(np.expand_dims(expanded_times, axis=2), (1, 1, num_init_states))
        expanded_energies = np.tile(np.expand_dims(expanded_energies, axis=2), (1, 1, num_init_states))
        # need to get this in a slightly different shape. 0-, 1-Times, 2-States
        init_states_expanded = np.tile(np.expand_dims(init_states_eig_basis, axis=1), (1, times.size, 1))
        # do time evolution
        evolved_states_eig_basis = np.multiply(np.exp(-1j * np.multiply(expanded_times, expanded_energies)),
                                               init_states_expanded)
        # transform back to initial basis. Need to multiply a stack of matrices by the same matrix. So want to do
        # matrix multiplication along the first two dimensions (0 and 1) and ignore the 3rd dimension (2)
        evolved_states = np.tensordot(eig_vects, evolved_states_eig_basis, axes=([1], [0]))
        evolved_states = np.squeeze(evolved_states)

        if evolved_states.ndim == 1:
            # evolved_states = evolved_states[:, None]
            evolved_states = np.expand_dims(evolved_states, axis=1)

        if print_results:
            tend = time.perf_counter()
            print("Time evolving %d state(s) for %d time point(s) took %0.2f s" % (
                num_init_states, times.size, tend - tstart))

        return times, evolved_states

    # ########################
    # Calculate operators
    # ########################

    def spinful2spinlessIndex(self,
                              spinful_site_index: int,
                              n_physical_sites: int,
                              spin_index: int) -> int:
        """
        Convert from spinful fermion indexing to spinless fermion indexing. i.e. convert from double index to single
        index. Assumes that you are using the Fermion basis where creation operators are organized first by spin index,
        with higher spin indexes further to the left, and then by site indexing, with higher site index further to the
        left within each spin index.

        :param spinful_site_index: Int, site index = {0,...nsites-1}
        :param n_physical_sites: Int, total number of physical sites
        :param spin_index: Int, spin state. |Down> = 0, |Up> = 1. Can also have more spin states.
        :return: "spinless" index
        """
        return n_physical_sites * spin_index + spinful_site_index

    def spinless2spinfullIndex(self,
                               spinless_site_index: int,
                               n_physical_sites: int) -> (int, int):
        """
        Convert from spinless fermion indexing to spinful fermion indexing. i.e. convert from single index to double
        index. Assumes that you are using the Fermion basis where creation operators are organized first by spin index,
        with higher spin indexes further to the left, and then by site indexing, with higher site index further to the
        left within each spin index.

        :param spinless_site_index: spinless site index
        :param n_physical_sites: number of physical sites
        :return: (spinfull site index, spin index)
        """
        site_index = np.mod(spinless_site_index, n_physical_sites)
        spin_index = (spinless_site_index - site_index) / n_physical_sites
        return site_index, spin_index

    def get_single_site_op(self,
                           site_index: int,
                           species_index: int,
                           op,
                           format: str = "fermion"):
        """
        Compute operator for a single site.

        :param int site_index: site to compute operator on
        :param int species_index:
        :param op: 2x2 operator, acting on a single site. e.g. Sz
        :param format:
        :return: sparse matrix
        """
        n_logical_sites = self.geometry.nsites * self.nspecies
        site_index_logical = self.spinful2spinlessIndex(site_index, self.geometry.nsites, species_index)

        if format == "fermion":
            endmat = sp.eye(1, 1, format="coo")
            for ii in range(0, n_logical_sites - site_index_logical - 1):
                endmat = sp.kron(endmat, self.parity_op, format="coo")
            # TODO: can I generate this parity matrix without a loop? Would it be faster?
        elif format == "boson":
            endmat = sp.eye(2 ** (n_logical_sites - site_index_logical - 1))
        else:
            raise Exception()

        return sp.kron(sp.kron(sp.eye(self.nbasis ** site_index_logical), op, "coo"), endmat, "coo")

    def get_two_site_op(self,
                        site1: int,
                        species1: int,
                        site2: int,
                        species2: int,
                        op1,
                        op2,
                        format: str = "fermion"):
        """
        Compute product of operators on two separate sites. First must sort the operators so that the first operator
        is the lower site.

        :param site1: first site
        :param species1:
        :param site2:
        :param species2:
        :param op1: 2x2 operator acting on site1
        :param op2: 2x2 operator acting on site2
        :param format:
        :return:
        """
        n_logical_sites = self.geometry.nsites * self.nspecies
        site1_index_logical = self.spinful2spinlessIndex(site1, self.geometry.nsites, species1)
        site2_index_logical = self.spinful2spinlessIndex(site2, self.geometry.nsites, species2)

        if site1_index_logical == site2_index_logical:
            # most obvious form, but is slower because requires multiplication
            # return self.getSingleSiteOp(site1_index, n_logical_sites, op1, format = format) * \
            #        self.getSingleSiteOp(site2_index, n_logical_sites, op2, format = format)

            # This should always be bosonic...the fermionic-ness on the same site is already conatained in the on-site
            # operators
            return self.get_single_site_op(site1, species1, op1.dot(op2), format="boson")
        else:
            # For fermions the order still matters, so keep track of that.
            sites = [site1_index_logical, site2_index_logical]
            site_a = min(sites)
            site_b = max(sites)
            min_ind = sites.index(site_a)
            if min_ind == 0:
                op_a = op1
                op_b = op2
                if format == "fermion":
                    op_b = self.parity_op.dot(op_b)
            else:
                op_a = op2
                op_b = op1
                if format == "fermion":
                    op_b = op_b.dot(self.parity_op)

            if format == "fermion":
                middlemat = sp.eye(1, 1, format="coo")
                for ii in range(0, site_b - site_a - 1):
                    middlemat = sp.kron(middlemat, self.parity_op, "coo")
            elif format == "boson":
                middlemat = sp.eye(self.nbasis ** (site_b - site_a - 1))
            else:
                raise Exception()
                # TODO: can I generate this parity matrix without a loop? Would it be faster?
            return (sp.kron(sp.kron(sp.kron(sp.eye(self.nbasis ** site_a), op_a, "coo"), middlemat, "coo"),
                            sp.kron(op_b, sp.eye(self.nbasis ** (n_logical_sites - site_b - 1)), "coo"), "coo"))

    def get_sum_op(self,
                   op,
                   species,
                   format: str = "fermion",
                   print_results: bool = False):
        """
        Get operator that is the sum of identical single site operators

        :param op: 2x2 operator acting on each site
        :param species: index of species operators act on
        :param format: "fermion" or "
        :param print_results:
        :return opmat: sparse matrix representing the sum operator
        """

        if print_results:
            tstart = time.perf_counter()

        opmat = 0
        for ii in range(0, self.geometry.nsites):
            opmat = opmat + self.get_single_site_op(ii, species, op, format=format)
        if print_results:
            tend = time.perf_counter()
            print("getSumOptook %0.2f s" % (tend - tstart))
        return opmat.tocsr()

    def get_sum_op_q(self,
                     species,
                     q_vector,
                     op,
                     format: str = "fermion",
                     print_results: bool = False):
        """

        :param species:
        :param q_vector:
        :param op:
        :param format:
        :param print_results:
        :return:
        """

        if print_results:
            tstart = time.perf_counter()

        qx = q_vector[0]
        qy = q_vector[1]
        sum_op_q = 0
        for ii in range(0, self.geometry.nsites):
            sum_op_q = sum_op_q + np.exp(1j * self.geometry.xlocs[ii] * qx + 1j * self.geometry.ylocs[ii] * qy) * \
                       self.get_single_site_op(ii, species, op, format=format)

        if print_results:
            tend = time.perf_counter()
            print("get_sum_op_q %0.2f s" % (tend - tstart))

        return sum_op_q

    def get_swap_op(self,
                    site1: int,
                    site2: int,
                    species: int):
        """

        :param site1:
        :param site2:
        :param species:
        :return:
        """
        pass

    # ##################################
    # General overlaps/expectation value functions
    # ##################################

    def get_exp_vals(self,
                     states,
                     full_op,
                     print_results: bool = False) -> np.ndarray:
        """
        Compute the expectation values for a number of states and a single operator. The operator and states must be
        written in the same basis. This may be the basis of the full space, or a projected subspace.

        :param states: NumPy array  or sparse matrix of size nbasis x nvectors. The columns represent state vectors.
        :param full_op: Sparse matrix of size nbasis x nbasis
        :param print_results: If true print information to terminal
        :return: NumPy array of size ???
        """
        if print_results:
            tstart = time.perf_counter()

        if states.ndim == 1:
            # TODO: can rewrite with reshape...
            # states = states[:, None]
            states = np.expand_dims(states, axis=1)

        if sp.issparse(states):
            states = states.tocsr()  # transpose converts from csc to csr
            exp_vals = sp.csr_matrix.sum(sp.csr_matrix.multiply(states.conj(), full_op.dot(states)), 0).real
        else:
            exp_vals = np.sum(np.multiply(states.conj(), full_op.dot(states)), 0).real
        if print_results:
            tend = time.perf_counter()
            print("get_exp_vals took %0.2f s" % (tend - tstart))

        return exp_vals

    def get_overlaps(self,
                     states1,
                     states2,
                     print_results: bool = False):
        """
        Compute the overlap of two collections of states.

        :param states1: NumPy array or sparse matrix of size nbasis x number of 1 states
        :param states2: NumPy array or sparse matrix of size nbasis x number of 2 states
        :param print_results:
        :return: NumPy array of size number of 1 states x number of 2 states. M[i,j] = <states1[i]|states2[j]>.
        """
        if print_results:
            tstart = time.perf_counter()

        if states1.ndim == 1:
            # states1 = states1[:, None]
            states1 = np.expand_dims(states1, axis=1)

        if states2.ndim == 1:
            # states2 = states2[:, None]
            states2 = np.expand_dims(states2, axis=1)

        if sp.issparse(states1):
            states1 = states1.tocsc()
            overlaps = sp.csr_matrix.dot(states1.conj().transpose(), states2)

        elif sp.issparse(states2):
            states2 = states2.tocsc()
            overlaps = sp.csr_matrix.dot(states2.conj().transpose(), states1).conj().transpose()
        else:
            # changed order for more sensible interpretation
            overlaps = states1.conj().transpose().dot(states2)

        if print_results:
            tend = time.perf_counter()
            print("get_overlaps took %0.2f s" % (tend - tstart))

        return np.squeeze(overlaps)

    def get_norms(self,
                  states,
                  print_results: bool = False) -> np.ndarray:
        """
        Compute the norms for a collction of states

        :param states: NumPy array or sparse matrix of size nbasis x numvectors. Columns represent state vectors.
        :param print_results:
        :return: NumPy array of size numvectors, giving the norms of each column of states
        """
        if print_results:
            tstart = time.perf_counter()

        if states.ndim == 1:
            # states = states[:, None]
            states = np.expand_dim(states, axis=1)

        if sp.issparse(states):
            states = states.tocsr()
            norms = sp.csr_matrix.sum(sp.csr_matrix.multiply(states.conj(), states), 0).real
        else:
            norms = np.sum(np.multiply(states.conj(), states), 0).real
        if print_results:
            tend = time.perf_counter()
            print("get_norms took %0.2f s" % (tend - tstart))

        return norms

    # ##################################
    # utility functions for correlators
    # ##################################
    def get_exp_vals_sites(self,
                           states,
                           site_op,
                           species,
                           sites=None,
                           projector=None,
                           format: str = "fermion"):
        """
        Get single-site operator expectation value given sites

        :param states: states to compute correlators
        :param site_op: single site operator
        :param species: species to use
        :param sites: if no argument is supplied, do computation at all sites
        :param projector:
        :param format:
        :return: (exp_vals, sites)
        """
        if states.ndim == 1:
            # states = states[:, None]
            states = np.expand_dims(states, axis=1)

        if projector is None:
            projector = sp.eye(states.shape[0])

        if sites is None:
            sites = range(0, self.geometry.nsites)
        sites = np.asarray(sites)

        exp_vals = np.zeros((len(sites), states.shape[1]))
        for ii, site in enumerate(sites):
            full_op = projector * self.get_single_site_op(site, species, site_op, format=format) * \
                      projector.conj().transpose()
            exp_vals[ii, :] = self.get_exp_vals(states, full_op)

        return exp_vals, sites

    def get_thermal_exp_sites(self,
                              eig_vects,
                              eig_vals,
                              site_op,
                              species,
                              temps,
                              sites=None,
                              projector=None,
                              format: str = "fermion",
                              print_results: bool = False):
        """

        :param eig_vects:
        :param eig_vals:
        :param site_op:
        :param species:
        :param temps:
        :param sites:
        :param projector:
        :param format:
        :param print_results:
        :return:
        """
        temps = np.asarray(temps, dtype=float)

        if projector is None:
            projector = sp.eye(eig_vects.shape[0])

        if sites is None:
            sites = range(0, self.geometry.nsites)
        sites = np.asarray(sites)

        exp_vals = np.zeros((len(sites), temps.size))
        for ii, site in enumerate(sites):
            full_op = projector * self.get_single_site_op(site, species, site_op, format=format) * projector.conj().transpose()
            exp_vals[ii, :] = self.get_exp_vals_thermal(eig_vects,
                                                        full_op,
                                                        eig_vals,
                                                        temps,
                                                        print_results=print_results)

        return exp_vals, sites

    def get_corr_sites(self,
                       states,
                       species1,
                       species2,
                       op1,
                       op2,
                       sites1=None,
                       sites2=None,
                       projector=None,
                       format: str = "fermion"):
        """
        Compute correlators between sites1 and sites2
        :param states: States to compute correlators at
        :param species1: species to associated with site1s
        :param species2: species associated with site2s
        :param op1: operator associated with site1s
        :param op2: operator associated with site2s
        :param sites1: site1s. If no argument is provided, will generate all possible site1s
        :param sites2: site2s If no argument is provided, will generate all possible site2s.
        :param projector: Projector to apply to expectation values
        :param format: "boson" or "fermion" depending on type of operator
        :return: (exp_vals, sites1, sites2)
        """
        if states.ndim == 1:
            # states = states[:, None]
            states = np.expand_dims(states, axis=1)

        if projector is None:
            projector = sp.eye(states.shape[0])

        if sites1 is None and sites2 is None:
            xx, yy = np.meshgrid(range(0, self.geometry.nsites), range(0, self.geometry.nsites))
            sites1 = xx[xx >= yy]
            sites2 = yy[xx >= yy]
        sites1 = np.asarray(sites1)
        sites2 = np.asarray(sites2)

        exp_vals = np.zeros((len(sites1), states.shape[1]))
        for ii in range(0, sites1.size):
            full_op = projector * self.get_two_site_op(sites1[ii], species1, sites2[ii], species2, op1,
                                                       op2, format=format) * projector.conj().transpose()
            exp_vals[ii, :] = self.get_exp_vals(states, full_op)

        return exp_vals, sites1, sites2

    def get_thermal_corr_sites(self,
                               eig_vects,
                               eig_vals,
                               species1,
                               species2,
                               op1,
                               op2,
                               temps,
                               sites1=None,
                               sites2=None,
                               projector=None,
                               format: str = "fermion",
                               print_results: bool = False):
        """

        :param eig_vects:
        :param eig_vals:
        :param species1:
        :param species2:
        :param op1:
        :param op2:
        :param temps:
        :param sites1:
        :param sites2:
        :param projector:
        :param format:
        :param print_results:
        :return:
        """
        temps = np.asarray(temps, dtype=float)

        if projector is None:
            projector = sp.eye(eig_vects.shape[0])

        if sites1 is None and sites2 is None:
            xx, yy = np.meshgrid(range(0, self.geometry.nsites), range(0, self.geometry.nsites))
            sites1 = xx[xx >= yy]
            sites2 = yy[xx >= yy]

        sites1 = np.asarray(sites1)
        sites2 = np.asarray(sites2)

        exp_vals = np.zeros((len(sites1), temps.size))
        for ii in range(0, sites1.size):
            full_op = projector * self.get_two_site_op(sites1[ii], species1, sites2[ii], species2, op1, op2,
                                                       format=format) * projector.conj().transpose()
            exp_vals[ii, :] = self.get_exp_vals_thermal(eig_vects, full_op, eig_vals, temps, print_results=print_results)

        return exp_vals, sites1, sites2

    # ##################################
    # Subspace overlaps
    # ##################################

    def get_subspace_overlaps(self,
                              states,
                              projectors,
                              print_results: bool = False):
        """
        Given a list of projectors and a matrix of states, compute the overlaps of each state with the subspace
        defined by each projector.

        :param states: NumPy array or sparse matrix, size nbasis x numvectors. Each column represents a state vector.
        :param projectors:
        :param print_results: Boolean, if true print results to terminal
        :return: projections, NumPy array of size ????
        """

        if print_results:
            tstart = time.perf_counter()
        if states.ndim == 1:
            states = states[:, None]

        if not isinstance(projectors, list):
            projectors = [projectors]

        projections = np.zeros([len(projectors), states.shape[1]])
        # the weight of the projected state is the same as the expectation value of the transpose of the
        # projector times the projector
        for ii in range(0, len(projectors)):
            projections[ii, :] = self.get_exp_vals(states, projectors[ii].conj().transpose().dot(projectors[ii]))
        if print_results:
            tend = time.perf_counter()
            print("get_subspace_overlaps took %0.2f s" % (tend - tstart))
        return np.squeeze(projections)

    def get_subspace_projs(self,
                           diagonal_op,
                           eig_vals=None,
                           print_results: bool = False):
        """
        Given a diagonal operator, get projection onto subspace with a given eigenvalue. For example, you can use
        this function in combination with get_npairs_op to create an operator that projects onto the subspace of all
        states with a single pair of rydberg atoms.

        :param diagonal_op:
        :param eig_vals:
        :param print_results:
        :return:
        """
        if print_results:
            tstart = time.perf_counter()

        if not sp.isspmatrix_csr(diagonal_op):
            raise Exception("diagonal_op in get_subspace_projs was not csr_matrix")

        opvector = sp.csr_matrix.diagonal(diagonal_op)
        # nstates = opvector.shape[0]
        nstates = opvector.size

        if eig_vals is None:
            eig_vals = np.unique(opvector)

        # if isinstance(eig_vals, (int, float)):
        #     eig_vals = np.array([eig_vals])
        eig_vals = np.array(eig_vals)
        if eig_vals.ndim == 0:
            eig_vals = eig_vals[None]

        projs = []
        for ii in range(0, eig_vals.size):
            indices = np.arange(0, opvector.size)
            indices = indices[opvector == eig_vals[ii]]
            projs.append(sp.csr_matrix((np.ones(indices.size), (np.arange(0, indices.size), indices)),
                                       shape=(indices.size, nstates)))
        if print_results:
            tend = time.perf_counter()
            print("get_subspace_projs took %0.2f s" % (tend - tstart))

        return projs, eig_vals

    # ##################################
    # finite temperature properties
    # ##################################

    def get_exp_vals_thermal(self,
                             eig_vects,
                             op,
                             eig_vals,
                             temps,
                             print_results: bool = False):
        """
        Calculate thermal expectation values

        :param eig_vects:
        :param op:
        :param eig_vals:
        :param temps:
        :param print_results:
        :return:
        """
        if print_results:
            tstart = time.perf_counter()
        # TODO: make capable of handling a vector of temperatures at once
        temps = np.asarray(temps, dtype=float)
        if temps.size > 1:
            thermal_exp_vals = np.zeros(temps.size)
            for ii in range(0, temps.size):
                thermal_exp_vals[ii] = self.get_exp_vals_thermal(eig_vects, op, eig_vals, temps[ii],
                                                                 print_results=print_results)

        else:
            thermal_weights, z = self.get_thermal_weights(eig_vals, temps, use_energies_offset=True)
            # try to be efficient by not bothering to compute results if weights are really zero
            if temps != 0:
                exp_vals = self.get_exp_vals(eig_vects, op)
                thermal_exp_vals = np.sum(np.multiply(exp_vals, thermal_weights)) / z
            else:
                nstates = len(thermal_weights[thermal_weights != 0])
                exp_vals = self.get_exp_vals(eig_vects[:, 0:nstates], op)
                thermal_exp_vals = np.sum(exp_vals) / z

        if print_results:
            tend = time.perf_counter()
            print("get_exp_vals_thermal for %d eigenvectors, %d x %d matrix and %d temps took %0.2f s" %
                  (eig_vects.shape[0], op.shape[0], op.shape[0], temps.size, tend - tstart))

        return thermal_exp_vals

    def get_thermal_weights(self,
                            energies,
                            temps,
                            use_energies_offset: bool = False):
        """
        Get partition function and thermal weights for a set of energies at given temperature.
        The probability have a having a given state is then thermal_weight / Z.
        Correctly accounts for degenerate states.

        :param energies: numpy array of energies. Assume energies are sorted
        :param temps: temperatures at which to compute thermal weights
        :param use_energies_offset: if 1 will first shift the lowest energy eigenvalue to occur at zero. Otherwise,
          will not
        :return:
        """

        # TODO: error if energies not sorted

        energies = np.asarray(energies)
        temps = np.asarray(temps)

        if temps.size > 1:
            thermal_weights = np.zeros((energies.size, temps.size))
            z = np.zeros(temps.size)
            for ii in range(0, temps.size):
                thermal_weights[:, ii], z[ii] = self.get_thermal_weights(energies, temps[ii], use_energies_offset)

        else:
            if use_energies_offset:
                energies = energies - energies[0]

            if temps != 0:
                # finite temperature
                beta = 1 / temps
                thermal_weights = np.exp(-beta * energies)
                z = np.sum(thermal_weights)

            else:
                # zero temperature
                # Check for degenerate states
                nstates = len(energies[np.round(energies, 13) == 0])
                thermal_weights = np.zeros(energies.size)
                thermal_weights[0:nstates] = 1.
                z = float(nstates)

        return thermal_weights, z

    def get_partition_fn_hamiltonian(self,
                                     hamiltonian,
                                     beta: float):
        """
        Compute partition function from Hamiltonian via trace

        :param hamiltonian:
        :param beta:
        :return:
        """
        # this works if hamiltonian not diagonalized, but is slow
        if beta == 0:
            return hamiltonian.shape[0]
        else:
            return np.trace(scipy.linalg.expm(- beta * hamiltonian).toarray().real)

    def get_partition_fn_sectors(self,
                                 energies_sectors: list,
                                 temp: np.ndarray) -> np.ndarray:
        """
        Get partition functions for each symmetry sector of our problem from the energies.

        This way we can shift the smallest eigenvalue to zero which simplifies numerical issues (i.e. too large or
        small values in the exponential).

        :param energies_sectors: a list of numpy arrays. Each array gives the eigenvalues for a sub-block of the
          Hamiltonian
        :param temp: temperature
        :return: z_sectors, a numpy array where each entry is the partition function of the given sub-block
        """
        temp = np.asarray(temp)

        # TODO: can this be vectorized?
        if temp.size > 1:
            # if more than one temperature, loop through temperatures and compute for each one
            z_sectors = []
            for ii in range(0, temp.size):
                z_sector_curr = self.get_partition_fn_sectors(energies_sectors, temp[ii])
                z_sectors.append(z_sector_curr)

        else:
            # for each temperature
            min_eigs_sectors = [np.min(es) for es in energies_sectors]
            min_eig = np.min(min_eigs_sectors)
            eigs_all = np.sort(np.concatenate(energies_sectors))

            # thermal averaging over sectors
            z_sectors = np.zeros(len(energies_sectors))

            # if temp == 0:
            #     # TODO: isn't this wrong if we have degenerate eigenvalues?
            #     #sorted_i = np.argsort(min_eigs_sectors)
            #     #z_sectors[sorted_i[0]] = 1
            #     #z_sectors[sorted_i[1:]] = 0
            #
            #     # still not right, bc sectors may contain more than one degenerate eigenvalue...
            #     # _, z = get_thermal_weights()
            #     z_sectors[ np.round(min_eigs_sectors - min_eig, 14) == 0 ] = 1
            # else:
            #     z_sectors[:] = np.array([np.sum(np.exp(-(ev - eigs_all[0]) / temp)) for ev in energies_sectors])

            for ii, e_sector in enumerate(energies_sectors):
                _, z_sectors[ii] = self.get_thermal_weights(e_sector - min_eig, temp, use_energies_offset=False)

        return z_sectors

    def thermal_avg_combine_sectors(self,
                                    expvals_sectors: list,
                                    eigvals_sectors: list,
                                    temps: np.ndarray) -> np.ndarray:
        """
        Get the thermal expectation value for a Hamiltonian diagonalized on multiple sectors
        TODO: is there a less confusing way to do this???

        :param expvals_sectors: should be of size n_sectors x n_temps x arbitrary size, where the
          hamiltonian is divided into n_sector sub-blocks and we have evaluated the expectation values
          for each sub-block at n_temps temperatures.
        :param eigvals_sectors: a list of numpy arrays, where each array gives all of the eigenvalues in a single
          sub-block of the Hamiltonian
        :param temps: numpy array of temperatures
        :return: thermal_exp_vals
        """
        temps = np.asarray(temps)

        # TODO: 'vectorize' this to take multiple temperatures and compute in efficient way if possible...
        #  In that case, quantity_sectors would be a nsectors x ntemps array
        if temps.size > 1:

            # so we can work on arbitrary extra dimensions, need to have at least a singleton third dimension
            if expvals_sectors.ndim == 2:
                # expvals_sectors = expvals_sectors[:, :, None]
                expvals_sectors = np.expand_dims(expvals_sectors, axis=2)
            # get shape for all except temperature
            shape = list(expvals_sectors.shape)
            shape.pop(1)

            thermal_exp_vals = np.zeros(([temps.size] + shape[1:]))
            for ii in range(0, temps.size):
                # expvals_sectors_temp = np.zeros((shape))
                expvals_sectors_temp = expvals_sectors[:, ii, :]
                thermal_exp_vals[ii, :] = self.thermal_avg_combine_sectors(expvals_sectors_temp,
                                                                           eigvals_sectors, temps[ii])
        else:
            z_sector = self.get_partition_fn_sectors(eigvals_sectors, temps)
            z = np.sum(z_sector)
            z_sector_expanded = z_sector
            for ii in range(1, expvals_sectors.ndim):
                z_sector_expanded = np.repeat(np.expand_dims(z_sector_expanded, ii), expvals_sectors.shape[ii], ii)

            thermal_exp_vals = np.nansum(np.multiply(expvals_sectors, z_sector_expanded), 0) / z

        return thermal_exp_vals

    def get_matrix_elems(self,
                         eig_vects,
                         op,
                         print_results: bool = False):
        """

        :param eig_vects:
        :param op:
        :param print_results:
        :return:
        """

        if print_results:
            tstart = time.perf_counter()

        # a, entries = np.meshgrid(eig_vals, eig_vals)
        # omegas = entries - a #energy difference between eigenvalues for matrix element
        matrix_elems = eig_vects.conj().transpose().dot(op.dot(eig_vects))

        if print_results:
            tend = time.perf_counter()
            print("get_matrix_elems for %dx%d matrix took %0.2f s" % (op.shape[0], op.shape[0], tend - tstart))

        return matrix_elems

    def get_response_fn_retarded(self,
                                 a_matrix_elems,
                                 b_matrix_elems,
                                 eig_vals,
                                 temperature,
                                 format: str = "boson",
                                 print_results: bool = False):
        """
        Create a function for computing the optical conductivity at arbitrary frequency omega_start and arbitrary broadening
        paramter eta

        :param a_matrix_elems: the current operator in the eigenvector basis
        :param b_matrix_elems:
        :param eig_vals: eigenvalues corresponding to the eigenvectors
        :param temperature:
        :param format:
        :param print_results:
        :return:
        """
        if print_results:
            tstart = time.perf_counter()

        if format == "boson":
            factor = -1
        elif format == "fermion":
            factor = 1
        else:
            raise Exception("wrong format string supplied to get_response_fn_retarded")

        # normalized lorentzian function. Area is constant with changing eta
        # expected to need a factor of 1/pi here to normalize lorentzian area to 1
        def lor(w, eta, amp): return (amp * eta / np.pi) / (eta ** 2 + np.square(w))
        temperature = float(temperature)
        eig_vals = eig_vals - eig_vals[0]

        [a, b] = np.meshgrid(eig_vals, eig_vals)
        omegas = b - a  # M_nm = E_n - E_m
        if temperature != 0:
            beta = 1 / temperature
            z = np.sum(np.exp(- beta * eig_vals))
            mat_elem = np.multiply(a_matrix_elems, b_matrix_elems.transpose())
            thermal_mat_elem = mat_elem.dot(np.diag(np.exp(- beta * eig_vals)))
            omega_fn = lambda wo, eta, es, amp: np.multiply(1 + factor * np.exp(beta * es), lor(wo + es, eta, amp))

        else:
            z = 1
            # for period_start = 0, we have two terms, corresponding to |n> = |0> and |m> = |0>
            # one of these terms gives all of the positive frequency information (i.e. delta(\omega_start - E_n + E_o))
            # and the other gives all of the negative frequency information
            # the positive frequency term has matrix element factor A_{0n}B_{n0}, and the negative frequency term
            # has A_{n0}B_{0n}
            omegas = np.concatenate((-eig_vals, eig_vals))
            thermal_mat_elem = np.concatenate((np.multiply(a_matrix_elems[0, :], b_matrix_elems[:, 0]),
                                               factor * np.multiply(a_matrix_elems[:, 0], b_matrix_elems[0, :])))
            omega_fn = lambda wo, eta, es, amp: lor(wo + es, eta, amp)

        max_imag = np.abs(thermal_mat_elem.imag).max()
        if max_imag > 1e-16:
            print("warning, maximum imaginary part of Anm*Bmn = %g" % max_imag)
            print("maximum real part of Anm*Bmn = %g" % np.abs(thermal_mat_elem.real).max())

        thermal_mat_elem = thermal_mat_elem.real
        omegas = omegas.flatten()
        thermal_mat_elem = thermal_mat_elem.flatten()

        response_fn_imag_part = lambda wo, eta: np.pi / z * np.nansum(omega_fn(wo, eta, omegas, thermal_mat_elem))

        if print_results:
            tend = time.perf_counter()
            print("get_response_fn_retarded took %0.2f s" % (tend - tstart))

        return response_fn_imag_part

    def get_response_fn_time_ordered(self,
                                     a_matrix_elems,
                                     b_matrix_elems,
                                     eig_vals,
                                     temperature,
                                     print_results: bool = False):
        """
        Create a function for computing the optical conductivity at arbitrary frequency
        omega_start and arbitrary broadening paramter eta

        :param a_matrix_elems: the current operator in the eigenvector basis
        :param b_matrix_elems: the current operator in the eigenvector basis
        :param eig_vals: eigenvalues corresponding to the eigenvectors
        :param temperature:
        :param print_results:
        :return:
        """
        if print_results:
            tstart = time.perf_counter()

        # normalized lorentzian function. Area is constant with changing eta
        # expected to need a factor of 1/pi here to normalize lorentzian area to 1
        def lor(wo, eta, w, amp): return (amp * eta / np.pi) / (eta ** 2 + np.square(w - wo))
        temperature = float(temperature)
        eig_vals = eig_vals - eig_vals[0]

        if temperature != 0:
            beta = 1 / temperature
            z = np.sum(np.exp(- beta * eig_vals))
            [a, b] = np.meshgrid(eig_vals, eig_vals)
            omegas = b - a  # M_{nm} = E_n - E_m

            thermal_mat_elem_a = np.multiply(a_matrix_elems, b_matrix_elems.transpose()).dot(
                np.diag(np.exp(- beta * eig_vals)))
            thermal_mat_elem_b = np.multiply(a_matrix_elems.transpose(), b_matrix_elems).dot(
                np.diag(np.exp(- beta * eig_vals)))

            omegas = omegas.flatten()
            thermal_mat_elem_a = thermal_mat_elem_a.flatten()
            thermal_mat_elem_b = thermal_mat_elem_b.flatten()

            omega_fn = lambda wo, eta, w, amp: lor(wo, eta, w, amp)

        else:
            z = 1
            omegas = eig_vals
            thermal_mat_elem_a = np.multiply(a_matrix_elems[:, 0], b_matrix_elems[0, :])
            thermal_mat_elem_b = np.multiply(a_matrix_elems[0, :], b_matrix_elems[:, 0])
            thermal_mat_elem_a = thermal_mat_elem_a.flatten()
            thermal_mat_elem_b = thermal_mat_elem_b.flatten()

            omega_fn = lambda wo, eta, w, amp: lor(np.abs(wo), eta, w, amp)

        response_fn_imag_part = lambda wo, eta: np.pi / z * (np.nansum(omega_fn(wo, eta, omegas, thermal_mat_elem_a)) -
                                                            (np.nansum(omega_fn(wo, eta, -omegas, thermal_mat_elem_b))))

        if print_results:
            tend = time.perf_counter()
            print("get_optical_cond_fn took %0.2f s" % (tend - tstart))

        return response_fn_imag_part
    # ########################
    # Load and save
    # ########################

    def save(self,
             file_name: str = None):
        """
        Save class as .mat file or .pkl. Defaults to .pkl

        :param file_name: String, should end in .mat or .pkl
        :return: FileName, string
        """
        if file_name is None:
            today_str = datetime.datetime.today().strftime('%Y-%m-%d_%H;%M;%S')
            file_name = "%s_ed.pkl" % today_str

        _, ext = os.path.splitext(file_name)

        if ext == '.mat':
            savemat(file_name, self.__dict__)
        elif ext == '.pkl':
            with open(file_name, 'wb') as output:
                # TODO implement savesparse. What does that mean?
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            print("Extension %s not understood" % ext)
            return ''
        print("Saved file to %s" % file_name)
        return file_name

    def load(self,
             file_name: str):
        """
        Load saved file to class. Also return as dictionary.
        :param file_name:
        :return:
        """
        # TODO come up with solution for issue that all loaded values are numpy arrays.
        # This does weird things when the code here expects integers. e.g. for NSites
        _, ext = os.path.splitext(file_name)

        if ext == '.mat':
            loaded = loadmat(file_name)
            self.__dict__ = {}
            for key, val in loaded.items():
                # if key not in self.__dict__.keys() and key[0:2]!='__':
                if key[0:2] != '__':
                    self.__dict__.update({key: val})
                else:
                    print("Skipped key %s while loading to class from file" % key)

        elif ext == '.pkl':
            with open(file_name, 'rb') as inputFile:
                loaded = pickle.load(inputFile)
            self.__dict__ = loaded.__dict__
        else:
            print("Extension %s not understood" % ext)
            loaded = {}
        return loaded
