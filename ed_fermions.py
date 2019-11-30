from __future__ import print_function
import time
import numpy as np
import scipy.sparse as sp
import ed_geometry as geom
import ed_symmetry as symm
import ed_base

# TODO: maybe I should wrap all operator functions in this class to hide the nspins arguments, since that is never
# going to change for fermions. nspins would only be visible in ed_base. This would mean that spinful2spinlessIndex
# and spinless2spinfulIndex would be called underneath the hood.

class fermions(ed_base.ed_base):

    nbasis = 2
    #nspecies = 2

    cdag_op = np.array([[0, 0], [1, 0]])
    c_op = np.array([[0, 1], [0, 0]])
    n_op = np.array([[0, 0], [0, 1]])
    swap_op = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]])

    def __init__(self, geometry, us_interspecies, ts, ns=None, mus=0, potentials=0, us_same_species=0, nspecies=2):
        """
        For Fermions (and bosons) basis states are combinations of creation operators acting on the vacuum
        state. For Fermions, the order of these creation operators matters. In that case, creation operators are
        organized first by spin index, with higher spin indexes further to the left, and then by site indexing, with
        higher site index further to the left within each spin index.

        Having established the order of creation operators within a basis state, we can now write our fermion
        occupations like spins, using a vector like [0, 0, 0, 1] to mean that there are no fermions on the first three logical
        sites, and one fermion on the fourth. The order of these vectors will mimic the order of the tensor product state
        We take the leftmost element of the vector to be the 'smallest' index. We say that the 'smallest' index has the
        lowest site number and the smallest spin (i.e. most down spin). Note that there is no necessary connection
        between the ordering of the basis states and the ordering of creation operators within a basis state.

        For example, consider a fermions model on two sites. The spin-like vectors specifying the basis states are
        [site 0 spin down, site 1 spin down, site 0 spin up, site 1 spin up], and the first few basis states in order are
        [0, 0, 0, 0]
        [0, 0, 0, 1] = c_{1,up}^\dag |0>
        [0, 0, 1, 0] = c_{0,up}^\dag |0>
        [0, 0, 1, 1] = c_{1,up}^\dag c_{0,up}^\dag |0>
        ...
        :param geometry: geometry object, specifying the number of lattice sites, their connections, and positions
        :param U: on-site interaction
        :param tunn: hopping in the x-direction
        :param ty: hopping in the y-direction
        :param n_ups: number of up spins
        :param n_dns: number of down spins
        """
        self.nspecies = nspecies
        ed_base.ed_base.__init__(self, geometry)

        # tunneling
        ts = np.asarray(ts)
        if ts.size == self.nspecies or ts.size == 1:
            self.ts = self.get_hopping_mat(ts)
        elif ts.shape == (self.geometry.nsites, self.geometry.nsites, self.nspecies):
            self.ts = ts
        elif  ts.shape == (self.geometry.nsites, self.geometry.nsites) and self.nspecies == 1:
            self.ts = ts[:, :, None]
        else:
            raise Exception("ts incorrect shape")

        if not self.ts.shape == (self.geometry.nsites, self.geometry.nsites, self.nspecies):
            raise Exception("ts matrix is inconsistent size")

        # interaction between different species
        us_interspecies = np.asarray(us_interspecies)
        if us_interspecies.size == 1:
            self.us_interspecies = us_interspecies * np.ones((self.geometry.nsites))
        else:
            self. us_interspecies = us_interspecies

        if self.us_interspecies.size != self.geometry.nsites:
            raise Exception("us_interpsecies matrix must be same size as number of sites")

        # interaction between same species
        us_same_species = np.asarray(us_same_species)
        if us_same_species.size == 1 or us_same_species.size == self.nspecies:
            self.us_same_species = self.get_samespecies_int_mat(us_same_species)
        elif us_same_species.shape == (self.geometry.nsites, self.geometry.nsites, self.nspecies):
            self.us_same_species = us_same_species
        elif us_same_species.shape == (self.geometry.nsites, self.geometry.nsites) and self.nspecies == 1:
            self.us_same_species = us_same_species[:, :, None]
        else:
            raise Exception("use_same_species incorrect shape")

        if self.us_same_species.shape != (self.geometry.nsites, self.geometry.nsites, self.nspecies):
            raise Exception("us_same_species matrix is an inconsistent size")

        # mus
        mus = np.asarray(mus)
        if mus.size == 1:
            self.mus = mus * np.ones((self.nspecies))
        elif mus.size == self.nspecies:
            self.mus = mus

        if self.mus.size != self.nspecies:
            raise Exception("mu matrix must be the same size as the number of species")

        # potential
        potentials = np.asarray(potentials)
        if potentials.size == 1:
            self.potentials = potentials * np.ones((self.geometry.nsites, self.nspecies))
        elif potentials.size == self.nspecies:
            self.potentials = np.zeros((self.geometry.nsites, self.nspecies))
            for ii in range(0, self.nspecies):
                self.potentials[ii, :] = potentials[ii] * np.ones(self.geometry.nsites)
        elif potentials.shape == (self.geometry.nsites, self.nspecies):
            self.potentials = potentials
        elif potentials.shape == (self.geometry.nsites,) and self.nspecies == 1:
            self.potentials = potentials[:, None]
        else:
            raise Exception("potential incorrect shape")

        if not self.potentials.shape == (self.geometry.nsites, self.nspecies):
            raise Exception("potentials inconsistent shape")

        # number projection
        if ns is not None:
            ns = np.asarray(ns)
            if ns.ndim == 0:
                ns = np.asarray([ns])

            if ns.size != self.nspecies:
                raise Exception("ns was inconsistent shape.")

        self.ns = ns
        # should I implement this in here? Then in all my functions I would write this as a built in projector, and
        # the other projector would be applied addtionally, and it would be assumed it was already in this basis...
        # right now I only create the operator and store it here...but it would make sense to trully integrate it into the class...
        # in that case, it should be an implementation detail that you don't need to know about externally...
        nops = []
        n_projs = []
        running_projector = sp.eye(self.nstates)
        if self.ns is not None:
            for ii in range(0, self.nspecies):
                if self.ns[ii] is None:
                    n_species_op = None
                    n_species_proj = np.array(1)
                else:
                    n_species_op = self.get_sum_op(self.n_op, ii, format="boson")
                    n_species_proj, _ = self.get_subspace_projs(running_projector.dot(n_species_op.dot(running_projector.conj().transpose())), eig_vals=self.ns[ii], print_results=0)
                    n_species_proj = n_species_proj[0]

                # concatenate projectors
                running_projector = n_species_proj.dot(running_projector)

                nops.append(n_species_op)
                n_projs.append(running_projector)

        self.n_projector = running_projector

    def get_state_vects(self, projector=None, print_results=0):
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

        n_logical_sites = self.geometry.nsites * self.nspecies
        state_spin_labels = sp.csc_matrix((self.nstates, n_logical_sites))  # csc 0.7s vs. csr 9s. As expected for vertical slicing.

        root_mat = sp.csc_matrix([[1], [0]])
        for ii in range(0, n_logical_sites):
            # simple pattern to create the columns of the matrix. It goes like this: the last column alternates as
            # 1,0,1,0,... the second to last column goes 1,1,0,0,1,1,..., all the way to the first, which goes,
            # 1,1,...,1,0,...0 (is first half ones, second half zeros).
            state_spin_labels[:, ii] = sp.kron(np.ones([2 ** ii, 1]), sp.kron(root_mat, np.ones([2 ** (n_logical_sites - ii - 1), 1])))

        #state_spin_labels.eliminate_zeros()
        state_spin_labels = 1 - state_spin_labels.toarray()

        if projector != None:
            state_spin_labels = projector * state_spin_labels

        if print_results:
            tend = time.time()
            print("Took %0.2f s to generate state vector labels" % (tend - tstart))

        text_labels = ''
        #for jj in range(self.nspecies - 1, -1, -1):
        for jj in range(0, self.nspecies, 1):
            for ii in range(0, self.geometry.nsites):
                text_labels = text_labels + '%d%s ' % (ii, chr(97 + jj))
        # for ii in range(0, self.geometry.nsites):
        #     text_labels = text_labels + '%dup ' % ii

        return state_spin_labels, text_labels

    # ########################
    # build hamiltonian
    # ########################
    def get_hopping_mat(self, t):
        """
        Create a matrix representing the amplitude of hopping between any two sites. This is an nsites x nsites matrix,
        where M[ii, jj] is the prefactor of c^\dag_i c_j in the Hamiltonian. This matrix must be hermitian.
        :param geom_obj: ed_geometry object specifying geometry
        :param tunn: hopping energy along the x-direction
        :param ty: hopping energy along the y-direction
        :return:
        """
        # hopping_mat = np.multiply(tunn * geom_obj.is_x_neighbor + ty * geom_obj.is_y_neighbor, geom_obj.phase_mat)
        if t.size == 1:
            t = t * np.ones(self.nspecies)

        hopping_mat = np.zeros((self.geometry.nsites, self.geometry.nsites, self.nspecies))
        for ii in range(0, self.nspecies):
            hopping_mat[:, :, ii] = t[ii] * np.multiply(self.geometry.adjacency_mat, self.geometry.phase_mat)

            if not np.array_equal(hopping_mat[:, :, ii], hopping_mat[:, :, ii].conj().transpose()):
                raise Exception('Hopping matrix in get_hopping_mat was not hermitian.')

        return hopping_mat

    def get_samespecies_int_mat(self, int):
        """
        Get matrix representing nearest-neighbor interactions_4
        :param int: either a single number or an array of length self.nspecies
        :return:
        """

        if int.size == 1:
            int = int * np.ones((self.nspecies))

        int_mat = np.zeros((self.geometry.nsites, self.geometry.nsites, self.nspecies))
        for ii in range(0, self.nspecies):
            int_mat[:, :, ii] = int[ii] * self.geometry.adjacency_mat

        return int_mat

    def createH(self, projector=None, print_results=0):
        """Construct Fermi-Hubbard Hamiltonian
        :param projector: Sparse matrix, projection operator to be applied to each term of the Hamiltonian as it is
        constructed. Applied as P * H * P.conj().transpose()
        :param print_results:
        :return: Sparse matrix, H, the Hamiltonian
        """

        if print_results:
            tstart = time.clock()

        if projector is None:
            projector = sp.eye(self.nstates)
        nstates = projector.shape[0]

        H = sp.csr_matrix((nstates, nstates))

        H = H + self.get_u_interspecies_op(self.us_interspecies, projector=projector)
        H = H + self.get_u_samespecies_op(self.us_same_species, projector=projector)
        H = H + self.get_kinetic_op(projector=projector, direction_vect=np.array([1, 0]))
        H = H + self.get_kinetic_op(projector=projector, direction_vect=np.array([0, 1]))
        H = H + self.get_chemical_pot_op(self.mus, projector=projector)
        H = H + self.get_potential_op(self.potentials, projector=projector)

        if print_results:
            tend = time.clock()
            print("Constructing hamiltonian of size %dx%d took %0.2f s" % (H.shape[0], H.shape[1], tend - tstart))

        return H

    def get_kinetic_op(self, projector=None, direction_vect=np.array([1, 0])):
        """
        Compute the kinetic energy operator for a specific hopping connection matrix
        :param hopping_mat: NumPy array of size n_physical_sites x n_physical_sites, specifying hopping. Hopping terms
        are assumed to be negative.
        i.e., the Hamiltonian uses terms like -tsmat[ii,jj] * op
        :param projector:
        :return:
        """
        # TODO: add option for kinetic energy along some arbitary direction
        n_physical_sites = self.geometry.nsites
        n_logical_sites = n_physical_sites * self.nspecies
        nstates = 2 ** n_logical_sites

        if projector is None:
            projector = sp.eye(nstates)
        nstates = projector.shape[0]

        ke_op = sp.csr_matrix((nstates, nstates))

        sep_along_vector = np.sqrt(np.square(direction_vect[0] * self.geometry.xdist_mat) +
                                   np.square(direction_vect[1] * self.geometry.ydist_mat))
        # hopping terms
        if n_physical_sites > 1:

            # sum over physical sites
            for ii in range(0, self.ts.shape[0]):
                for jj in range(0, self.ts.shape[1]):
                    # avoid repeating elements
                    if ii > jj:
                        # sum over species
                        for kk in range(0, self.nspecies):
                            # avoid calculating operators unless will be nonzero
                            if self.ts[ii, jj, kk] != 0:
                                ke_op = ke_op + self.ts[ii, jj, kk] * sep_along_vector[ii, jj] * (
                                        - projector * self.get_two_site_op(ii, kk, jj, kk, self.cdag_op, self.c_op, "fermion") * projector.conj().transpose()
                                        - projector * self.get_two_site_op(jj, kk, ii, kk, self.cdag_op, self.c_op, "fermion") * projector.conj().transpose()
                                        )
        return ke_op

    def get_u_interspecies_op(self, us, projector=None):
        """
        Get operator representing onsite interaction between spin-up and spin-down fermions. Right now, only works if
        class has two spin species
        :param onsite-int: Int or NumPy array of length n_physical_sites, specifying interaction term between up and down spins
        :param projector:
        :return:
        """

        if projector is None:
            projector = sp.eye(self.nstates)
        nstates = projector.shape[0]

        if isinstance(us, (int, float)):
            us = float(us) * np.ones(self.geometry.nsites)

        u_op = sp.csr_matrix((nstates, nstates))

        for ii in range(0, self.geometry.nsites):
            # onsite interaction terms
            if us[ii] != 0:
                u_op = u_op + us[ii] * projector * self.get_two_site_op(ii, 0, ii, 1, self.n_op, self.n_op, "boson") * projector.conj().transpose()
            # TODO: if add offsite interaction terms need to avoid double counting!

        return u_op

    def get_u_samespecies_op(self, us, projector=None):
        """
        Get operator representing offsite interaction between fermions of the same species
        :param us:
        :param projector:
        :return:
        """
        if projector is None:
            projector = sp.eye(self.nstates)
        nstates = projector.shape[0]

        u_op = sp.csr_matrix((nstates, nstates))

        for ii in range(0, self.geometry.nsites):
            for jj in range(0, self.geometry.nsites):
                if ii >= jj: # avoid double counting offsite terms
                    for kk in range(0, self.nspecies):
                        if us[ii, jj, kk] != 0:
                            u_op = u_op + us[ii, jj, kk] * projector * self.get_two_site_op(ii, kk, jj, kk, self.n_op, self.n_op, "boson") * projector.conj().transpose()

        return u_op

    def get_potential_op(self, vs, projector=None):
        if projector is None:
            projector = sp.eye(self.nstates)
        nstates = projector.shape[0]

        v_op = sp.csr_matrix((nstates, nstates))

        for ii in range(0, self.geometry.nsites):
            for kk in range(0, self.nspecies):
                if vs[ii, kk] != 0:
                    v_op = v_op + vs[ii, kk] * projector * self.get_single_site_op(ii, kk, self.n_op, "boson") * projector.conj().transpose()

        return v_op

    def get_chemical_pot_op(self, mus, projector=None):
        """

        :param mus: Int or NumPy array of size n_physical_sites x nspins, specifying chemical potential
        :param projector:
        :return:
        """
        # chemical potential terms
        n_physical_sites = self.geometry.nsites
        n_logical_sites = n_physical_sites * self.nspecies
        nstates = 2 ** n_logical_sites

        if projector is None:
            projector = sp.eye(nstates)
        nstates = projector.shape[0]

        mu_op = sp.csr_matrix((nstates, nstates))

        for ii in range(0, n_physical_sites):
            for jj in range(0, self.nspecies):
                if mus[jj] != 0:
                    mu_op = mu_op - mus[jj] * projector * self.get_single_site_op(ii, jj, self.n_op, "boson") * projector.conj().transpose()
        return mu_op

    # ########################
    # calculate operators
    # ########################

    def get_swap_op(self, site1, site2, species):
        """
        Construct an operator that swaps the states of two sites. This version does not require any recursion.
        :param site1: Integer value specifying first site
        :param site2: Integer value specifying second site
        :param n_logical_sites: Integer specifying total number of logical sites, typically the number of real
        sites * number of spins
        :return: Sparse matrix
        """
        return self.get_two_site_op(site1, species, site2, species, self.cdag_op, self.c_op, "fermion") + \
               self.get_two_site_op(site2, species, site1, species, self.cdag_op, self.c_op, "fermion") + \
               sp.eye(self.nstates) - self.get_single_site_op(site1, species, self.n_op, format="boson") - \
               self.get_single_site_op(site2, species, self.n_op, format="boson")

    def get_current_op(self, direction_vector = np.array([[1], [0]])):
        """
        Get the paramagnetic current operator along a specific spatial direction.
        :param direction_vector: spatial direction to compute the current operator along. By default, the x direction.
        :return:
        """

        # NOTE: the current operator doesn't make sense for a 2 site chain with periodic boundary conditions, because
        # such a chain (ring) doesn't have a consistent notion of right vs. left (clockwise vs. counterclockwise)

        geom_obj = self.geometry
        tx = self.tx
        ty = self.ty

        direction_vector = np.reshape(np.asarray(direction_vector), [2])

        current_op = 0
        distance_component_x = geom_obj.xdist_mat * direction_vector[0]
        distance_component_y = geom_obj.ydist_mat * direction_vector[1]
        is_x_neighbor = np.round(np.abs(geom_obj.xdist_mat), self._round_decimals) == 1
        is_y_neighbor = np.round(np.abs(geom_obj.ydist_mat), self._round_decimals) == 1

        for ii in range(0, geom_obj.nsites):
            for jj in range(0, geom_obj.nsites):
                for kk in range(0, self.nspecies):
                    # logical_site1_index = self.spinful2spinlessIndex(ii, geom_obj.nsites, kk)
                    # logical_site2_index = self.spinful2spinlessIndex(jj, geom_obj.nsites, kk)

                    if geom_obj.adjacency_mat[ii,jj] and is_x_neighbor[ii, jj]:
                        link_current = tx * distance_component_x[ii, jj] * \
                                       self.get_two_site_op(ii, kk, jj, kk, self.cdag_op, self.c_op, format="fermion")
                                       #self.getTwoSiteOp(logical_site1_index, logical_site2_index, geom_obj.nsites * self.nspecies, self.cdag_op, self.c_op, format = "fermion")
                        current_op = current_op + link_current

                    if geom_obj.adjacency_mat[ii,jj] and is_y_neighbor[ii, jj]:
                        link_current = ty * distance_component_y[ii, jj] * \
                                       self.get_two_site_op(ii, kk, jj, kk, self.cdag_op, self.c_op, format="fermion")
                                       #self.getTwoSiteOp(logical_site1_index, logical_site2_index, geom_obj.nsites * self.nspecies, self.cdag_op, self.c_op, format = "fermion")
                        current_op = current_op + link_current
        current_op = 1j * current_op

        return current_op

    def integrate_conductivity(self, current_matrix_elems, eig_vals, temperature, print_results = 0):
        """
        Computes the two-sided integral of the optical conductivity in frequency space. This does not require any
        broadening, unlike computing the optical conductivity directly.
        :param current_matrix_elems: the current operator in the eigenvector basis
        :param eig_vals: eigenvalues corresponding to the eigenvectors
        :param temperature:
        :param print_results:
        :return:
        """
        # no broadening is necessary to perform this integral, so better to split this off as a separate function

        # \int_{-\inf}^\inf d\omega_start Re(sigma(\omega_start)) = \pi/N/Z *\sum_{nm} e^{-\beta * E_n} |J_nm|^2 * (1 - exp(-beta * (E_m - E_n))/(E_m - E_n)
        # but we need to exclude any E_m = E_n, so a more proper way to write this might be twice the one-sided integral of the optical
        # conductivity, which excludes the zero frequency weights...
        # TODO: could also handle period_start = infinity case...
        if print_results:
            tstart = time.clock()

        temperature = float(temperature)
        nsites = self.geometry.nsites

        if temperature != 0:

            beta = 1 / temperature
            eig_vals = eig_vals - eig_vals[0]
            z = np.sum(np.exp( - beta * eig_vals))
            [a, b] = np.meshgrid(eig_vals, eig_vals)
            omegas = b - a
            #omegas[omegas == 0] = 1e9
            weight = np.divide(1 - np.exp(- beta * omegas), omegas)
            weight[omegas == 0] = 0
            # multiply each column by the appropriate exponentail factor and weight. Then sum over everything.
            # think of n as columns, n as rows
            sigma_re_two_sided_int = np.pi / z / nsites * np.sum(np.sum(np.multiply(np.square(np.abs(current_matrix_elems)).dot(np.diag(np.exp(- beta * eig_vals))), weight)))

        else:
            #period_start = 0, only need to sum over one set of states
            omegas = eig_vals - eig_vals[0]
            #omegas[omegas == 0] = 1e9
            weight = np.divide(1, omegas)
            weight[omegas == 0 ] = 0
            current_matrix_elems = current_matrix_elems[:, 0]
            # we are effectively doing the one sided integral, therefore we need to multiply this by 2
            sigma_one_sided_int = np.pi / nsites * np.sum(np.multiply(np.square(np.abs(current_matrix_elems)), weight))
            sigma_re_two_sided_int = 2 * sigma_one_sided_int

        if print_results:
            tend = time.clock()
            print("integrate_current for %d x %d matrix took %0.2f s" % (omegas.shape[0], omegas.shape[0], tend - tstart))

        return sigma_re_two_sided_int

    def get_optical_cond_fn(self, current_matrix_elems, eig_vals, temperature, print_results = 0):
        """
        Create a function for computing the optical conductivity at arbitrary frequency omega_start and arbitrary broadening
        paramter eta
        :param current_matrix_elems: the current operator in the eigenvector basis
        :param eig_vals: eigenvalues corresponding to the eigenvectors
        :param temperature:
        :param print_results:
        :return:
        """
        if print_results:
            tstart = time.clock()

        #normalized lorentzian function. Area is constant with changing eta
        #expected to need a factor of 1/pi here to normalize lorentzian area to 1
        lor = lambda wo, eta, w, amp: (amp * eta / np.pi) / (eta ** 2 + np.square(w - wo))
        temperature = float(temperature)
        eig_vals = eig_vals - eig_vals[0]

        if temperature != 0:
            beta = 1 / temperature
            z = np.sum(np.exp(- beta * eig_vals))
            [a, b] = np.meshgrid(eig_vals, eig_vals)
            omegas = b - a

            thermal_mat_elem = np.square(np.abs(current_matrix_elems)).dot(np.diag(np.exp(- beta * eig_vals)))
            omegas = omegas[np.abs(current_matrix_elems) > 10 ** -16].flatten()
            thermal_mat_elem = thermal_mat_elem[np.abs(current_matrix_elems) > 10 ** -16].flatten()
            #there is some ambiguity in how to define this function. In particular, should we put e_n-e_m or omega_start as
            #the argument for the exponential and the denominator? In the limit eta -> 0, it shouldn't matter. But for
            #finite eta, it does. This controls whether or not the matrix elements at e_m-e_n = 0 contribute to the sum
            #or not. If we put e_m-e_n in the exponential and omega_start in the denominator, then we get rid of these
            #elements, because this is zero (and we shouldn't evaluate omega_start = 0). If we put both as e_m-e_n, then
            #we get a nan, which is also removed later. But we should not put them both as omega_start, because???
            omega_fn = lambda wo, eta, w, amp: np.divide(
                np.multiply(1 - np.exp(-beta * w), lor(wo, eta, w, amp)), wo)

        else:
            z = 1
            omegas = eig_vals
            thermal_mat_elem = np.square(np.abs(current_matrix_elems[:, 0]))
            omegas = omegas[np.abs(current_matrix_elems[:, 0]) > 10 ** -16].flatten()
            thermal_mat_elem = thermal_mat_elem[np.abs(current_matrix_elems[:, 0]) > 10 ** -16].flatten()
            # need to turn infs into nans to more easily get rid of
            omega_fn = lambda wo, eta, w, amp: np.multiply(np.divide(lor(np.abs(wo), eta, w, amp), w), 1 - np.isinf(1/w))

        opt_cond_fn = lambda wo, eta: np.pi / z / self.geometry.nsites * np.nansum(omega_fn(wo, eta, omegas, thermal_mat_elem))

        if print_results:
            tend = time.clock()
            print("get_optical_cond_fn took %0.2f s" % (tend - tstart))

        return opt_cond_fn


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # create and diagonalize systems
    nx = 8
    ny = 1
    print_all = 1

    # define geometry
    n_ups = np.ceil(nx * ny / 2)
    n_dns = nx * ny - n_ups
    phi1 = 0
    phi2 = 0
    if nx > 1:
        bc1_open = 0
    else:
        bc1_open = 1

    if ny > 1:
        bc2_open = 0
    else:
        bc2_open = 1

    gm = geom.Geometry.createSquareGeometry(nx, ny, phi1, phi2, bc1_open, bc2_open)
    #gm = geom.Geometry.createTiltedSquareGeometry(3, 1, phi1, phi2, bc1_open, bc2_open)

    # instantiate class with fermions parameters
    U = 1
    t = 1
    mu = 0
    h = fermions(gm, U, t, ns=np.array([n_ups, n_dns]))

    # build symmetry projection operators
    # x-translation
    xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
    xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, gm)
    xtransl_op = h.n_projector * h.get_xform_op(xtransl_cycles) * h.n_projector.conj().transpose()

    # y-translations
    if not bc2_open:
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, gm)
        ytransl_op = h.n_projector * h.get_xform_op(ytransl_cycles) * h.n_projector.conj().transpose()

    # rot-fn
    # rot_fn = symm.getRotFn(4)
    # rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, gm)
    # rot_op = h.n_projector * h.get_xform_op(rot_cycles) * h.n_projector.conj().transpose()

    # reflection about y-axis
    # refl_fn = symm.getReflFn(np.array([0, 1]))
    # refl_cycles, max_cycle_len_refl = symm.findSiteCycles(refl_fn, gm)
    # refl_op = h.n_projector * h.get_xform_op(refl_cycles) * h.n_projector.conj().transpose()

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
        projs = None

    # or c4v symmetry
    #projs = symm.getC4VProjectors(rot_op, refl_op, print_all)

    # diagonalize all symmetry sectors
    eigvals_sectors = []
    for proj in projs:
        H = h.createH(projector=proj * h.n_projector, print_results=print_all)
        eigs_temp, _ = h.diagH(H, print_results=print_all)

        eigvals_sectors.append(eigs_temp)

    eigs_all_sectors = np.sort(np.concatenate(eigvals_sectors))
    print("smallest eigenvalue was %0.2f" % eigs_all_sectors[0])

    if h.n_projector.shape[1] < 5000:
        # build and diagonalize Hamiltonian
        hamiltonian = h.createH(projector=h.n_projector, print_results=print_all)
        eigvals_full, _ = h.diagH(hamiltonian, print_results=print_all)

        # compare eigenvalues using symmetries and not
        print("Eigenvalues calculated with and without symmetries differed by at most %0.3e" % (eigvals_full - eigs_all_sectors).max())

        # find maximum difference by sector...
        for eig_sector, kx, ky in zip(eigvals_sectors, kxs, kys):
            print("sector kx = %0.2f, ky = %0.2f had size %d" % (kx, ky, eig_sector.size))
            if eig_sector.size:
                closest_vals = [np.min(np.abs(e - eigvals_full)) for e in eig_sector]
                max_diff = np.asarray(closest_vals).max()
                print("maximum eig value difference was %0.3e" % max_diff)

    else:
        print("skipping full Hamiltonian diagonalization because size of space was large = %d states. Total size of symmetry spaces = %d states" % (h.n_projector.shape[1], len(eigs_all_sectors)))

    gm.dispGeometry()
    plt.show()
