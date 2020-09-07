import numpy as np
import os

# TODO: need to modify get reciprocal vectors for case when matrix inverse fails

class Geometry():
    _round_decimals = 14

    def __init__(self, xlocs=None, ylocs=None, adjacency_mat=None, phase_mat=None, lattice=None):
        """
        Initialize for Geometry class. This should never be called directly. Instead, you should use one of the
        classmethods createPeriodicGeometry or createNonPeriodicGeometry. Those should be thought of as alternate
        constructors for the class. The reason for this somewhat complicated calling method is we want to be able to
        represent two types of geometries: (1) geometries with a high amount of symmetry, which can be specified with
        only two vectors and boundary conditions and (2) arbitrary clusters which have little symmetry and are most
        conveniently defined by giving the x and y coordinates of the sites involved.

        :param xlocs: x location of sites in real space. 1D numpy array.
        :param ylocs: y location of sites in real space. 1D numpy array.
        :param adjacency_mat: nsites x nsites numpy 2D array. M[i,j] = 1 if two sites are "connected". M[i,j] = M[j,i]
        :param phase_mat: nsites x nsites numpy 2D complex array. M[i, j] is the phase acquired moving from site i to j.
        M[i,j] = M[i, j]*
        """

        # class storing periodicity vectors, and other lattice information if our geometry is part of a lattice
        self.lattice = lattice
        self.xlocs = ensure_1d_vect(xlocs)
        self.ylocs = ensure_1d_vect(ylocs)
        self.nsites = len(self.xlocs)

        if self.lattice is not None:
            # if periodic system
            self.xdist_mat, self.ydist_mat, latt_vect1_sep, latt_vect2_sep = \
                self.lattice.get_reduced_distance(self.xlocs, self.ylocs)

            self.adjacency_mat = (np.round(np.sqrt(np.square(self.xdist_mat) +
                                                   np.square(self.ydist_mat)), self._round_decimals) == 1.).astype(float)

            self.phase_mat = self.lattice.get_phase_mat(self.xdist_mat, self.ydist_mat)

            # construct adjacency matrix based on which sites are a single lattice vector away
            # but this method doesn't seem to work on a triangular lattice, as some sites which should be connected
            # differ by p1 + p2 !
            # self.adjacency_mat = np.logical_and(np.abs(latt_vect1_sep) == 1., np.abs(latt_vect2_sep) == 0.).astype(float) + \
            #                      np.logical_and(np.abs(latt_vect1_sep) == 0., np.abs(latt_vect2_sep) == 1.).astype(float)
            # TODO: still not convinced if this is best way to construct the adjacency matrix from the lattice data
        else:
            # if not a periodic system
            self.xdist_mat, self.ydist_mat = self.get_site_distance()

            if phase_mat is None:
                self.phase_mat = np.ones((self.nsites, self.nsites))
            else:
                self.phase_mat = phase_mat

            if adjacency_mat is None:
                # if do not supply is_y_neighbor or is_x_neighbor, then we will assume that sites distance one apart
                # are neighbors
                _, _, is_neighbor = \
                    self.get_neighbors_by_distance(self.xdist_mat, self.ydist_mat)
                self.adjacency_mat = is_neighbor
            else:
                self.adjacency_mat = adjacency_mat

        # validate instance
        if not self.validate_instance():
            raise Exception('Geometry instance failed validation.')
        # TODO: it seems like there is a problem with one of the phase mats! Problem seems to be that if we have two
        # sites and they are equally far away in 'positive' and 'negative' distance, i.e. equally far away directly
        # and through the periodic entries.c., then defining their distance is ambiguous

    # #################################
    # Alternate constructors -- typically these should be used instead of __init__
    # #################################

    @classmethod
    def create_lattice_geom(cls, latt_vect1, latt_vect2, basis_vects, periodicity_vect1, periodicity_vect2,
                            phase1=0, phase2=0, bc1_open=True, bc2_open=True):
        """
        Create a Geometry instance by specifying an underlying lattice

        :param latt_vect1:
        :param latt_vect2:
        :param basis_vects:
        :param periodicity_vect1:
        :param periodicity_vect2:
        :param phase1:
        :param phase2:
        :param bc1_open:
        :param bc2_open:
        :return:
        """

        # trick to generate sites using lattice even if using open boundary conditions in the end
        lattice_helper = Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1, periodicity_vect2, phase1, phase2)

        nsites, xlocs, ylocs = lattice_helper.get_unique_sites()

        # if don't want lattice to be periodic, set periodicity vectors to zero
        if bc1_open:
            periodicity_vect1 = np.zeros((2, 1))
        if bc2_open:
            periodicity_vect2 = np.zeros((2, 1))

        lattice = Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1, periodicity_vect2, phase1, phase2)

        return cls(xlocs, ylocs, lattice=lattice)

    @classmethod
    def createNonPeriodicGeometry(cls, xlocs, ylocs, adjacency_mat=None, phase_mat=None):
        """
        Create a geometry without specifying an underlying lattice

        :param xlocs: list of x coordinates for each site
        :param ylocs:
        :param adjacency_mat: adjacency matrix between sites
        :param phase_mat: phase matrix between sites
        :return: Geometry instance
        """
        return cls(xlocs, ylocs, adjacency_mat, phase_mat)

    @classmethod
    def createRegularPolygonGeometry(cls, nsides, adjacency_mat=None, phase_mat=None):

        if adjacency_mat is None:
            adjacency_mat = np.zeros((nsides, nsides))
            adjacency_mat[0, 1] = 1
            adjacency_mat[0, -1] = 1
            adjacency_mat[-1, 0] = 1
            adjacency_mat[-1, -2] = 1
            for ii in range(1, nsides-1):
                adjacency_mat[ii, ii - 1] = 1
                adjacency_mat[ii, ii + 1] = 1

        theta_in = 2*np.pi / nsides
        d = 0.5 / np.sin(theta_in / 2)
        xlocs = [d * np.sin(theta_in * ii) for ii in range(nsides)]
        ylocs = [d * np.cos(theta_in * ii) for ii in range(nsides)]
        return cls(xlocs, ylocs, adjacency_mat, phase_mat)

    @classmethod
    def createSquareGeometry(cls, nx_sites, ny_sites, phase1=0, phase2=0, bc1_open=True, bc2_open=True):
        """
        Create a Geometry instance for a square cluster on a square lattice

        :param nx_sites:
        :param ny_sites:
        :param phase1:
        :param phase2:
        :param bc1_open:
        :param bc2_open:
        :return:
        """

        if nx_sites == 1 and not bc1_open:
            raise Exception('Invalid createSquareGeometry options specified. nx_sites = 1, '
                            'but periodic boundary conditions selected.')

        if ny_sites == 1 and not bc2_open:
            raise Exception('Invalid createSquareGeometry options specified. ny_sites = 1, '
                            'but periodic boundary conditions selected.')

        latt_vect1 = np.array([[1], [0]])
        latt_vect2 = np.array([[0], [1]])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = np.array([[nx_sites], [0]])
        periodicity_vect2 = np.array([[0], [ny_sites]])

        return cls.create_lattice_geom(latt_vect1, latt_vect2, basis_vects,
                                       periodicity_vect1, periodicity_vect2, phase1,
                                       phase2, bc1_open, bc2_open)

    @classmethod
    def createTiltedSquareGeometry(cls, nsites_right, nsites_up, phase1=0, phase2=0, bc1_open=True, bc2_open=True):
        """
        Create a geometry instance for a tilted square cluster on a square lattice

        :param nsites_right:
        :param nsites_up:
        :param phase1:
        :param phase2:
        :param bc1_open:
        :param bc2_open:
        :return:
        """

        latt_vect1 = np.array([[1], [0]])
        latt_vect2 = np.array([[0], [1]])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = np.array([[nsites_right], [nsites_up]])
        periodicity_vect2 = np.array([[-nsites_up], [nsites_right]])

        return cls.create_lattice_geom(latt_vect1, latt_vect2, basis_vects, periodicity_vect1,
                                       periodicity_vect2, phase1, phase2, bc1_open, bc2_open)

    @classmethod
    def createTriangleGeometry(cls, n1_sites, n2_sites, phase1=0, phase2=0, bc1_open=True, bc2_open=True):
        """
        Create a geometry instance for a triangular lattice on a 'square' cluster

        :param n1_sites:
        :param n2_sites:
        :param phase1:
        :param phase2:
        :param bc1_open:
        :param bc2_open:
        :return:
        """

        if n1_sites == 1 and not bc1_open:
            raise Exception('Invalid createSquareGeometry options specified. nx_sites = 1, '
                            'but periodic boundary conditions selected.')

        if n2_sites == 1 and not bc2_open:
            raise Exception('Invalid createSquareGeometry options specified. ny_sites = 1, '
                            'but periodic boundary conditions selected.')

        latt_vect1 = np.array([[1], [0]])
        latt_vect2 = np.array([[0.5], [np.sqrt(3)/2]])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = n1_sites * latt_vect1
        periodicity_vect2 = n2_sites * latt_vect2

        return cls.create_lattice_geom(latt_vect1, latt_vect2, basis_vects, periodicity_vect1,
                                       periodicity_vect2, phase1, phase2, bc1_open, bc2_open)

    @classmethod
    def createHexagonalGeometry(cls, n1_sites, n2_sites, phase1=0, phase2=0, bc1_open=True, bc2_open=True):
        """
        Create a geometry instance for a hexagonal lattice

        :param n1_sites:
        :param n2_sites:
        :param phase1:
        :param phase2:
        :param bc1_open:
        :param bc2_open:
        :return:
        """

        if n1_sites == 1 and not bc1_open:
            raise Exception('Invalid createHexagonalGeometry options specified. n1_sites = 1, '
                            'but periodic boundary conditions selected.')

        if n2_sites == 1 and not bc2_open:
            raise Exception('Invalid createHexagonalGeometry options specified. n2_sites = 1, '
                            'but periodic boundary conditions selected.')

        latt_vect1 = np.array([[1.5], [np.sqrt(3)/2]])
        latt_vect2 = np.array([[1.5], [-np.sqrt(3)/2]])
        # i.e. origin in the center of the unit cell
        basis_vects = [np.array([[-1.], [0.]]), np.array([[-0.5], [np.sqrt(3)/2]])]
        periodicity_vect1 = n1_sites * np.array([[3.], [0.]])
        periodicity_vect2 = n2_sites * np.array([[0.], [np.sqrt(3)]])

        return cls.create_lattice_geom(latt_vect1, latt_vect2, basis_vects, periodicity_vect1, periodicity_vect2,
                                       phase1, phase2, bc1_open, bc2_open)

    # #################################
    # General geometry functions
    # #################################

    def get_center_of_mass(self):
        """
        Find the center of mass of the given geometry
        :return: cx, cy
        """
        cx = np.mean(self.xlocs)
        cy = np.mean(self.ylocs)
        return cx, cy

    def get_sorting_permutation(self, sorting_mode='top_alternating'):
        """
        Sort lattice sites using a type of lexographical order

        :param sorting_mode: the type of sorting mode to be used. The available options are
        'top_alternating': order sites top-to-bottom, and then alternating left-to-right then right-to-left for each row.
        'top_left': order sites top-to-bottom then left-to-right
        'bottom_right': order sites bottom-to-top, then left-to-right
        'adjacency': sort sites by the number of adjacent sites

        :return: sorted_indices
        """

        if sorting_mode == 'top_alternating':
            # put sites in order top-to-bottom, alternating order in each row
            # For lattice where periodicity respects A and B sublattices, this should
            # order the sites so that every other number is on a different sublattice.
            # This is helpful because it makes identifying some states easier
            sorted_indices = np.lexsort((self.xlocs * np.power(-1., self.ylocs - np.max(self.ylocs)), -self.ylocs))
        elif sorting_mode == 'top_left':
            # put sites in order top-to-bottom, then left-to-right
            # sort on largest-y (smallest -y), then sort on smallest-x
            sorted_indices = np.lexsort((self.xlocs, -self.ylocs))
        elif sorting_mode == 'bottom_right':
            # put sites in order bottom-to-top, then left-to-right
            sorted_indices = np.lexsort((self.xlocs, self.ylocs))
        elif sorting_mode == 'adjacency':
            # sort sites by number of adjacent sites
            # if sites have the same number of adjacent sites, sort by how many sites can be reach in k steps for
            # k = 1, ..., nsites
            adj_mat_powers = [self.adjacency_mat]
            adjacency_order_n = [np.sum(self.adjacency_mat, 1)]
            adjacency_test_n = np.zeros((self.nsites, self.nsites))
            adjacency_test_n[:, 0] = np.sum(self.adjacency_mat, 1)
            for ii in range(1, self.nsites):
                adj_mat_powers.append(adj_mat_powers[ii - 1].dot(self.adjacency_mat))
                adjacency_order_n.append(np.sum(adj_mat_powers[ii], 1))
                adjacency_test_n[:, ii] = np.sum(adj_mat_powers[ii], 1)

            sorted_indices = np.lexsort(tuple(adjacency_order_n))

            # however, if two sites are equivalent under a symmetry operation, this process does not distinguish them.
            # It doesn't matter which site we select first, but we need to choose the next site in some way based on
            # this first site.
        else:
            raise Exception("sorting_mode must be 'top_alternating', 'top_left', 'bottom_right' or"
                            " 'adjacency' but was %s" % sorting_mode)

        return sorted_indices

    def get_site_distance(self):
        """
        Return the x and y distances between sites
        :return: xdist, ydist
        """
        xdist = np.zeros([self.nsites, self.nsites])
        ydist = np.zeros([self.nsites, self.nsites])
        for ii in range(0, self.nsites):
            for jj in range(0, self.nsites):
                xdist[ii, jj] = self.xlocs[ii] - self.xlocs[jj]
                ydist[ii, jj] = self.ylocs[ii] - self.ylocs[jj]
        return xdist, ydist

    @staticmethod
    def get_neighbors_by_distance(xdist_mat, ydist_mat):
        """
        Determine which sites are neighbors. Two sites are neighbors if either their x or y coordinates differ by one
        (but not both).
        :param xdist_mat: An nsites x nsites matrix where M[ii, jj] is the distance between sites ii and jj in the
         x-direction
        :param ydist_mat:
        :return: is_x_neighbor, is_y_neighbor, is_neighbor
        """
        # TODO: this function is assuming points lie on an underlying lattice. Add lattice vectors as an argument
        # and deal with this similar to how would for lattice class???
        xdist_reduced = xdist_mat
        ydist_reduced = ydist_mat
        is_x_neighbor = (np.round(np.abs(xdist_reduced) - 1) == 0) * (np.round(np.abs(ydist_reduced)) == 0)
        is_y_neighbor = (np.round(np.abs(ydist_reduced) - 1) == 0) * (np.round(np.abs(xdist_reduced)) == 0)
        is_neighbor = np.logical_or(is_x_neighbor, is_y_neighbor)

        is_x_neighbor = is_x_neighbor.astype(int)
        is_y_neighbor = is_y_neighbor.astype(int)
        is_neighbor = is_neighbor.astype(int)

        return is_x_neighbor, is_y_neighbor, is_neighbor

    # #################################
    # transformation functions
    # #################################
    def permute_sites(self, permutation):
        """
        Rearrange sites according to a given permutation. Transform all other quantities of the instance, including
        the adjacency matrix, etc. to match the new ordering.
        :param permutation: permutation[jj] = ii means that site ii before the transformation becomes site jj after.
        permutation should be a list.
        :return:
        """
        if not np.array_equal(np.array(sorted(permutation)), np.array(range(0, self.nsites))):
            raise Exception('permutation must be a list of length nsites containing all numbers between 0 and n-1')
        self.xlocs = self.xlocs[permutation]
        self.ylocs = self.ylocs[permutation]

        basis_change_mat = np.zeros([self.nsites, self.nsites])
        for ii, jj in enumerate(permutation):
            basis_change_mat[ii, jj] = 1

        self.adjacency_mat = basis_change_mat.dot(self.adjacency_mat.dot(basis_change_mat.transpose()))
        self.phase_mat = basis_change_mat.dot(self.phase_mat.dot(basis_change_mat.transpose()))
        self.xdist_mat = basis_change_mat.dot(self.xdist_mat.dot(basis_change_mat.transpose()))
        self.ydist_mat = basis_change_mat.dot(self.ydist_mat.dot(basis_change_mat.transpose()))
        #self.dist_reduced_multiplicity = basis_change_mat.dot(self.dist_reduced_multiplicity.dot(basis_change_mat.transpose()))
        # self.is_x_neighbor = basis_change_mat.dot(self.is_x_neighbor.dot(basis_change_mat.transpose()))
        # self.is_y_neighbor = basis_change_mat.dot(self.is_y_neighbor.dot(basis_change_mat.transpose()))
        # self.is_neighbor = basis_change_mat.dot(self.is_neighbor.dot(basis_change_mat.transpose()))

    # #################################
    # display functions
    # #################################

    def dispGeometry(self):
        """
        Display the connection matrix which defines our geometry.
        :return:
        """

        r = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
        if r == 0:
            import matplotlib.pyplot as plt

            nsites = len(self.xlocs)
            fig_handle = plt.figure()

            plt.subplot(1, 3, 1)
            plt.scatter(self.xlocs, self.ylocs)
            for ii in range(0, nsites):
                plt.text(self.xlocs[ii], self.ylocs[ii], ii)
                for jj in range(0, nsites):
                    if np.round(np.abs(self.adjacency_mat[ii, jj]), self._round_decimals) > 0:
                        plt.plot([self.xlocs[ii], self.xlocs[jj]], [self.ylocs[ii], self.ylocs[jj]])
            plt.axis('equal')
            plt.title('site geometry')

            plt.subplot(1, 3, 2)
            plt.imshow(np.abs(self.adjacency_mat), interpolation='none')
            plt.title('site connection matrix amplitude')

            plt.subplot(1, 3, 3)
            angles = np.angle(np.round(self.phase_mat, self._round_decimals))
            # angles[angles > np.pi] = angles[angles > np.pi] - 2 * np.pi
            plt.imshow(angles, vmin=-np.pi, vmax=np.pi, interpolation='none')
            plt.title('phase matrix phase angle')

        else:
            print("loading matplotlib.pyplot threw an error, so skipping dispGeometry")
            fig_handle = 0

        return fig_handle

    # #################################
    # Validation functions
    # #################################

    def validate_instance(self):
        """
        validate entire instance
        :return:
        """
        if not self.validate_adjacency_mat():
            return 0
        if not self.validate_phase_mat():
            return 0
        # TODO: add more validation checks
        return 1

    def validate_adjacency_mat(self):
        """
        Check that adjacency_mat is sensible
        :return: bool
        """
        if not np.array_equal(self.adjacency_mat, np.transpose(self.adjacency_mat)):
            # ensure symmetric under transpose
            return 0
        if not np.sum(np.logical_or(self.adjacency_mat == 0, self.adjacency_mat == 1)) == self.adjacency_mat.size:
            # ensure matrix of zeros and ones
            return 0
        if not self.adjacency_mat.shape == (self.nsites, self.nsites):
            # check matrix is the correct size
            return 0
        return 1

    def validate_phase_mat(self):
        """
        check that phase_mat is sensible
        :return:
        """
        if not np.sum(np.round(np.abs(self.phase_mat - self.phase_mat.conj().transpose()), self._round_decimals) == 0) == self.phase_mat.size:
            # ensure symmetric under conjugate transpose
            return 0
        if not np.sum(np.round(np.abs(self.phase_mat), self._round_decimals) == 1) == self.phase_mat.size:
            # ensure matrix of phases
            return 0
        if not self.phase_mat.shape == (self.nsites, self.nsites):
            # check matrix is the correct size
            return 0
        return 1

    # #################################
    # Comparison functions
    # #################################

    def isequal_adjacency(self, other):
        """
        Compare two geometry instances based only on distance between sites and adjacency. In particular, ignore
        absolute coordinate positions
        :param other:
        :return:
        """
        # np.array_equal(self.dist_reduced_multiplicity, other.dist_reduced_multiplicity) and \
        if np.array_equal(self.xdist_mat, other.xdist_mat) and \
            np.array_equal(self.ydist_mat, other.ydist_mat) and \
            np.array_equal(self.adjacency_mat, other.adjacency_mat) and \
            self.lattice == other.lattice:
            return True
        else:
            return False

    def __eq__(self, other):
        """
        Test if two geometry instances are equal, in the sense that all properties are identical. We require, e.g.,
        that the adjacency matrices are the same. Two clusters that are the same under some permutation will *not*
        evaluate as equal.
        :param other:
        :return:
        """
        # TODO: what is the easiest way to compare equality of two clusters?
        if np.array_equal(self.xlocs, other.xlocs) and \
            np.array_equal(self.ylocs, other.ylocs) and \
            np.array_equal(self.xdist_mat, other.xdist_mat) and \
            np.array_equal(self.ydist_mat, other.ydist_mat) and \
            np.array_equal(self.adjacency_mat, other.adjacency_mat) and \
            self.lattice == other.lattice:
            return True
        else:
            return False

    def __ne__(self, other):
        """
        Test if two geometry instances are not equal. Note that if this method is not explicitly defined, it does not
        return the opposite of __eq__. Therefore it is necessary to define it.
        :param other:
        :return:
        """
        return not self.__eq__(other)


class Lattice():
    _round_decimals = 14

    def __init__(self, lattice_vect1, lattice_vect2, basis_vects=[[0, 0]], periodicity_vect1=(0, 0),
                 periodicity_vect2=(0, 0), phase1=0., phase2=0.):
        """

        :param lattice_vect1:
        :param lattice_vect2:
        :param basis_vects: #TODO: need to change way this is handled as default argument
        :param periodicity_vect1: Cell periodicity vector 1. Given a lattice site, if you add periodicity vector 1 to
        its coordinates, you will find yourself at an equivalent lattice site. i.e. at a lattice site that is identified
        with the first one.
        :param periodicity_vect2: Cell periodicity vector 2. Periodicity vectors 1 and 2 should not be linearly dependent
        :param phase1: Phase which is picked up along reciprocal vector 1. Useful for imposing twisted boundary conditions
        :param phase2: Phase which is picked up along reciprocal vector 2
        """
        # TODO: what is the best way to represent periodicity vectors I want to ignore? Should they be zero or None?

        self.lattice_vect1 = ensure_column_vect(lattice_vect1).astype(float)
        self.lattice_vect2 = ensure_column_vect(lattice_vect2).astype(float)
        self.reciprocal_latt_vect1, self.reciprocal_latt_vect2 = get_reciprocal_vects(self.lattice_vect1, self.lattice_vect2)

        self.basis_vects = [ensure_column_vect(v) for v in basis_vects]

        self.periodicity_vect1 = ensure_column_vect(periodicity_vect1).astype(float)
        self.periodicity_vect2 = ensure_column_vect(periodicity_vect2).astype(float)
        self.reciprocal_periodicity_vect1, self.reciprocal_periodicity_vect2 = \
            get_reciprocal_vects(self.periodicity_vect1, self.periodicity_vect2)

        self.phase1 = phase1
        self.phase2 = phase2

        if not self.validate_instance():
            raise Exception('validation failed for lattice instance.')

    def get_unique_sites(self):
        """
        Returns sites within the unit periodicity cell of the lattice
        :return: nsites, xlocs, ylocs

        nsites:

        xlocs:

        ylocs:
        """

        # the origin, periodicity vector 1, periodicity vector 2, and their sum form a parallelogram
        # we want to enumerate all points in this parallelogram
        # We can rewrite the edges of this parallelogram in terms of an integer number of lattice vectors
        # P1 = n * l1 + m * l2
        # P2 = i * l1 + j * l2
        # P1 + P2 = (n + i) * l1 + (m +j) * l
        _, _, n, m = reduce_vectors(self.lattice_vect1, self.lattice_vect2, self.periodicity_vect1[0, 0],
                                    self.periodicity_vect1[1, 0], mode='positive')
        _, _, i, j = reduce_vectors(self.lattice_vect1, self.lattice_vect2, self.periodicity_vect2[0, 0],
                                    self.periodicity_vect2[1, 0], mode='positive')

        # Every lattice point in our parallelogram can be written in the form
        # v = a * l1 + entries * l2,
        # if we suppose that n, m, i, j > 0 we would have 0 <= a <= n + i and 0 <= entries <= m + j
        # If we don't restrict these to be positive, then we only know a has to be between
        # the smallest and largest combination of n and i (and similarly for entries)
        latt_vect1_mult = np.arange(np.min([0, n, i, n+i]), np.max([0, n, i, n+i]))
        if latt_vect1_mult.size == 0:
            # in the case where periodicity_vect1 = [[0], [0]] we want this to be non-empty
            latt_vect1_mult = np.array([0.])

        latt_vect2_mult = np.arange(np.min([0, m, j, m+j]), np.max([0, m, j, m+j]))
        if latt_vect2_mult.size == 0:
            latt_vect2_mult = np.array([0.])

        # expanded list of all possible sums of lattice vectors
        xx, yy = np.meshgrid(latt_vect1_mult, latt_vect2_mult)
        xrav = xx.ravel()
        yrav = yy.ravel()

        # size of basis
        nbasis = len(self.basis_vects)

        vects = np.zeros((2, xx.size * nbasis))
        for ii in range(0, xx.size):
            for jj in range(0, nbasis):
                vects[:, ii * nbasis + jj][:, None] = xrav[ii] * self.lattice_vect1 + yrav[ii] * self.lattice_vect2 + self.basis_vects[jj]

        # reduce to sites with periodicity unit self
        xlocs_red, ylocs_red, _, _ = reduce_vectors(self.periodicity_vect1, self.periodicity_vect2,
                                                vects[0, :], vects[1, :], mode='positive')
        xlocs_red = np.round(xlocs_red, self._round_decimals)
        ylocs_red = np.round(ylocs_red, self._round_decimals)
        # eliminate duplicates
        locs = np.unique(np.concatenate([xlocs_red[None, :], ylocs_red[None, :]], 0), axis=1)
        xlocs = locs[0, :]
        ylocs = locs[1, :]
        nsites = len(xlocs)

        return nsites, xlocs, ylocs

    def get_reduced_distance(self, xlocs, ylocs):
        """
        Returns the distance between two sites taking into account the periodicity of our lattice.

        :param xlocs: a list of the x-coordinates of the lattice sites
        :param ylocs: a list of the y-coordinates of the lattice sites
        :return: xdist_min, ydist_min, latt_vect1_dist, latt_vect2_dist

        xdist_min: is an nsites x nsites matrix where M[ii, jj] is the x-distance between sites i and j

        ydist_min:

        latt_vect1_dist: is an nsites x nsites matrix where M[ii, jj] is the number latt_vect1's separating sites i and j

        latt_vect2_dist:
        """
        nsites = len(xlocs)

        xdist_min = np.zeros([nsites, nsites])
        ydist_min = np.zeros([nsites, nsites])
        latt_vect1_dist = np.zeros([nsites, nsites])
        latt_vect2_dist = np.zeros([nsites, nsites])

        for ii in range(0, nsites):
            for jj in range(0, ii):
                xdist_min[ii, jj], ydist_min[ii, jj], _, _ = \
                    reduce_vectors(self.periodicity_vect1, self.periodicity_vect2,
                                   xlocs[ii] - xlocs[jj], ylocs[ii] - ylocs[jj], mode='centered')
                _, _, latt_vect1_dist[ii, jj], latt_vect2_dist[ii, jj] = \
                    reduce_vectors(self.lattice_vect1, self.lattice_vect2,
                                   xdist_min[ii, jj], ydist_min[ii, jj], mode='centered')

                xdist_min[jj, ii] = - xdist_min[ii, jj]
                ydist_min[jj, ii] = - ydist_min[ii, jj]
                latt_vect1_dist[jj, ii] = - latt_vect1_dist[ii, jj]
                latt_vect2_dist[jj, ii] = - -latt_vect2_dist[ii, jj]

        return xdist_min, ydist_min, latt_vect1_dist, latt_vect2_dist

    def get_phase_mat(self, xdist_matrix, ydist_matrix):
        """
        Create a matrix of phase factors that should be included on e.g. hoppings or interaction terms between sites i
        and j, based on the phases given by the class.
        :param xdist_matrix: matrix of size nsites x nsites, where M[i,j] is the minimum distance between sites i and j
        :param ydist_matrix:
        :return: phase_mat

        phase_mat:
        """

        nsites = xdist_matrix.shape[0]

        # create phase factors
        phase_mat = np.zeros([nsites, nsites], dtype=np.complex)
        for ii in range(0, nsites):
            for jj in range(0, nsites):
                # phase_mat[ii, jj] = site_phases1[ii] * site_phases1[jj].conj() * site_phases2[ii] * site_phases2[
                #   jj].conj()
                amp1 = np.exp(1j * self.phase1 * (
                    xdist_matrix[ii, jj] * self.reciprocal_periodicity_vect1[0] +
                    ydist_matrix[ii, jj] * self.reciprocal_periodicity_vect1[1]))
                amp2 = np.exp(1j * self.phase2 * (
                    xdist_matrix[ii, jj] * self.reciprocal_periodicity_vect2[0] +
                    ydist_matrix[ii, jj] * self.reciprocal_periodicity_vect2[1]))
                phase_mat[ii, jj] = amp1 * amp2

        if not np.any(phase_mat.imag > 10 ** -self._round_decimals):
            phase_mat = phase_mat.real

        return phase_mat

    def reduce_to_unit_cell(self, xlocs, ylocs, mode='positive'):
        return reduce_vectors(self.periodicity_vect1, self.periodicity_vect2, xlocs, ylocs, mode=mode)

    # #################################
    # Validation functions
    # #################################

    def validate_instance(self):
        """
        Validate if the lattice class instance is correctly formed
        :return:
        """
        if not self.validate_latt_vects():
            return 0

        if not self.validate_periodicity_vects():
            return 0

        # check compatibility or periodicity vectors with lattice vectors
        xred_p1, yred_p1, _, _ = reduce_vectors(self.lattice_vect1, self.lattice_vect2, self.periodicity_vect1[0, 0],
                                                self.periodicity_vect1[1, 0], mode='positive')
        if not np.array_equiv(xred_p1, 0) and np.array_equiv(yred_p1, 0):
            return 0

        xred_p2, yred_p2, _, _ = reduce_vectors(self.lattice_vect1, self.lattice_vect2, self.periodicity_vect2[0, 0],
                                                self.periodicity_vect2[1, 0], mode='positive')
        if not np.array_equiv(xred_p2, 0) and np.array_equiv(yred_p2, 0):
            return 0

        return 1

    def validate_latt_vects(self):
        # validate lattice vectors
        norm1 = np.sqrt(self.lattice_vect1.transpose().dot(self.lattice_vect1))
        norm2 = np.sqrt(self.lattice_vect2.transpose().dot(self.lattice_vect2))
        det = np.linalg.det(np.concatenate((self.lattice_vect1, self.lattice_vect2), 1))

        if np.round(norm1, self._round_decimals) == 0 or \
           np.round(norm2, self._round_decimals) == 0 or \
           np.round(det, self._round_decimals) == 0:
            return 0

        return 1

    def validate_periodicity_vects(self):
        # ensure periodicity vectors exist
        if self.periodicity_vect1 is None or self.periodicity_vect2 is None:
            return 0

        # ensure periodicity vectors are not linearly dependent, if they are non-zero
        # TODO: want to allow periodicity vectors to be zero in some cases ... maybe don't want this test
        norm1 = np.sqrt(self.periodicity_vect1.transpose().dot(self.periodicity_vect1))
        norm2 = np.sqrt(self.periodicity_vect2.transpose().dot(self.periodicity_vect2))
        det = self.periodicity_vect1[0] * self.periodicity_vect2[1] - self.periodicity_vect1[1] * \
                                                                      self.periodicity_vect2[0]
        if not np.round(norm1, self._round_decimals) == 0 and not \
               np.round(norm2, self._round_decimals) == 0 and \
               np.round(det, self._round_decimals) == 0:
            # i.e. if our periodicity vectors are linearly dependent but non-zero
            return 0
        return 1

    # #################################
    # Comparison functions
    # #################################

    def __eq__(self, other):
        if np.array_equal(self.lattice_vect1, other.lattice_vect1) and \
           np.array_equal(self.lattice_vect2, other.lattice_vect2) and \
           np.array_equal(self.periodicity_vect1, other.periodicity_vect1) and \
           np.array_equal(self.periodicity_vect2,  other.periodicity_vect2) and \
           self.phase1 == other.phase1 and \
           self.phase2 == other.phase2:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

# ##########################
# module functions
# ##########################

def get_reciprocal_vects(vect1, vect2):
    """
    Compute the reciprocal vectors. If we call the periodicity vecors a_i and the
    reciprocal vectors b_j, then these should be defined such that dot(a_i, b_j) = delta_{ij}.
    :return: reciprocal_vect1, reciprocal_vect2
    """

    vect1 = ensure_column_vect(vect1)
    vect2 = ensure_column_vect(vect2)

    if not np.array_equal(vect1, np.zeros((2, 1))) and not np.array_equal(vect2, np.zeros((2, 1))):
        A_mat = np.concatenate([vect1.transpose(), vect2.transpose()], 0)
        try:
            inv_a = np.linalg.inv(A_mat)
            reciprocal_vect1 = inv_a[:, 0][:, None]
            reciprocal_vect2 = inv_a[:, 1][:, None]
        except np.linalg.LinAlgError:
            raise Exception('vect1 and vect2 are linearly independent, so their reciprocal vectors could not be computed.')
        # TODO: could catch singular matrix error and give more informative error
    elif np.array_equal(vect1, np.zeros((2, 1))) and not np.array_equal(vect2, np.zeros((2, 1))):
        reciprocal_vect1 = np.zeros((2, 1))

        norm2 = np.sqrt(vect2.transpose().dot(vect2))
        reciprocal_vect2 = vect2 / norm2 ** 2
    elif not np.array_equal(vect1, np.zeros((2, 1))) and np.array_equal(vect2, np.zeros((2, 1))):
        reciprocal_vect2 = np.zeros((2, 1))
        norm1 = np.sqrt(vect1.transpose().dot(vect1))
        reciprocal_vect1 = vect1 / norm1 ** 2
    else:
        reciprocal_vect1 = np.zeros((2, 1))
        reciprocal_vect2 = np.zeros((2, 1))


    return reciprocal_vect1, reciprocal_vect2

def reduce_vectors(vect1, vect2, xlocs, ylocs, mode='positive'):
    """
    Given an arbitrary vector and a pair of basis vectors specifying a periodicity (TODO: sharpend this defn),
    reduce the arbitrary vector to its representative in the Brillouin zone (analog). (TODO: add support for
    either using a symmetric BZ or an always positive BZ).
    :param vect1: size 2 x 1, i.e. a column vector
    :param vect2: size 2 x 1, i.e. a column vector
    :param xlocs:
    :param ylocs:
    :param mode: "positive" or "centered"
    :return: xs_red, xs reduced to lie within the symmetry region
    :return: ys_red, ys reduced to lie within the symmetry region
    :return: n1s number of bvect1's subtracted from vects to get vects_reduced
    :return: n2s number of bvect2's subtracted from vects to get vects_reduced
    """

    # ensure input vectors have the desired format
    periodicity_vect1 = ensure_column_vect(vect1)
    periodicity_vect2 = ensure_column_vect(vect2)
    # norms and dot products of these vectors
    norm1 = np.sqrt(np.sum(periodicity_vect1 * periodicity_vect1))
    norm2 = np.sqrt(np.sum(periodicity_vect2 * periodicity_vect2))
    det = float(periodicity_vect1[0, 0] * periodicity_vect2[1, 0] - periodicity_vect1[1, 0] * periodicity_vect2[0, 0])

    # shape xlocs as desired
    xlocs = ensure_row_vect(xlocs)
    ylocs = ensure_row_vect(ylocs)
    # create a row vector of column vectors. Left multiplying this by a matrix M applies M to each of our column vectors
    vects = np.concatenate([xlocs, ylocs], 0)

    # we want to write each vector v = a*P^a + entries*P^entries
    # if P1 and P2 are orthogonal, this is easy. We simply take the dot product of v and P1, P2
    # if P1 and P2 are not orthogonal, then we change coordinates so that they are.
    # Suppose M is a basis change matrix such that M*P^a = e1 and M*P^entries = e2
    # Then Mv = a*e1 + entries*e2
    # It is easy to see that M^(-1) is a matrix where the first column is P^a and the second is P^entries
    # So we see that
    # M^(1) = | P^b_2   -P^b_1 |
    #         |-P^a_2    P^a_1 | / (P^a_1*P^b_2 - P^b_1*P^a_2)
    # Therefore, a = ( v1 * P^b_2 - v2 * P^b_1) / (P^a_1*P^b_2 - P^b_1*P^a_2)
    #            entries = (-v1 * P^a_2 + v2 * P^a_1 )/ (P^a_1*P^b_2 - P^b_1*P^a_2)
    # If we call the row vectors M^x and M^y, then we have
    # a = M^x \cdot v, M^x = ( P^b_2 -P^b_1) / (P^a_1*P^b_2 - P^b_1*P^a_2)
    # entries = M^y \cdot v, M^y = (-P^a_2  P^a_1) / (P^a_1*P^b_2 - P^b_1*P^a_2)

    if np.round(norm1, 14) == 0:
        # handle case where one of our periodicity vectors is zero
        proj1s = np.zeros([1, vects.shape[1]])
    else:
        if np.round(norm2, 14) == 0:
            # if second periodicity vector is zero, we ignore it
            vect1 = periodicity_vect1.transpose() / norm1 ** 2
        else:
            # otherwise we use the technique described above
            vect1 = np.array([[periodicity_vect2[1, 0], -periodicity_vect2[0, 0]]]) / det

        proj1s = vect1.dot(vects)

    if np.round(norm2, 14) == 0:
        proj2s = np.zeros([1, vects.shape[1]])
    else:
        if np.round(norm1, 14) == 0:
            vect2 = periodicity_vect2.transpose() / norm2 ** 2
        else:
            vect2 = np.array([[-periodicity_vect1[1, 0], periodicity_vect1[0, 0]]]) / det

        proj2s = vect2.dot(vects)

    proj1s = np.round(proj1s, 14)
    proj2s = np.round(proj2s, 14)

    # TODO: add check of this!
    # expect proj1s * periodicity_vect1 + proj2s * periodicity_vect2 = vects

    if mode == 'positive':
        n1s = np.floor(proj1s)
        n2s = np.floor(proj2s)
    elif mode == 'centered':
        # for points exactly half-way between two integers, np.round takes the nearest even number,
        # which is not what we want in this case, because e.g. -0.5 rounds to 0, and 0.5 round to 0.
        # Instead, we want to always round up. So do this manually.
        proj1s[np.mod(proj1s, 1) == 0.5] = proj1s[np.mod(proj1s, 1) == 0.5] + 0.5
        proj2s[np.mod(proj2s, 1) == 0.5] = proj2s[np.mod(proj2s, 1) == 0.5] + 0.5
        n1s = np.round(proj1s)
        n2s = np.round(proj2s)
    else:
        raise Exception

    vects_red = vects - np.kron(n1s, periodicity_vect1) - np.kron(n2s, periodicity_vect2)
    xs_red = vects_red[0, :]
    ys_red = vects_red[1, :]

    return xs_red, ys_red, n1s, n2s

def ensure_column_vect(vector):
    """
    Given an input vector, which may be a 1D numpy array, or a column or row vector, return it as a column vector.

    :param vector: 1D numpy array, or 2D numpy array with one dimensions having size 1.
    :return:
    """

    if vector is None:
        return None

    vect_temp = np.asarray(vector)
    return vect_temp.reshape([vect_temp.size, 1])

def ensure_row_vect(vector):
    """
    Given an input vector, return it as a row vector.

    :param vector: 1D numpy array, or 2D numpy array with one dimensions having size 1.
    :return:
    """

    if vector is None:
        return None
    vect_temp = np.asarray(vector)
    return vect_temp.reshape([1, vect_temp.size])

def ensure_1d_vect(vector):
    """
    Given an input vector, return it as a 1D numpy array.

    :param vector: 1D numpy array, or 2D numpy array with one dimensions having size 1.
    :return:
    """

    if vector is None:
        return None

    vect_temp = np.asarray(vector)
    return vect_temp.reshape([vect_temp.size,])


if __name__ == "__main__":
    # example usage of Geometry class
    import matplotlib.pyplot as plt
    # from ed_geometry import *

    # 8 x 1 chain geometry
    nx = 8
    ny = 1
    phi1 = np.pi/3
    phi2 = 0
    bc_open1 = False
    bc_open2 = True
    gm = Geometry.createSquareGeometry(nx, ny, phi1, phi2, bc_open1, bc_open2)
    gm.dispGeometry()

    # permute 8 x 1 chain
    permutation = [3, 2, 1, 0, 4, 7, 6, 5]
    # permutation = range(geom.nsites - 1, -1, -1)
    gm.permute_sites(permutation)
    gm.dispGeometry()

    # 10 site square
    nr = 3
    nv = 1
    geom_tilted = Geometry.createTiltedSquareGeometry(nr, nv, 0, 0, bc1_open=False, bc2_open=False)
    geom_tilted.dispGeometry()

    # triangular lattice
    n1 = 4
    n2 = 4
    geom_triangle = Geometry.createTriangleGeometry(n1, n2, 0, 0, bc1_open=True, bc2_open=True)
    geom_triangle.dispGeometry()

    # hexagonal lattice
    n1 = 4
    n2 = 4
    geom_hex = Geometry.createHexagonalGeometry(n1, n2, 0, 0, bc1_open=False, bc2_open=False)
    geom_hex.dispGeometry()

    # non-periodic geometry
    xlocs = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    ylocs = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    geom_arb = Geometry.createNonPeriodicGeometry(xlocs, ylocs)
    geom_arb.dispGeometry()

    plt.show()
