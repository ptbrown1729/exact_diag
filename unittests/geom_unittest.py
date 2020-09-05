import unittest
import numpy as np
import ed_geometry


class TestGeom(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_red_vects_positive(self):
        """
        Test ed_geometry.reduceVects using mode = "positive"
        :return:
        """
        vect1 = np.array([[3], [1]])
        vect2 = np.array([[0], [4]])

        xlocs, ylocs = np.meshgrid(range(3, 6), range(3, 6))

        xlocs_reduced, ylocs_reduced, vect1_multiplier, vect2_multiplier = \
            ed_geometry.reduce_vectors(vect1, vect2, xlocs, ylocs, mode='positive')

        xlocs_reduced_expected = np.array([0., 1., 2., 0., 1., 2., 0., 1., 2.])
        ylocs_reduced_expected = np.array([2., 2., 2., 3., 3., 3., 0., 4., 4.])
        vect1_multiplier_expected = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        vect2_multiplier_expected = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0.]])

        self.assertTrue(np.array_equal(xlocs_reduced, xlocs_reduced_expected) and
                        np.array_equal(ylocs_reduced, ylocs_reduced_expected) and
                        np.array_equal(vect1_multiplier, vect1_multiplier_expected) and
                        np.array_equal(vect2_multiplier, vect2_multiplier_expected))

    def test_get_red_vects_centered(self):
        """
        Test ed_geometry.reduceVects using mode = "centered"
        :return:
        """
        vect1 = np.array([[3], [1]])
        vect2 = np.array([[0], [4]])

        xlocs, ylocs = np.meshgrid(range(3, 6), range(3, 6))

        xlocs_reduced, ylocs_reduced, vect1_multiplier, vect2_multiplier = \
            ed_geometry.reduce_vectors(vect1, vect2, xlocs, ylocs, mode='centered')

        xlocs_reduced_expected = np.array([0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.])
        ylocs_reduced_expected = np.array([-2.,  2.,  1., -1., -1., -2.,  0.,  0., -1.])
        vect1_multiplier_expected = np.array([[1., 1., 2., 1., 1., 2., 1., 1., 2.]])
        vect2_multiplier_expected = np.array([[1., 0., 0., 1., 1., 1., 1., 1., 1.]])

        self.assertTrue(np.array_equal(xlocs_reduced, xlocs_reduced_expected) and
                        np.array_equal(ylocs_reduced, ylocs_reduced_expected) and
                        np.array_equal(vect1_multiplier, vect1_multiplier_expected) and
                        np.array_equal(vect2_multiplier, vect2_multiplier_expected))

    def test_get_recp_vects(self):
        """
        Test ed_geometry.get_reciprocal_vects
        :return:
        """
        periodicity_vect1 = np.array([3, 0])
        periodicity_vect2 = np.array([0, 3])

        recp_v1, recp_v2 = ed_geometry.get_reciprocal_vects(periodicity_vect1, periodicity_vect2)

        recp_v1_expected = np.array([[1. / 3.], [0.]])
        recp_v2_expected = np.array([[0.], [1. / 3.]])

        correct_recps = np.array_equal(recp_v1, recp_v1_expected) and \
                        np.array_equal(recp_v2, recp_v2_expected)

        self.assertTrue(correct_recps)

    def test_lattice_get_unique_sites_squaregeom(self):
        """
        Test ed_geometry.Lattice.get_unique_sites for a square lattice
        :return:
        """
        latt_vect1 = np.array([1, 0])
        latt_vect2 = np.array([0, 1])
        periodicity_vect1 = np.array([3, 0])
        periodicity_vect2 = np.array([0, 3])
        basis_vects = [np.array([[0.], [0.]])]
        phase1 = 0
        phase2 = 0
        latt = ed_geometry.Lattice(latt_vect1, latt_vect2, basis_vects,
                                   periodicity_vect1, periodicity_vect2, phase1, phase2)

        nsites, xlocs, ylocs = latt.get_unique_sites()

        nsites_expected = 9
        xlocs_expected = np.array([0., 0., 0., 1., 1., 1., 2., 2., 2.])
        ylocs_expected = np.array([0., 1., 2., 0., 1., 2., 0., 1., 2.])
        correct_sites = nsites == nsites_expected and \
                        np.array_equal(xlocs, xlocs_expected) and \
                        np.array_equal(ylocs, ylocs_expected)
        self.assertTrue(correct_sites)

    def test_lattice_get_unique_sites_tiltedgeom(self):
        """
        Test ed_geometry.Lattice.get_unique_sites for a square lattice with periodicity vectors that are not colinear
        with the lattice vectors.
        :return:
        """
        latt_vect1 = np.array([1, 0])
        latt_vect2 = np.array([0, 1])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = np.array([3, 1])
        periodicity_vect2 = np.array([-1, 3])
        phase1 = 0
        phase2 = 0
        latt = ed_geometry.Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1,
                                   periodicity_vect2, phase1, phase2)

        nsites, xlocs, ylocs = latt.get_unique_sites()

        nsites_expected = 10
        xlocs_expected = np.array([0., 0., 0., 0., 1., 1., 1., 2., 2., 2.])
        ylocs_expected = np.array([0., 1., 2., 3., 1., 2., 3., 1., 2., 3.])
        correct_sites = nsites == nsites_expected and \
                        np.array_equal(xlocs, xlocs_expected) and \
                        np.array_equal(ylocs, ylocs_expected)
        self.assertTrue(correct_sites)

    def test_lattice_get_unique_sites_triangle_geom(self):
        """
        Test ed_geometry.Lattice.get_unique_sites for a triangular lattice
        :return:
        """
        latt_vect1 = np.array([1, 0])
        latt_vect2 = np.array([0.5, np.sqrt(3)/2])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = 3 * latt_vect1
        periodicity_vect2 = 3 * latt_vect2
        phase1 = 0
        phase2 = 0
        latt = ed_geometry.Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1,
                                   periodicity_vect2, phase1, phase2)

        nsites, xlocs, ylocs = latt.get_unique_sites()

        nsites_expected = 9
        xlocs_expected = np.array([0., 0.5, 1., 1., 1.5, 2., 2., 2.5, 3.])
        ylocs_expected = np.array([0., np.sqrt(3)/2, 0., np.sqrt(3), np.sqrt(3)/2,
                                   0., np.sqrt(3), np.sqrt(3)/2, np.sqrt(3)])

        self.assertTrue(nsites == nsites_expected and
                        np.round(np.abs(xlocs - xlocs_expected).max(), 12) == 0 and
                        np.round(np.abs(ylocs - ylocs_expected).max(), 12) == 0)

    def test_lattice_get_reduced_distance(self):
        """
        Test ed_geometry.Lattice.get_reduced_distances for a 1D chain
        :return:
        """
        num_sites = 7

        latt_vect1 = np.array([1, 0])
        latt_vect2 = np.array([0, 1])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = np.array([num_sites, 0])
        periodicity_vect2 = np.array([0, 0])
        phase1 = 0.333
        phase2 = 0.
        bc1_open = False
        bc2_open = True
        latt = ed_geometry.Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1,
                                   periodicity_vect2, phase1, phase2)

        nsites, xlocs, ylocs = latt.get_unique_sites()
        # xdist_min_mat, ydist_min_mat, dist_reduced_multiplicity = \
        #     latt.get_reduced_distance(xlocs, ylocs)
        xdist_min_mat, ydist_min_mat, _, _ = latt.get_reduced_distance(xlocs, ylocs)

        xi, xj = np.meshgrid(np.arange(0, num_sites), np.arange(0, num_sites))
        xdist_exp = xj - xi
        xdist_exp[xdist_exp < -num_sites/2.] = xdist_exp[xdist_exp < -num_sites/2.] + num_sites
        xdist_exp[xdist_exp > num_sites/2.] = xdist_exp[xdist_exp > num_sites/2.] - num_sites

        self.assertTrue(np.array_equal(xdist_min_mat, xdist_exp))

    @unittest.skip('Not finished!')
    def test_lattice_get_phase_mat(self):
        latt_vect1 = np.array([1, 0])
        latt_vect2 = np.array([0, 0])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = np.array([6, 0])
        periodicity_vect2 = np.array([0, 0])
        phase1 = 0.333
        phase2 = 0.
        bc1_open = False
        bc2_open = True
        latt = ed_geometry.Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1,
                                   periodicity_vect2, phase1, phase2)

        nsites, xlocs, ylocs = latt.get_unique_sites()
        xdist_min_mat, ydist_min_mat, _, _ = \
            latt.get_reduced_distance(xlocs, ylocs)

        phase_mat = latt.get_phase_mat(xdist_min_mat, ydist_min_mat)

        phase_mat_expected = 0
        #phase_mat_expected = np.exp(1j * phase1 * )

        self.assertTrue(np.array_equal(phase_mat, phase_mat_expected))

    def test_lattice_reduce_to_unit_cell(self):
        """
        Test ed_geometry.Lattice.reduce_to_unit_cell
        :return:
        """
        latt_vect1 = np.array([1, 0])
        latt_vect2 = np.array([0, 1])
        basis_vects = [np.array([[0.], [0.]])]
        periodicity_vect1 = np.array([3, 1])
        periodicity_vect2 = np.array([-1, 3])
        phase1 = 0
        phase2 = 0
        latt = ed_geometry.Lattice(latt_vect1, latt_vect2, basis_vects, periodicity_vect1, periodicity_vect2, phase1, phase2)

        xlocs_test = [0., 1., 2., 3., 4., 5., 6., 7., 8.]
        ylocs_test = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        xs_reduced, ys_reduced, n1s, n2s = latt.reduce_to_unit_cell(xlocs_test, ylocs_test, mode='positive')

        xs_reduced_expected = np.array([0., 0., 1., 2., 0., 1., 2., 0., 1.])
        ys_reduced_expected = np.array([0., 3., 3., 3., 2., 2., 2., 1., 1.])
        n1s_expected = np.array([[0., 0., 0, 0., 1., 1., 1., 2., 2.]])
        n2s_expected = np.array([[0., -1., -1., -1., -1., -1., -1., -1., -1.]])

        self.assertTrue(np.array_equal(xs_reduced, xs_reduced_expected) and
                        np.array_equal(ys_reduced, ys_reduced_expected) and
                        np.array_equal(n1s, n1s_expected) and
                        np.array_equal(n2s, n2s_expected))

    def test_geom_get_center_of_mass(self):
        gm = ed_geometry.Geometry.createSquareGeometry(3, 3, 0, 0, 1, 1)
        cx, cy = gm.get_center_of_mass()
        self.assertTrue(cx == 1.0 and cy == 1.0)

    def test_geom_get_sorting_permutation(self):
        gm = ed_geometry.Geometry.createSquareGeometry(3, 3, 0, 0, 1, 1)
        permutation = gm.get_sorting_permutation(sorting_mode='top_alternating')
        permutation_expected = np.array([2., 5., 8., 7., 4., 1., 0., 3., 6.])
        self.assertTrue(np.array_equal(permutation, permutation_expected))

    def test_geom_get_site_distance(self):
        gm = ed_geometry.Geometry.createSquareGeometry(3, 3, 0, 0, 0, 0)
        xdist, ydist = gm.get_site_distance()

        xdist_expected = np.array([[ 0.,  0.,  0., -1., -1., -1., -2., -2., -2.],
                                   [ 0.,  0.,  0., -1., -1., -1., -2., -2., -2.],
                                   [ 0.,  0.,  0., -1., -1., -1., -2., -2., -2.],
                                   [ 1.,  1.,  1.,  0.,  0.,  0., -1., -1., -1.],
                                   [ 1.,  1.,  1.,  0.,  0.,  0., -1., -1., -1.],
                                   [ 1.,  1.,  1.,  0.,  0.,  0., -1., -1., -1.],
                                   [ 2.,  2.,  2.,  1.,  1.,  1.,  0.,  0.,  0.],
                                   [ 2.,  2.,  2.,  1.,  1.,  1.,  0.,  0.,  0.],
                                   [ 2.,  2.,  2.,  1.,  1.,  1.,  0.,  0.,  0.]])

        ydist_expected = np.array([[ 0., -1., -2.,  0., -1., -2.,  0., -1., -2.],
                                   [ 1.,  0., -1.,  1.,  0., -1.,  1.,  0., -1.],
                                   [ 2.,  1.,  0.,  2.,  1.,  0.,  2.,  1.,  0.],
                                   [ 0., -1., -2.,  0., -1., -2.,  0., -1., -2.],
                                   [ 1.,  0., -1.,  1.,  0., -1.,  1.,  0., -1.],
                                   [ 2.,  1.,  0.,  2.,  1.,  0.,  2.,  1.,  0.],
                                   [ 0., -1., -2.,  0., -1., -2.,  0., -1., -2.],
                                   [ 1.,  0., -1.,  1.,  0., -1.,  1.,  0., -1.],
                                   [ 2.,  1.,  0.,  2.,  1.,  0.,  2.,  1.,  0.]])

        self.assertTrue(np.array_equal(xdist, xdist_expected) and
                        np.array_equal(ydist, ydist_expected))
        pass

    def test_geom_get_neighbors_by_distance(self):
        pass

    @unittest.skip('Not finished!')
    def test_geom_permute_sites(self):
        gm = ed_geometry.Geometry.createSquareGeometry(3, 3, 0, 0, 1, 1)
        permutation = gm.get_sorting_permutation(sorting_mode='top_alternating')
        gm.permute_sites(permutation)

        xlocs = [0., 1., 2., 2., 1., 0., 0., 1., 2.]
        ylocs = [2., 2., 2., 1., 1., 1., 0., 0., 0.]
        gm_expected = ed_geometry.Geometry.createNonPeriodicGeometry(xlocs, ylocs)
        self.assertTrue(gm == gm_expected)


if __name__ == "__main__":
    unittest.main()
