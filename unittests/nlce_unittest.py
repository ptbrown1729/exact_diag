import unittest
import numpy as np
from exact_diag import ed_geometry as geom
from exact_diag import ed_nlce

class TestNLCE(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_clusters_next_order(self):
        """

        :return:
        """
        cluster_list = [geom.Geometry.createSquareGeometry(2, 1, 0, 0, bc1_open=True, bc2_open=True)]
        lattice_vect1 = np.array([1, 0])
        lattice_vect2 = np.array([0, 1])
        use_symmetry = 1
        cluster_list_next, multiplicity = ed_nlce.get_clusters_next_order(cluster_list, lv1=lattice_vect1,
                                                                          lv2=lattice_vect2, use_symmetry=use_symmetry)
        # TODO: compare cluster lists
        expected_cluster_list = [geom.Geometry.createNonPeriodicGeometry([-1., 0., 1.], [0., 0., 0.]),
                                 geom.Geometry.createNonPeriodicGeometry([0., 1., 0.], [1., 0., 0.])]
        expected_multiplicity = [2, 4]

        cluster_correct = all(c.isequal_adjacency(ce) for c, ce in zip(cluster_list_next, expected_cluster_list)) and \
                          multiplicity == expected_multiplicity
        self.assertTrue(cluster_correct)

    def test_get_all_clusters(self):
        """

        :return:
        """
        max_cluster_order = 4
        lattice_vect1 = np.array([1, 0])
        lattice_vect2 = np.array([0, 1])
        use_symmetry = True

        full_cluster_list, cluster_multiplicities, order_start_indices = \
            ed_nlce.get_all_clusters(max_cluster_order, lv1=lattice_vect1, lv2=lattice_vect2,
                                     use_symmetry=use_symmetry)

        expected_cluster_mults = np.array([1, 2, 2, 4, 2, 8, 4, 1, 4])
        expected_order_start_inds = [0, 1, 2, 4, 9]
        clusters_correct = np.array_equal(cluster_multiplicities, expected_cluster_mults) and \
                           order_start_indices == expected_order_start_inds

        self.assertTrue(clusters_correct)

    def test_get_all_clusters_with_subclusters(self):
        """

        :return:
        """
        max_cluster_order = 4
        clusters_list, cluster_multiplicities, sub_cluster_mult, order_start_indices = \
            ed_nlce.get_all_clusters_with_subclusters(max_cluster_order)

        expected_cluster_mults = np.array([1, 2, 2, 4, 2, 8, 4, 1, 4])
        expected_sub_cluster_mult = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                              [2., 0., 0., 0., 0., 0., 0., 0., 0.],
                                              [3., 2., 0., 0., 0., 0., 0., 0., 0.],
                                              [3., 2., 0., 0., 0., 0., 0., 0., 0.],
                                              [4., 3., 2., 0., 0., 0., 0., 0., 0.],
                                              [4., 3., 1., 1., 0., 0., 0., 0., 0.],
                                              [4., 3., 1., 2., 0., 0., 0., 0., 0.],
                                              [4., 4., 0., 4., 0., 0., 0., 0., 0.],
                                              [4., 3., 0., 2., 0., 0., 0., 0., 0.]])
        expected_order_start_indices = [0, 1, 2, 4, 9]
        clusters_correct = np.array_equal(cluster_multiplicities, expected_cluster_mults) and \
                           np.array_equal(sub_cluster_mult.toarray(), expected_sub_cluster_mult) and \
                           order_start_indices == expected_order_start_indices

        self.assertTrue(clusters_correct)

    def test_get_subclusters_next_order(self):
        """

        :return:
        """
        parent_geometry = geom.Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=True, bc2_open=True)
        cluster_list_next, old_clusters_contained_in_new_clusters = ed_nlce.get_subclusters_next_order(parent_geometry)

        expected_cluster_list = [geom.Geometry.createNonPeriodicGeometry([0], [0]),
                                 geom.Geometry.createNonPeriodicGeometry([0], [1]),
                                 geom.Geometry.createNonPeriodicGeometry([0], [2]),
                                 geom.Geometry.createNonPeriodicGeometry([1], [0]),
                                 geom.Geometry.createNonPeriodicGeometry([1], [1]),
                                 geom.Geometry.createNonPeriodicGeometry([1], [2]),
                                 geom.Geometry.createNonPeriodicGeometry([2], [0]),
                                 geom.Geometry.createNonPeriodicGeometry([2], [1]),
                                 geom.Geometry.createNonPeriodicGeometry([2], [2])]
        expected_old_clusters_contained_in_new_clusters = [[], [], [], [], [], [], [], [], []]

        clusters_correct = all(a==b for a,b in zip(cluster_list_next, expected_cluster_list)) and \
                           all(a==b for a,b in zip(old_clusters_contained_in_new_clusters, expected_old_clusters_contained_in_new_clusters))

        self.assertTrue(clusters_correct)

    def test_get_all_subclusters(self):
        """

        :return:
        """
        parent_geometry = geom.Geometry.createSquareGeometry(2, 2, 0, 0, bc1_open=True, bc2_open=True)
        cluster_orders_list, sub_cluster_indices, sub_cluster_mat = ed_nlce.get_all_subclusters(parent_geometry)

        expected_sub_cluster_indices = [[], [], [], [], [0, 1], [0, 2], [1, 3], [2, 3], [0, 1, 3, 4, 6],
                                        [0, 1, 2, 4, 5], [0, 2, 3, 5, 7], [1, 2, 3, 6, 7],
                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        expected_sub_cluster_mat = np.array([
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
               [1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
               [1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
               [0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]])
        # TODO: add clusters to this comparision
        cluster_correct = sub_cluster_indices == expected_sub_cluster_indices and\
                          np.array_equal(sub_cluster_mat.toarray(), expected_sub_cluster_mat)

        self.assertTrue(cluster_correct)

    def test_get_reduced_subclusters(self):
        """

        :return:
        """
        parent_geometry = geom.Geometry.createSquareGeometry(2, 2, 0, 0, bc1_open=True, bc2_open=True)
        parent_geometry.permute_sites(parent_geometry.get_sorting_permutation())
        clusters_list, sub_cluster_mult, order_start_indices = ed_nlce.get_reduced_subclusters(parent_geometry)

        # expected_cluster_list = [geom.Geometry.createSquareGeometry(1, 1, 0, 0, 1, 1),
        #                          geom.Geometry.createSquareGeometry(2, 1, 0, 0, 1, 1),
        #                          geom.Geometry.createNonPeriodicGeometry(np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])),
        #                          geom.Geometry.createSquareGeometry(2, 2, 0, 0, 1, 1)]
        expected_sub_cluster_mult = np.array([[0., 0., 0., 0.], [2., 0., 0., 0.], [3., 2., 0., 0.], [4., 4., 4., 0.]])
        expected_order_start_indices = [0, 1, 2, 3, 4]

        sub_clusters_correct = np.array_equal(sub_cluster_mult.toarray(), expected_sub_cluster_mult) and \
                               expected_order_start_indices == order_start_indices
        self.assertTrue(sub_clusters_correct)

    def test_get_clusters_rel_by_symmetry(self):
        pass

    def test_get_nlce_exp_val(self):
        pass

    def test_euler_resum(self):
        pass

    def test_wynn_resm(self):
        pass

if __name__ == "__main__":
    unittest.main()

