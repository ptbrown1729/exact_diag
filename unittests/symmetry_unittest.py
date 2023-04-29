import unittest
import numpy as np
from exact_diag import ed_geometry
from exact_diag import ed_symmetry


class TestSymm(unittest.TestCase):

    def SetUp(self):
        pass

    def test_get_rot_fn(self):
        """
        Test coordinate rotation function
        :return:
        """
        rot_fn_4 = ed_symmetry.getRotFn(4, cx=1, cy=1)
        x = np.array([3.0])
        y = np.array([5.0])
        x_rot, y_rot = rot_fn_4(x, y)

        x_rot_expected = np.array([-3.0])
        y_rot_expected = np.array([3.0])

        self.assertAlmostEqual(np.linalg.norm(x_rot - x_rot_expected), 0, 14)
        self.assertAlmostEqual(np.linalg.norm(y_rot - y_rot_expected), 0, 14)

    def test_get_refl_fn(self):
        """
        Test coordinate reflection function
        :return:
        """
        refl_vect = np.array([[1.0], [1.0]])
        refl_fn = ed_symmetry.getReflFn(refl_vect, cx=-1.0, cy=2.0)
        x = np.array([3.0])
        y = np.array([5.0])
        x_refl, y_refl = refl_fn(x, y)

        x_refl_expected = np.array([2.0])
        y_refl_expected = np.array([6.0])

        self.assertAlmostEqual(np.linalg.norm(x_refl - x_refl_expected), 0, 14)
        self.assertAlmostEqual(np.linalg.norm(y_refl - y_refl_expected), 0, 14)

    def test_get_inversion_fn(self):
        """
        Test coordinate inversion function
        :return:
        """
        inv_fn = ed_symmetry.getInversionFn(cx=0.0, cy=3.0)
        x = np.array([3.0])
        y = np.array([5.0])
        x_inv, y_inv = inv_fn(x, y)

        x_inv_expected = np.array([-3.0])
        y_inv_expected = np.array([1.0])

        self.assertAlmostEqual(np.linalg.norm(x_inv - x_inv_expected), 0, 14)
        self.assertAlmostEqual(np.linalg.norm(y_inv - y_inv_expected), 0, 14)

    def test_get_translation_fn(self):
        """
        Test coordinate translation function
        :return:
        """
        transl_fn = ed_symmetry.getTranslFn(np.array([[1.0], [0.0]]))
        x = np.array([3.0])
        y = np.array([5.0])
        x_transl, y_transl = transl_fn(x, y)

        x_transl_expected = np.array([4.0])
        y_transl_expected = np.array([5.0])

        self.assertAlmostEqual(np.linalg.norm(x_transl - x_transl_expected), 0, 14)
        self.assertAlmostEqual(np.linalg.norm(y_transl - y_transl_expected), 0, 14)

    def test_get_transformed_sites(self):
        """
        Test function which determines how sites are permuted under the action of a transformation
        :return:
        """
        geom = ed_geometry.Geometry.createSquareGeometry(8, 1, 0, 0, 0, 1)
        transl_fn = ed_symmetry.getTranslFn(np.array([[1.0], [0.0]]))

        sites, trans_sites = ed_symmetry.getTransformedSites(transl_fn, range(0, geom.nsites), geom)

        sites_expected = np.arange(0, geom.nsites)
        trans_sites_expected = np.array([1., 2., 3., 4., 5., 6., 7., 0.])

        self.assertTrue(np.array_equal(sites, sites_expected))
        self.assertTrue(np.array_equal(trans_sites, trans_sites_expected))

    def test_findSiteCycles(self):
        """
        Test function which finds cyclese of sites under repeated transformations
        :return:
        """
        geom = ed_geometry.Geometry.createSquareGeometry(4, 4, 0, 0, 1, 1)
        cx, cy = geom.get_center_of_mass()
        permutation = geom.get_sorting_permutation('top_alternating')
        geom.permute_sites(permutation)

        rot_fn = ed_symmetry.getRotFn(4, cx=cx, cy=cy)
        cycles, max_cycle_len = ed_symmetry.findSiteCycles(rot_fn, geom)

        cycles_expected = [[0, 15, 12, 3], [1, 8, 13, 4], [2, 7, 14, 11], [5, 6, 9, 10]]

        self.assertEqual(max_cycle_len, 4)
        self.assertEqual(cycles, cycles_expected)

if __name__ == "__main__":
    unittest.main()
