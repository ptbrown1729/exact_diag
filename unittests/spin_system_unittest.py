import unittest
import numpy as np
from exact_diag.ed_spins import spinSystem
from exact_diag.ed_geometry import Geometry
import exact_diag.ed_symmetry as symm


class TestSpinSys(unittest.TestCase):

    def setUp(self):
        pass

    def test_higher_spin(self):
        """
        Test heisenberg system with spin > 0.5
        :return:
        """
        spin = 2.5

        gm = Geometry.createSquareGeometry(nx_sites=4,
                                           ny_sites=1,
                                           bc1_open=True,
                                           bc2_open=True)
        gm.dispGeometry()

        j1 = 0.3333
        j2 = 0.23623627
        j3 = 0.8434783478
        # -1 to account for difference in definitions
        js = -np.array([[0, j1, j2, j3],
                        [j1, 0, j2, j3],
                        [j2, j2, 0, j3],
                        [j3, j3, j3, 0]])
        ss = spinSystem(gm, jx=js, jy=js, jz=js, spin=spin)
        hamiltonian = ss.createH()

        eig_vals, eig_vects = ss.diagH(hamiltonian)

        # verify these are equal to expected result
        # E = j1 * [s1*(s1+1) + s2*(s2+1)] + j2 * s3*(s3+1) + j3 * s4*(s4+1) + (j2-j1)*s12*(s12+1) +
        # (j3-j2)*s123*(s123+1) - j3 * s1234 * (s1234 + 1)
        eigs_analytic = []
        s1 = spin
        s2 = s1
        s3 = s1
        s4 = s1
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

        max_diff = np.max(np.abs(eigs_analytic - eig_vals))
        self.assertAlmostEqual(max_diff, 0, 12)

    def test_ising_model_3x3(self):
        """
        Test diagonalization of 3x3 ising model with periodic boundary conditions.

        reference: J. Phys. A: Math. Gen 33 6683 (2000).
        "Finite-size scaling in the transverse Ising model on a square lattice" by C J Hamer
        https://doi.org/10.1088/0305-4470/33/38/303

        :return:
        """

        cluster = Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=False, bc2_open=False)
        # paper definition of hamiltonian differs by a factor of two from mine
        jz_critical = -2 * 0.32424925229
        h_transverse = 2 * 1.0
        spin_model = spinSystem(cluster,
                                jx=0.0, jy=0.0, jz=jz_critical,
                                hx=h_transverse, hy=0.0, hz=0.0)
        hamiltonian = spin_model.createH()
        eig_vals, eig_vects = spin_model.diagH(hamiltonian)
        # runner_offset the energy by the number of sites to match Hamiltonian in paper
        min_eig_per_site = eig_vals[0] / spin_model.geometry.nsites + 1
        # paper quote -0.075901188836 ... I get a difference in the 11th-12th decimal place
        # I find 38 instead of 36
        self.assertAlmostEqual(min_eig_per_site, -0.0759011888, 10)

    def test_ising_model_4x4(self):
        """
        Test diagonalization of 4x4 ising model with periodic boundary conditions.

        reference: J. Phys. A: Math. Gen 33 6683 (2000).
        "Finite-size scaling in the transverse Ising model on a square lattice" by C J Hamer
        https://doi.org/10.1088/0305-4470/33/38/303

        :return:
        """

        cluster = Geometry.createSquareGeometry(4, 4, 0, 0, bc1_open=False, bc2_open=False)
        # paper definition of hamiltonian differs by a factor of two from mine
        jz_critical = -2 * 0.32424925229
        h_transverse = 2 * 1.0
        spin_model = spinSystem(cluster,
                                jx=0.0, jy=0.0, jz=jz_critical,
                                hx=h_transverse, hy=0.0, hz=0.0)

        # symmetry swapping up/down spins
        spin_swap_op = spin_model.get_swap_up_down_op()

        # x-translations
        xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
        xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, spin_model.geometry)
        xtransl_op = spin_model.get_xform_op(xtransl_cycles)
        xtransl_op = xtransl_op

        # y-translations
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, spin_model.geometry)
        ytransl_op = spin_model.get_xform_op(ytransl_cycles)
        ytransl_op = ytransl_op

        symm_projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                               max_cycle_len_transly)

        eig_vals_sectors = []
        full_proj_list = []
        hfull = spin_model.createH()
        for ii, proj in enumerate(symm_projs):
            spin_swap_proj_op = proj * spin_swap_op * proj.conj().transpose()

            # char_table = np.array([[1, 1], [1, -1]])
            # ccs = [[sp.eye(spin_swap_proj_op.shape[0], format="csr")], [spin_swap_proj_op]]
            swap_projs, phase = symm.getZnProjectors(spin_swap_proj_op, 2)

            for jj, sproj in enumerate(swap_projs):
                full_proj_list.append(sproj * proj)

                h_sector = sproj * proj * hfull * proj.conj().transpose() * sproj.conj().transpose()
                eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector, print_results=True)
                eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        min_eig_per_site = eigs_all_sectors[0] / spin_model.geometry.nsites + 1

        self.assertAlmostEqual(min_eig_per_site, -0.066430308096, 11)

    @unittest.skip("time and memory intensive")
    def test_ising_model_5x5(self):
        """
        Test diagonalization of 5x5 ising model with periodic boundary conditions.

        reference: J. Phys. A: Math. Gen 33 6683 (2000).
        "Finite-size scaling in the transverse Ising model on a square lattice" by C J Hamer
        https://doi.org/10.1088/0305-4470/33/38/303

        :return:
        """

        cluster = Geometry.createSquareGeometry(5, 5, 0, 0, bc1_open=False, bc2_open=False)
        # paper definition of hamiltonian differs by a factor of two from mine
        jz_critical = -2 * 0.32669593806
        h_transverse = 2 * 1.0
        spin_model = spinSystem(cluster,
                                jx=0.0, jy=0.0, jz=jz_critical,
                                hx=h_transverse, hy=0.0, hz=0.0)

        # symmetry swapping up/down spins
        spin_swap_op = spin_model.get_swap_up_down_op()

        # x-translations
        xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
        xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, spin_model.geometry)
        xtransl_op = spin_model.get_xform_op(xtransl_cycles)
        xtransl_op = xtransl_op

        # y-translations
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, spin_model.geometry)
        ytransl_op = spin_model.get_xform_op(ytransl_cycles)
        ytransl_op = ytransl_op

        symm_projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                               max_cycle_len_transly)

        eig_vals_sectors = []
        full_proj_list = []
        hfull = spin_model.createH()
        for ii, proj in enumerate(symm_projs):
            spin_swap_proj_op = proj * spin_swap_op * proj.conj().transpose()

            # char_table = np.array([[1, 1], [1, -1]])
            # ccs = [[sp.eye(spin_swap_proj_op.shape[0], format="csr")], [spin_swap_proj_op]]
            swap_projs, phase = symm.getZnProjectors(spin_swap_proj_op, 2)

            for jj, sproj in enumerate(swap_projs):
                full_proj_list.append(sproj * proj)

                h_sector = sproj * proj * hfull * proj.conj().transpose() * sproj.conj().transpose()
                eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector, print_results=True)
                eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        min_eig_per_site = eigs_all_sectors[0] / spin_model.geometry.nsites + 1

        self.assertAlmostEqual(min_eig_per_site, -0.064637823298, 11)

    def test_three_by_three_xy_model(self):
        """
        Test exact diagonalization of 3x3 xy model with periodic boundary conditions

        reference: J. Phys. A: Math. Gen 32 51 (1999).
        "Finite-size scaling in the spin-1/2 xy model on a square lattice" by C J Hamer et al.
        """
        cluster = Geometry.createSquareGeometry(3,
                                                3,
                                                bc1_open=False,
                                                bc2_open=False)
        spin_model = spinSystem(cluster,
                                jx=-1.0, jy=-1.0, jz=0.0,
                                hx=0.0, hy=0.0, hz=0.0)
        hamiltonian = spin_model.createH()
        eig_vals, eig_vects = spin_model.diagH(hamiltonian)
        min_eig_per_site = eig_vals[0] / spin_model.geometry.nsites
        self.assertAlmostEqual(min_eig_per_site, -1.149665828185, 12)

    @unittest.skip("This test is time intensive.")
    def test_four_by_four_xy_model_symm(self):
        """
        Test exact diagonalization of 4x4 xy model with periodic boundary conditions

        reference: J. Phys. A: Math. Gen 32 51 (1999).
        "Finite-size scaling in the spin-1/2 xy model on a square lattice" by C J Hamer et al.
        """

        cluster = Geometry.createSquareGeometry(4,
                                                4,
                                                bc1_open=False,
                                                bc2_open=False)

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx=-1.0, jy=-1.0, jz=0.0,
                                hx=0.0, hy=0.0, hz=0.0)

        # x-translations
        xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
        xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, spin_model.geometry)
        xtransl_op = spin_model.get_xform_op(xtransl_cycles)
        xtransl_op = xtransl_op

        # y-translations
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, spin_model.geometry)
        ytransl_op = spin_model.get_xform_op(ytransl_cycles)
        ytransl_op = ytransl_op

        symm_projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                               max_cycle_len_transly)

        eig_vals_sectors = []
        hfull = spin_model.createH()
        for ii, proj in enumerate(symm_projs):
            h_sector = proj * hfull * proj.conj().transpose()
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector, print_results=True)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        min_eig_per_site = eigs_all_sectors[0] / spin_model.geometry.nsites
        self.assertAlmostEqual(min_eig_per_site, -1.124972697436, 12)

    def test_fourfold_rotation_symm_3x3(self):
        """
        Test rotation symmetry by diagonalizing random spin system with and without using it
        :return:
        """
        cluster = Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=False, bc2_open=False)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        cx, cy = spin_model.geometry.get_center_of_mass()
        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)
        symm_projs, _ = symm.getZnProjectors(rot_op, 4)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_d2_symm_3x2(self):
        """
        Test d2 symmetry by diagonalizing random spin system with and without it
        :return:
        """
        cluster = Geometry.createSquareGeometry(3, 2, 0, 0, bc1_open=False, bc2_open=False)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        cx, cy = spin_model.geometry.get_center_of_mass()
        # pi/2 rotation function
        rot_fn = symm.getRotFn(2, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        # reflection about y-axis
        refl_fn = symm.getReflFn(np.array([0, 1]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_refl = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD2Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_d3_symm_3sites(self):
        """
        Test d3 symmetry by diagonalizing random spin system with and without it
        :return:
        """
        # adjacency_mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        cluster = Geometry.createRegularPolygonGeometry(3)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        rot_fn = symm.getRotFn(3)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]))
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD3Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 12)

    def test_d3_symm_6sites(self):
        adjacency_mat = np.array([[0, 1, 0, 1, 0, 0],
                                  [1, 0, 1, 1, 1, 0],
                                  [0, 1, 0, 0, 1, 0],
                                  [1, 1, 0, 0, 1, 1],
                                  [0, 1, 1, 1, 0, 1],
                                  [0, 0, 0, 1, 1, 0]])
        cluster = Geometry.createNonPeriodicGeometry(xlocs=(0, 1, 2, 0.5, 1.5, 1),
                                                     ylocs=(0, 0, 0, np.sqrt(3)/2, np.sqrt(3)/2, np.sqrt(3)),
                                                     adjacency_mat=adjacency_mat)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        # triangle center
        cx, cy = 1, 1 / np.sqrt(3)
        rot_fn = symm.getRotFn(3, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD3Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 12)

    def test_d4_symm_4sites(self):
        cluster = Geometry.createRegularPolygonGeometry(4)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        rot_fn = symm.getRotFn(4)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]))
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD4Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 12)

    def test_d5_symm_5sites(self):
        cluster = Geometry.createRegularPolygonGeometry(5)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        # triangle center
        rot_fn = symm.getRotFn(5)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]))
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD5Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 14)

    def test_d6_symm_6sites(self):
        cluster = Geometry.createRegularPolygonGeometry(6)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        # triangle center
        rot_fn = symm.getRotFn(6)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]))
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD6Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 13)

    @unittest.skip("not working yet")
    def test_d7_symm_7sites(self):

        cluster = Geometry.createRegularPolygonGeometry(7)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        rot_fn = symm.getRotFn(7)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]))
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD6Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 14)

    def test_d4_symm(self):
        """
        Test d4 symmetry by diagonalizing random spin system with and without it
        :return:
        """
        cluster = Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=False, bc2_open=False)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster,
                                jx, jy, jz,
                                hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        # use rotationl symmetry
        cx, cy = spin_model.geometry.get_center_of_mass()

        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        refl_fn = symm.getReflFn(np.array([0, 1]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getD4Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_translational_symm_3x3(self):
        """
        Test translational symmetry by diagonalizing random 3x3 spin system with and without it
        :return:
        """
        cluster = Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=False, bc2_open=False)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster, jx, jy, jz, hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
        xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, spin_model.geometry)
        xtransl_op = spin_model.get_xform_op(xtransl_cycles)
        xtransl_op = xtransl_op

        # y-translations
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, spin_model.geometry)
        ytransl_op = spin_model.get_xform_op(ytransl_cycles)
        ytransl_op = ytransl_op

        symm_projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op,
                                                               max_cycle_len_translx,
                                                               ytransl_op,
                                                               max_cycle_len_transly)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_full_symm_3x3(self):
        cluster = Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=False, bc2_open=False)
        jx = np.random.rand()
        jy = np.random.rand()
        jz = np.random.rand()
        hx = np.random.rand()
        hy = np.random.rand()
        hz = np.random.rand()

        # diagonalize full hamiltonian
        spin_model = spinSystem(cluster, jx, jy, jz, hx, hy, hz)
        hamiltonian_full = spin_model.createH()
        eig_vals_full, eig_vects_full = spin_model.diagH(hamiltonian_full)

        xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
        xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, spin_model.geometry)
        tx_op = spin_model.get_xform_op(xtransl_cycles)

        # y-translations
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, spin_model.geometry)
        ty_op = spin_model.get_xform_op(ytransl_cycles)

        # rotation
        cx, cy = spin_model.geometry.get_center_of_mass()
        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, spin_model.geometry)
        rot_op = spin_model.get_xform_op(rot_cycles)

        # reflection about y-axis
        refl_fn = symm.getReflFn(np.array([0, 1]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_refl = symm.findSiteCycles(refl_fn, spin_model.geometry)
        refl_op = spin_model.get_xform_op(refl_cycles)

        symm_projs = symm.getC4V_and_3byb3(rot_op, refl_op, tx_op, ty_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = spin_model.createH(projector=proj)
            eig_vals_sector, eig_vects_sector = spin_model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_directions_equivalent(self):
        """
        Test if all directions are equivalent for the spin system by picking three random fields and interactions_4 and
        diagonalizing all six permutations of spin systems which can be created from these.
        The compare their eigenvalues

        :return:
        """
        cluster = Geometry.createSquareGeometry(3, 3, 0, 0, bc1_open=False, bc2_open=False)
        j1 = np.random.rand()
        j2 = np.random.rand()
        j3 = np.random.rand()
        h1 = np.random.rand()
        h2 = np.random.rand()
        h3 = np.random.rand()
        spin_model_list = [spinSystem(cluster, jx=j1, jy=j2, jz=j3, hx=h1, hy=h2, hz=h3),
                           spinSystem(cluster, jx=j2, jy=j3, jz=j1, hx=h2, hy=h3, hz=h1),
                           spinSystem(cluster, jx=j3, jy=j1, jz=j2, hx=h3, hy=h1, hz=h2),
                           spinSystem(cluster, jx=j1, jy=j3, jz=j2, hx=h1, hy=h3, hz=h2),
                           spinSystem(cluster, jx=j2, jy=j1, jz=j3, hx=h2, hy=h1, hz=h3),
                           spinSystem(cluster, jx=j3, jy=j2, jz=j1, hx=h3, hy=h2, hz=h1)
                           ]

        eig_vals_list = []
        for spin_model in spin_model_list:
            hamiltonian = spin_model.createH()
            eig_vals, eig_vects = spin_model.diagH(hamiltonian)
            # cannot get agreement better than 1e-10
            eig_vals_list.append(np.round(eig_vals, 10))

        n_comparisons = len(spin_model_list) - 1
        eig_val_comparisons = np.zeros(n_comparisons)
        for ii in range(0, n_comparisons):
            eig_val_comparisons[ii] = np.abs(eig_vals_list[ii] - eig_vals_list[ii + 1]).max() == 0

        self.assertTrue(np.all(eig_val_comparisons))


if __name__ == "__main__":
    unittest.main()
