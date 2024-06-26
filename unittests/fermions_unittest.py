import unittest
import numpy as np
import exact_diag.ed_geometry as geom
import exact_diag.ed_symmetry as symm
from exact_diag import ed_fermions
from exact_diag.fermi_gas import fg_density, fg_corr


class TestFermions(unittest.TestCase):

    def setUp(self):
        pass

    # single species fermi gas
    def test_fg_energies(self):
        """
        Test energies of single species of non-interacting fermions. Compare with Fermi sea calculation.
        """
        u = 0
        t = 1
        nsites = 10
        temps = np.logspace(-1, 1, 30)

        # fermi sea calculation
        kxs = 2 * np.pi * np.arange(0, nsites) / nsites
        dispersion = - 2 * t * np.cos(kxs)
        energy_exps = np.zeros(len(temps))
        for ii, temp in enumerate(temps):
            # two for two spin states
            energy_exps[ii] = np.sum(np.divide(dispersion, np.exp(dispersion / temp) + 1))

        # fermions calculation
        cluster = geom.Geometry.createSquareGeometry(nsites,
                                                     1,
                                                     0,
                                                     0,
                                                     bc1_open=False,
                                                     bc2_open=True)
        spinless_fermions = ed_fermions.fermions(cluster, u, t, nspecies=1)
        hamiltonian = spinless_fermions.createH()
        eig_vals, eig_vects = spinless_fermions.diagH(hamiltonian)

        energy_exps_model = spinless_fermions.get_exp_vals_thermal(eig_vects,
                                                                   hamiltonian,
                                                                   eig_vals,
                                                                   temps,
                                                                   False)

        max_diff = np.abs(energy_exps - energy_exps_model).max()

        self.assertAlmostEqual(max_diff, 0, 12)

    def test_fg_density(self):
        """
        compare single component fermi gas density calculated using ED versus expected result
        :return:
        """

        # hopping
        t = 1
        # temperatures
        temps = np.logspace(-1, 1, 30) * t
        temps = np.concatenate((np.array([0.]), temps))
        betas = np.divide(1, temps)
        betas[temps == 0] = np.inf

        # mus
        mus = np.linspace(-4, 4, 30)

        # geometry
        gm = geom.Geometry.createSquareGeometry(8,
                                                1,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=True)

        ed_dens = np.zeros((mus.size, temps.size))
        fg_dens = np.zeros((mus.size, temps.size))
        for ii, mu in enumerate(mus):
            # ED
            sf = ed_fermions.fermions(gm,
                                      0,
                                      t,
                                      mus=mu,
                                      us_same_species=0,
                                      potentials=0,
                                      nspecies=1)
            ham = sf.createH(print_results=False)
            eig_vals, eig_vects = sf.diagH(ham, print_results=False)

            ed_dens[ii, :], _ = sf.get_thermal_exp_sites(eig_vects,
                                                         eig_vals,
                                                         sf.n_op,
                                                         0,
                                                         temps,
                                                         sites=[0],
                                                         format="boson")

            # non-interacting fg calc
            fg_dens[ii, :] = fg_density(betas, mu, nsites=gm.nsites, dim='1d')

        max_diff = np.abs(fg_dens - ed_dens).max()
        self.assertAlmostEqual(max_diff, 0, 12)

    def test_fg_correlations(self):
        """
        compare single component fermi gas correlations calculated using ED versus expected result
        :return:
        """
        # hopping
        t = 1
        # temperatures
        temps = np.logspace(-1, 1, 30) * t
        temps = np.concatenate((np.array([0.]), temps))
        # temps = np.array([0.])
        betas = np.divide(1, temps)
        betas[temps == 0] = np.inf

        # mus
        mus = np.linspace(-4, 4, 30)

        # geometry
        gm = geom.Geometry.createSquareGeometry(8,
                                                1,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=True)

        # solve at each mu
        ed_corr = np.zeros((mus.size, temps.size))
        fgc = np.zeros((mus.size, temps.size))
        for ii, mu in enumerate(mus):
            # ED
            sf = ed_fermions.fermions(gm, 0, t, mus=mu, us_same_species=0, potentials=0, nspecies=1)
            ham = sf.createH(print_results=False)
            eig_vals, eig_vects = sf.diagH(ham, print_results=False)

            exps, _ = sf.get_thermal_exp_sites(eig_vects,
                                               eig_vals,
                                               sf.n_op,
                                               0,
                                               temps,
                                               projector=sf.n_projector,
                                               sites=[0, 1],
                                               format="boson")
            corrs, _, _ = sf.get_thermal_corr_sites(eig_vects,
                                                    eig_vals,
                                                    0,
                                                    0,
                                                    sf.n_op,
                                                    sf.n_op,
                                                    temps,
                                                    sites1=np.array([0]),
                                                    sites2=np.array([1]),
                                                    projector=sf.n_projector,
                                                    format="boson")
            ed_corr[ii, :] = corrs - exps[0, :] * exps[1, :]

            # non-interacting fg calc
            fgc[ii, :] = fg_corr(betas, mu, nsites=gm.nsites, dim='1d')

        max_diff = np.abs(fgc - ed_corr).max()
        self.assertAlmostEqual(max_diff, 0, 12)

    @unittest.skip("not finished.")
    def test_heisenberg(self):
        """"
         Spinless fermion model
        \sum_<i,j> t (c^\dag_i c_j + h.c.) + U \sum_i (n_i - 0.5) * (n_{i+1} - 0.5)
        maps to heisenberg model
        \sum_<i,j> J * (S^x_i \cdot S^x_j + S^y_i \cdot S^y_j) + J_z * S^z_i \cdot S^z_j
        J = 2 * t
        J_z = U
        """

        # TODO: finish this ... need to get comparison working ...
        t = -0.5
        U = 1.
        mu = U
        gm = geom.Geometry.createSquareGeometry(3,
                                                3,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=False)
        sf = ed_fermions.fermions(gm, 0, t, mus=mu, us_same_species=U, potentials=0, nspecies=1)
        ham = sf.createH()
        eig_vals, eig_vects = sf.diagH(ham, print_results=False)

        offset = 0.25 * U * gm.nsites
        eig_vals = eig_vals + offset

    # hubbard
    def test_hubbard_atomic_limit(self):
        """
        Test Hubbard model with zero tunneling. Compare with atomic limit calculation
        :return:
        """
        u = 8
        t = 0
        temps = np.logspace(-1, 1, 30)

        # atomic limit calculation for one site
        z = 3 + np.exp(-np.divide(1.0, temps) * u)
        energy_exps = u * np.divide(np.exp(-np.divide(1.0, temps) * u), z)

        # ed calculation
        cluster = geom.Geometry.createSquareGeometry(1,
                                                     1,
                                                     0,
                                                     0,
                                                     bc1_open=True,
                                                     bc2_open=True)
        hubbard_model = ed_fermions.fermions(cluster, u, t, ns=np.array([1, 1]))
        hamiltonian = hubbard_model.createH()
        eig_vals, eig_vects = hubbard_model.diagH(hamiltonian)

        energy_exps_model = np.zeros(len(temps))
        for ii, temp in enumerate(temps):
            energy_exps_model[ii] = hubbard_model.get_exp_vals_thermal(eig_vects,
                                                                       hamiltonian,
                                                                       eig_vals,
                                                                       temp,
                                                                       False)

        max_diff = np.abs(energy_exps - energy_exps_model).max()
        self.assertAlmostEqual(max_diff, 0, 12)

    def test_hubbard_non_interacting(self):
        """
        Test Hubbard system with U=0. Compare with explicit Fermi sea calculation.
        :return:
        """
        u = 0
        t = 1
        nsites = 5
        temps = np.logspace(-1, 1, 30)

        # fermi sea calculation
        kxs = 2 * np.pi * np.arange(0, nsites) / nsites
        dispersion = - 2 * t * np.cos(kxs)
        energy_exps = np.zeros(len(temps))
        for ii, temp in enumerate(temps):
            # two for two spin states
            energy_exps[ii] = 2 * np.sum(np.divide(dispersion, np.exp(dispersion / temp) + 1))

        # ed calculation
        cluster = geom.Geometry.createSquareGeometry(nsites,
                                                     1,
                                                     0,
                                                     0,
                                                     bc1_open=False,
                                                     bc2_open=True)
        hubbard_model = ed_fermions.fermions(cluster, u, t)
        hamiltonian = hubbard_model.createH()
        eig_vals, eig_vects = hubbard_model.diagH(hamiltonian)

        energy_exps_model = hubbard_model.get_exp_vals_thermal(eig_vects,
                                                               hamiltonian,
                                                               eig_vals,
                                                               temps,
                                                               False)

        max_diff = np.abs(energy_exps - energy_exps_model).max()

        self.assertAlmostEqual(max_diff, 0, 12)

    def test_two_sites(self):
        """
        Test two-site Hubbard system with open boundary conditions and no restriction on particle number.
        :return:
        """
        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(2,
                                                1,
                                                0,
                                                0,
                                                bc1_open=True,
                                                bc2_open=True)
        model = ed_fermions.fermions(gm, U, t, ns=None)
        hamiltonian = model.createH()
        eig_vals, eig_vects = model.diagH(hamiltonian)

        # vacuum, E = 0
        # one up, E = -t, +t
        # one down, E = -t, +t
        # two ups, E = 0
        # two downs, E = 0
        # two ups and one down, E = U - t, U + t
        # two downs and one up, E = U - t, U + t
        # two ups and two downs, E = 2*U
        # one up and one down subspace:
        # symm combination of doublon site 0 and site 1, E = 8
        # other three states, E =
        # TODO: analytic expression for this remaining subspace
        val0 = 0
        val1 = 0.5 * (U + np.sqrt(U ** 2 + 16 * t ** 2))
        val2 = 0.5 * (U - np.sqrt(U ** 2 + 16 * t ** 2))

        expected_eig_vals = np.array([-t, -t, val0, val1, val2, 0., 0., 0., t, t, U - t, U - t, U, U + t, U + t, 2 * U])
        expected_eig_vals.sort()
        max_diff = np.abs(eig_vals - expected_eig_vals).max()

        self.assertAlmostEqual(max_diff, 0, 13)

    def test_two_sites_numbersubspace(self):
        """
        Test 2-site hubbard system with open boundary conditions and 1 atom of each spin species
        :return:
        """
        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(2,
                                                1,
                                                0,
                                                0,
                                                bc1_open=True,
                                                bc2_open=True)
        model = ed_fermions.fermions(gm, U, t, ns=np.array([1, 1]))
        hamiltonian = model.createH(projector=model.n_projector)
        eig_vals, eig_vects = model.diagH(hamiltonian)

        expected_eig_vals = np.array([0,
                                      0.5 * (U + np.sqrt(U ** 2 + 16 * t ** 2)),
                                      0.5 * (U - np.sqrt(U ** 2 + 16 * t ** 2)),
                                      U])
        expected_eig_vals.sort()
        max_diff = np.abs(eig_vals - expected_eig_vals).max()

        self.assertAlmostEqual(max_diff, 0, 13)

    # hubbard with symmetries
    def test_c4_symmetry_3by3(self):
        """
        Test fourfold rotational symmetry (generated by 90 degree rotation) on a 3x3 Hubbard cluster with open
         boundary conditions.
        :return:
        """
        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(3,
                                                3,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=False)
        model = ed_fermions.fermions(gm, U, t, ns=np.array([1, 1]))

        # no symmetry
        hamiltonian_full = model.createH(projector=model.n_projector)
        eig_vals_full, eig_vects_full = model.diagH(hamiltonian_full)

        # use rotationl symmetry
        cx, cy = model.geometry.get_center_of_mass()
        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, model.geometry)
        rot_op = model.n_projector.dot(model.get_xform_op(rot_cycles).dot(model.n_projector.conj().transpose()))
        symm_projs, _ = symm.getZnProjectors(rot_op, 4)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = model.createH(projector=proj.dot(model.n_projector))
            eig_vals_sector, eig_vects_sector = model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_d4_symmetry_3by3(self):
        """
        Test D4 symmetry (generated by 90 degree rotation and a reflection) on a 3x3 Hubbard cluster with open
         boundary conditions.
        :return:
        """

        # todo: do all number sectors
        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(3,
                                                3,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=False)
        model = ed_fermions.fermions(gm, U, t, ns=np.array([1, 2]))

        # no symmetry
        hamiltonian_full = model.createH(projector=model.n_projector)
        eig_vals_full, eig_vects_full = model.diagH(hamiltonian_full)

        # use rotationl symmetry
        cx, cy = model.geometry.get_center_of_mass()
        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, model.geometry)
        rot_op = model.n_projector.dot(model.get_xform_op(rot_cycles).dot(model.n_projector.conj().transpose()))

        # reflection symmetry
        refl_fn = symm.getReflFn(np.array([[0], [1]]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_refl = symm.findSiteCycles(refl_fn, model.geometry)
        refl_op = model.n_projector.dot(model.get_xform_op(refl_cycles).dot(model.n_projector.conj().transpose()))

        symm_projs = symm.getD4Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = model.createH(projector=proj.dot(model.n_projector))
            eig_vals_sector, eig_vects_sector = model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_d2_symmetry_3by3(self):
        """
        Test D2 symmetry (generated by 180 degree rotation and a reflection) on a 3x3 Hubbard cluster
        with open boundary conditions.
        :return:
        """

        # todo: do all number sectors

        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(3,
                                                3,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=False)
        model = ed_fermions.fermions(gm, U, t, ns=np.array([3, 4]), nspecies=2)

        # no symmetry
        hamiltonian_full = model.createH(projector=model.n_projector)
        eig_vals_full, eig_vects_full = model.diagH(hamiltonian_full)

        # rotational symmetry
        cx, cy = model.geometry.get_center_of_mass()
        rot_fn = symm.getRotFn(2, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, model.geometry)
        rot_op_full = model.get_xform_op(rot_cycles)
        rot_op = model.n_projector.dot(rot_op_full.dot(model.n_projector.conj().transpose()))

        # reflection symmetry
        refl_fn = symm.getReflFn(np.array([[0], [1]]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_refl = symm.findSiteCycles(refl_fn, model.geometry)
        refl_op_full = model.get_xform_op(refl_cycles)
        refl_op = model.n_projector.dot(refl_op_full.dot(model.n_projector.conj().transpose()))

        symm_projs = symm.getD2Projectors(rot_op, refl_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = model.createH(projector=proj.dot(model.n_projector))
            eig_vals_sector, eig_vects_sector = model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_full_symm_3byb3(self):
        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(3,
                                                3,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=False)

        model = ed_fermions.fermions(gm, U, t, ns=np.array([2, 2]))

        # no symmetry
        hamiltonian_full = model.createH(projector=model.n_projector)
        eig_vals_full, eig_vects_full = model.diagH(hamiltonian_full)

        # rotational symmetry
        cx, cy = model.geometry.get_center_of_mass()
        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, model.geometry)
        rot_op = model.n_projector.dot(model.get_xform_op(rot_cycles).dot(model.n_projector.conj().transpose()))

        # reflection symmetry
        refl_fn = symm.getReflFn(np.array([[0], [1]]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_refl = symm.findSiteCycles(refl_fn, model.geometry)
        refl_op = model.n_projector.dot(model.get_xform_op(refl_cycles).dot(model.n_projector.conj().transpose()))

        # translational symmetry
        tx_fn = symm.getTranslFn(np.array([[1], [0]]))
        tx_cycles, tx_max = symm.findSiteCycles(tx_fn, model.geometry)
        tx_op_full = model.get_xform_op(tx_cycles)
        tx_op = model.n_projector.dot(tx_op_full.dot(model.n_projector.conj().transpose()))

        ty_fn = symm.getTranslFn((np.array([[0], [1]])))
        ty_cycles, ty_max = symm.findSiteCycles(ty_fn, model.geometry)
        ty_op_full = model.get_xform_op(ty_cycles)
        ty_op = model.n_projector.dot(ty_op_full.dot(model.n_projector.conj().transpose()))

        symm_projs = symm.getC4V_and_3byb3(rot_op, refl_op, tx_op, ty_op)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = model.createH(projector=proj.dot(model.n_projector))
            eig_vals_sector, eig_vects_sector = model.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)

    def test_translation_symmetry_3by3(self):
        """
        Test translational symmetry for a 3x3 Hubbard system with periodic boundary conditions
        :return:
        """
        U = 20 * (np.random.rand() - 0.5)
        t = np.random.rand()

        gm = geom.Geometry.createSquareGeometry(3,
                                                3,
                                                0,
                                                0,
                                                bc1_open=False,
                                                bc2_open=False)
        hubbard = ed_fermions.fermions(gm, U, t, ns=np.array([1, 1]))

        # no symmetry
        hamiltonian_full = hubbard.createH(projector=hubbard.n_projector)
        eig_vals_full, eig_vects_full = hubbard.diagH(hamiltonian_full)

        # with symmetry
        xtransl_fn = symm.getTranslFn(np.array([[1], [0]]))
        xtransl_cycles, max_cycle_len_translx = symm.findSiteCycles(xtransl_fn, hubbard.geometry)
        xtransl_op = hubbard.n_projector * hubbard.get_xform_op(xtransl_cycles) * hubbard.n_projector.conj().transpose()

        # y-translations
        ytransl_fn = symm.getTranslFn(np.array([[0], [1]]))
        ytransl_cycles, max_cycle_len_transly = symm.findSiteCycles(ytransl_fn, hubbard.geometry)
        ytransl_op = hubbard.n_projector * hubbard.get_xform_op(ytransl_cycles) * hubbard.n_projector.conj().transpose()

        symm_projs, kxs, kys = symm.get2DTranslationProjectors(xtransl_op, max_cycle_len_translx, ytransl_op,
                                                               max_cycle_len_transly)

        eig_vals_sectors = []
        for ii, proj in enumerate(symm_projs):
            h_sector = hubbard.createH(projector=proj.dot(hubbard.n_projector))
            eig_vals_sector, eig_vects_sector = hubbard.diagH(h_sector)
            eig_vals_sectors.append(eig_vals_sector)

        # why only accurate to 10 decimal places?
        eigs_all_sectors = np.sort(np.concatenate(eig_vals_sectors))
        max_diff = np.abs(eig_vals_full - eigs_all_sectors).max()

        self.assertAlmostEqual(max_diff, 0, 10)


if __name__ == '__main__':
    unittest.main()
