"""

Character tables
######################
Many function refer to the character table of a given symmetry group. The character table can be interpreted as
follows. The labels in the leftmost column label the irreducible representations. Each row is associated with
a single irreducible representation. The columns are associated with each conjugacy class of the symmetry group.
The members of each conjugacy class are listed at the top of the column.

"""

from time import perf_counter
import numpy as np
import scipy.sparse as sp
from exact_diag.ed_geometry import Geometry

# #################################################
# coordinate transformation functions
# #################################################


def getRotFn(n_rotations: int,
             cx: float = 0.,
             cy: float = 0.):
    """
    Returns a function which performs a coordinate rotation about a given origin

    :param n_rotations: Int, number of rotations required to go 360 deg. Rotation angle = 2pi/NumRots
    :param cx:
    :param cy:
    :return: function
    """
    angle = 2 * np.pi / n_rotations
    # should I do the rounding after computing the values? Or possibly should rely on getTransformedSites.
    # That seems like the right philosophical approach to me...
    #rotation_mat = np.round(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), _round_decimals)
    rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotation_fn = lambda Xs, Ys: rotation_mat.dot(np.concatenate((Xs[None, :] - cx, Ys[None, :] - cy), axis=0)) + np.array([[cx], [cy]])

    return rotation_fn


def getReflFn(reflection_vect: np.ndarray,
              cx: float = 0.,
              cy: float = 0.):
    """
    Returns a function which performs a coordinate reflection about a given axis.

    # TODO: the reflection center implementation is not correct...

    :param reflection_vect: vector about which to do the reflection
    :param cx:
    :param cy:
    :return: reflection_fn:
    """

    # ensure Vect is the correct shape
    reflection_vect = np.asarray(reflection_vect)
    reflection_vect = reflection_vect.reshape([reflection_vect.size])

    if np.array_equal(np.asarray(reflection_vect), np.zeros(2)):
        raise Exception()

    elif reflection_vect[1] == 0:
        angle = np.pi / 2
    else:
        angle = np.arctan(reflection_vect[0] / reflection_vect[1])  # angle between ReflAxis and y-axis

    # rotate coordinates so reflection is about the y-axis
    rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # in this basis, reflection matrix is easy to write down
    reflection_y_axis = np.array([[-1, 0], [0, 1]])
    # reflection_mat = np.round(rotation_mat.transpose().dot(reflection_y_axis.dot(rotation_mat)), _round_decimals)
    reflection_mat = rotation_mat.transpose().dot(reflection_y_axis.dot(rotation_mat))
    reflection_fn = lambda Xs, Ys: reflection_mat.dot(np.concatenate((Xs[None, :] - cx, Ys[None, :] - cy), 0)) + np.array([[cx], [cy]])

    return reflection_fn


def getInversionFn(cx: float = 0.,
                   cy: float = 0.):
    """
    Returns a function that performs inversion of coordinates about a given center.

    :param cx: x coordinate of inversion center
    :param cy: y coordinate of inversion center
    :return:
    """
    inversion_fn = lambda xs, ys: np.concatenate([ -(xs[None, :] - cx) + cx, -(ys[None, :] - cy) + cy])
    return inversion_fn


def getTranslFn(translation_vector):
    """
    Returns a function that performs coordinate translation along a given vector.

    :param translation_vector:
    :return:
    """
    # ensure TransVect is correct shape
    translation_vector = np.asarray(translation_vector)
    translation_vector = translation_vector.reshape([translation_vector.size])

    # moving periodicity to getTransformedSites, so don't include it here also
    # transl_fn = lambda xs, ys: np.round(np.concatenate([xs[None, :] + translation_vector[0], ys[None, :] + translation_vector[1]], 0), _round_decimals)
    transl_fn = lambda xs, ys: np.concatenate([xs[None, :] + translation_vector[0], ys[None, :] + translation_vector[1]], 0)
    return transl_fn

# #################################################
# Functions to determine how sites transform
# #################################################


def getTransformedSites(transform_fn,
                        sites,
                        geom_obj: Geometry,
                        tol: float = 1e-10):
    """
    Determine how sites are permuted under the action of a given transformation.

    :param transform_fn: A function of the form f(x, y) which returns a 2 x n matrix where each column represents the
      transformed position of site at (x,y).
    :param sites: list of initial sites by index e.g. [0, 1, 2, ..., n]
    :param geom_obj: instances of geometry class
    :param tol:
    :return: (initial_sites, transformed_sites)
    """

    # use periodicity vectors to get reduced locations, to avoid problems where two equivalent locations
    # will not evaluate as equal
    if geom_obj.lattice is not None:
        xlocs_red, ylocs_red, _, _ = geom_obj.lattice.reduce_to_unit_cell(geom_obj.xlocs, geom_obj.ylocs, "centered")
        xlocs_red = xlocs_red
        ylocs_red = ylocs_red
    else:
        xlocs_red = geom_obj.xlocs
        ylocs_red = geom_obj.ylocs

    # still need to use actual locations for transform function
    xformed_coords = transform_fn(geom_obj.xlocs, geom_obj.ylocs)
    trans_xlocs = xformed_coords[0, :]
    trans_ylocs = xformed_coords[1, :]
    # also reduce the transformed locations
    if geom_obj.lattice is not None:
        trans_xlocs_red, trans_ylocs_red, _, _ = geom_obj.lattice.reduce_to_unit_cell(trans_xlocs, trans_ylocs, "centered")
        trans_xlocs_red = trans_xlocs_red
        trans_ylocs_red = trans_ylocs_red
    else:
        trans_xlocs_red = trans_xlocs
        trans_ylocs_red = trans_ylocs

    trans_sites = np.zeros(len(sites))

    for ii in range(0, len(sites)):
        # index = np.where((trans_xlocs_red == xlocs_red[ii]) & (trans_ylocs_red == ylocs_red[ii]))
        condition = np.logical_and(np.abs(trans_xlocs_red[ii] - xlocs_red) < tol,
                                   np.abs(trans_ylocs_red[ii] - ylocs_red) < tol)
        index = np.where(condition)
        if index[0].shape[0] == 0:
            print('site %d at x = %0.2f, y = %0.2f did not transform to another site' % (ii, xlocs_red[ii], ylocs_red[ii]))
            raise Exception
        trans_sites[ii] = sites[index[0][0]]
    # TODO: add descriptive error when one site doesn't have a partner under transformation.
    return np.array(sites), trans_sites


def findSiteCycles(transform_fn,
                   geom_obj: Geometry,
                   tol: float = 1e-10):
    """
    Find closed cycles of sites which transform into each other under a given transformation.
    The transformation operator should be unity after NumTransToClose.
    E.g. for pi/2 rotation on a 3x3 lattices we sould have cycles = [[0,2,8,6],[1,5,7,3],[4]]

    :param transform_fn:
    :param geom_obj:
    :param tol:
    :return: (cycles, max_cycle_len)
      cycles is a list of lists, where each list defines a cycle of sites mapped to each other
      by successive transformations. max_cycle_len is the length of the longest cycle
    """

    nsites = geom_obj.nsites
    sites = np.arange(0, nsites)

    # find the number of transformations required to get us_interspecies back to our initial configuration
    # column k of trans_sites is the site labels after performing the transformation k-times
    trans_sites = sites[:, None]
    current_sites = np.zeros(sites.shape)
    max_iter = nsites + 1
    ii = 0
    while not np.array_equal(current_sites, sites) and ii < max_iter:
        ii = ii + 1
        _, current_sites = getTransformedSites(transform_fn, trans_sites[:, ii - 1], geom_obj, tol=tol)
        trans_sites = np.concatenate([trans_sites, current_sites[:, None]], 1)
    max_cycle_len = ii
    if max_cycle_len == max_iter:
        raise Exception("Number of cycles required to close transformation greater than maximum allowed iterations")

    cycles = []
    for ii in range(0, len(sites)):
        if sites[ii] not in [x for cycle in cycles for x in cycle]:

            _, indices = np.unique(trans_sites[ii, :], return_index=True)
            indices.sort()
            if indices.size == 0:
                print("sites do not transform correctly according to this symmetry")
                return [], 0
            cycle = np.ndarray.tolist(np.ndarray.flatten(trans_sites[ii, indices]))
            cycle = list(map(int, cycle))
            cycles.append(cycle)

    return cycles, max_cycle_len

# #################################################
# Functions to determine how states transform
# #################################################


def reduce_symm_projector(p_full,
                          dim,
                          states_related_by_symm,
                          print_results: bool = False):
    """
    Create an operator which projects onto a certain subspace which has definite transformation properties with
    respect to a given transformation operator.

    Typically this function is called on the output of getCyclicProj(), which produces projop_full2full

    :param p_full: Needs to be a csc sparse matrix. Takes a state in the full space and projects it onto
      a state of a given symmetry. Returns a state in the initial basis. Does not reduce the size of the space.
      projOp_full2full * vector does not necessarily produce a normalized vector.
    :param dim: dimension of the irreducible representation in question
    :param states_related_by_symm:
    :param print_results: Whether to print information to the terminal
    :return proj_full2reduced: sparse csr matrix. symm_proj is defined such that
      symm_proj*Op*symm_proj.transpose() is the operator Op in the projected subspace
    """
    if print_results:
        tstart = perf_counter()

    if not sp.isspmatrix_csr(p_full):
        print("warning, projop_full2full was not csr. Converted it to csr.")
        p_full = p_full.tocsr()

    if not sp.isspmatrix_csr(states_related_by_symm):
        print("warning, projop_full2full was not csr. Converted it to csr.")
        states_related_by_symm = states_related_by_symm.tocsr()

    # round to avoid any finite precision issues
    p_full.data[np.abs(p_full.data) < 1e-10] = 0
    p_full.eliminate_zeros()

    # remove any zero rows
    rows_noz = p_full.indptr[:-1] != p_full.indptr[1:]
    p_full = p_full[rows_noz, :]

    # find orthonormal basis for row space of matrix
    if dim == 1:
        # for each row, find first non-zero index
        first_nonzero_col = p_full.indices[p_full.indptr[:-1]]

        # We only need to check if the first indices are unique to find the unique rows.
        # For dim=1, if these are the same then so are the entire columns

        _, unique_indices = np.unique(first_nonzero_col, return_index=True)
        p_red = p_full[unique_indices, :]
    else:
        # can do for the full matrix, but then must be able to convert projector to dense
        # rank_full = np.linalg.matrix_rank(p_full.todense())
        # u, s, vh = np.linalg.svd(p_full.todense(), full_matrices=True, compute_uv=True)
        # p_red_full = sp.csr_matrix(np.round(vh[:rank_full, :], 12))

        # instead of SVD on entire matrix, we can split matrix into sets of states related by symmetry
        # only these states can transform into each other, so can do each of these sectors on its own
        states_related_by_symm = states_related_by_symm[rows_noz, :]

        # get identifier for which sets of states transform into one another under action of
        first_nonzero_col = states_related_by_symm.indices[states_related_by_symm.indptr[:-1]]

        # unique gives column indices of a unique set of basis vectors from which
        # all others can be obtained by applying symmetry operations
        unique, unique_indices, unique_inverse, counts = np.unique(first_nonzero_col, return_index=True, return_inverse=True, return_counts=True)
        # first_indices = unique[unique_inverse]
        # unique = first_indices[unique_indices].

        # get collection of all rows that related by symmetry
        unique_inv_sorted = np.argsort(unique_inverse)

        # construct reduced projector matrix
        # maximum rank, but real rank can be less than this
        p_red = sp.lil_matrix((p_full.shape))
        p_row_counter = 0
        inv_counter = 0

        # loop over subspaces represented by
        for ii in range(len(unique)):
            # all column indices = unique_inv_sorted[inv_counter : inv_counter + counts[ii]]
            # non-zero rows for this the first column vector (which will be identical for the other related vectors)
            cols = states_related_by_symm.indices[states_related_by_symm.indptr[unique_inv_sorted[inv_counter]]:
                                  states_related_by_symm.indptr[unique_inv_sorted[inv_counter] + 1]]

            # get relevant matrices, only keep nonzero rows and columns for this subpsace
            ccols, rrows = np.meshgrid(cols, unique_inv_sorted[inv_counter: inv_counter + counts[ii]])

            mat = p_full[rrows, ccols].todense()
            rank = np.linalg.matrix_rank(mat)
            u, s, vh = np.linalg.svd(mat, full_matrices=True, compute_uv=True)
            mat_red = np.round(vh[:rank, :], 15)

            # put these back in the correct rows for the final projector
            ccols_out, rrows_out = np.meshgrid(cols, np.arange(p_row_counter, p_row_counter + rank))
            p_red[rrows_out, ccols_out] = mat_red

            # advance counters
            inv_counter += counts[ii]
            p_row_counter += rank

        p_red = p_red[:p_row_counter, :].tocsr()

    # normalize each row
    norms = np.sqrt(np.asarray(p_red.multiply(p_red.conj()).sum(axis=1))).flatten()
    p_red = sp.diags(np.reciprocal(norms), 0, format="csr") * p_red

    if print_results:
        tend = perf_counter()
        print("Finding projector took %0.2f s" % (tend - tstart))

    return p_red


def get_symmetry_projectors(character_table,
                            conjugacy_classes,
                            print_results: bool = False):
    """

    :param character_table: each row gives the characters of a different irreducible rep. Each column
      corresponds to a different conjugacy classes
    :param conjugacy_classes: List of lists of conjugacy class elements
    :param print_results:
    :return projs:
    """

    if not validate_char_table(character_table, conjugacy_classes):
        raise Exception("invalid character table/conjugacy class combination")

    # columns (or rows, since orthogonal mat) represent basis states that can be transformed into one another by symmetries
    states_related_by_symm = sum([sum([np.abs(g) for g in cc]) for cc in conjugacy_classes])

    # only need sums over conjugacy classes to build projectors
    class_sums = [sum(cc) for cc in conjugacy_classes]

    projs = [reduce_symm_projector(
             sum([np.conj(ch) * cs for ch, cs in zip(chars, class_sums)]), chars[0], states_related_by_symm, print_results=print_results)
             for chars in character_table]

    # test projector size
    proj_to_dims = np.asarray([p.shape[0] for p in projs]).sum()
    proj_from_dims = projs[0].shape[1]
    if proj_to_dims != proj_from_dims:
        raise Exception("total span of all projectors was %d, but expected %d." % (proj_to_dims, proj_from_dims))

    return projs

# #################################################
# implementations for specific groups
# #################################################


def getZnProjectors(xform_op,
                    n_xforms,
                    print_results: bool = False):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is a cyclic group. Examples of appropriate
    symmetries are translational or pure rotational symmetry.

    Representations of :math:`Z_n`, the nth cyclic group: There are n irreducible representations, and
    all are one dimensional (since Z_n is abelian). Let :math:`z_r` be the nth root of unity given by
    :math:`z_r = \\exp \\left[ -2 \\pi i \\frac{r}{n} \\right] = z_1^r.`
    We can regard :math:`2 \\pi r/n` as the wave-vector (:math:`k_r`) associated with a representation.
    The character table is given by:

    .. csv-table:: Irreducibles representations of :math:`Z_n`
      :header: "Rep", ":math:`E`", ":math:`R`", ":math:`R^2`", "...", ":math:`R^l`", "...", ":math:`R^{n-1}`"

      "0", "1", "1",                    "1",    "...",          "1",   "...",           "1"
      "1", "1", ":math:`z_1`", ":math:`z_1^2`", "...", ":math:`z_1^l`", ...,   ":math:`z_1^{n-1}`"
      "..."
      "p", "1", ":math:`z_p`", ":math:`z_p^2`", "...", ":math:`z_p^l`", ...,   ":math:`z_p^{n-1}`"
      "..."
      "n-1", "..."

    :param xform_op: transformation operator acting on state space
    :param n_xforms: minimum number of transformations to return to the starting condition, i.e. xform_op**n = 1
    :param print_results: if 1, print timing information
    :return: (projs, ks) a list of projectors and a numpy array of ks associated with each projector
    """

    kx, ky = np.meshgrid(range(n_xforms), range(n_xforms))
    char_table = np.round(np.exp(-2*np.pi*1j / n_xforms * kx * ky), 14)

    id = sp.eye(xform_op.shape[0], format="csr")
    conj_classes = [[id]] + [[xform_op**n] for n in range(1, n_xforms)]

    projs = get_symmetry_projectors(char_table, conj_classes, print_results)

    ks = 2 * np.pi * np.arange(0, n_xforms) / n_xforms
    return projs, ks


def get2DTranslationProjectors(translation_op1,
                               n1,
                               translation_op2,
                               n2,
                               print_results: bool = False):
    """

    :param translation_op1:
    :param n1:
    :param translation_op2:
    :param n2:
    :param print_results:
    :return:
    """

    projs1, ks1 = getZnProjectors(translation_op1, n1, print_results)

    all_projs = []
    for proj, k1 in zip(projs1, ks1):
        sub_projs2, ks2 = getZnProjectors(proj * translation_op2 * proj.conj().transpose(), n2)
        all_projs = all_projs + [sub_proj * proj for sub_proj in sub_projs2]

        # check size at each step
        sub_proj_ndims = np.asarray([p.shape[0] for p in sub_projs2]).sum()
        if sub_proj_ndims != proj.shape[0]:
            print("At translation 1 projector %d, translation 2 sub-projector size did not match"
                  " translation 1 projector dimension" % k1)
            raise Exception

    # ks1 = [ka, ka, ..., ka, kb, kb,...]
    ks1 = np.repeat(ks1, n2, 0)
    # ks2 = [ka, kb, kc, ..., ka, kb, kc, ...]
    ks2 = np.reshape(np.kron(np.ones([1, n1]), ks2), n1 * n2)

    # remove empty projectors
    a = [[proj, ii] for proj, ii in zip(all_projs, range(0, len(all_projs))) if proj.size > 0]
    all_projs, allowed_indices = zip(*a)
    ks1 = ks1[list(allowed_indices)]
    ks2 = ks2[list(allowed_indices)]

    return all_projs, ks1, ks2


def getD2Projectors(rot_op,
                    refl_op,
                    print_results: bool = False):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is :math:`C_{2v} = D_2`, the 2nd dihedral group.
    This is the symmetry group of the rectangle, generated by a two-fold rotation and a reflection and
    has 4 element, i.e. :math:`|D_2] = 4`.

    .. csv-table:: Irreducible representations of :math:`D_2`
       :header: "Rep", ":math:`E`", ":math:`R = C_2(z)`", ":math:`sigma = C_2(y)`", ":math:`R*sigma = C_2(x)`"

       ":math:`A`",   "1",  "1",  "1",  "1"
       ":math:`B_1`", "1",  "1", "-1", "-1"
       ":math:`B_2`", "1", "-1", "1",  "-1"
       ":math:`B_3`", "1", "-1", "-1",  "1"

    :param rot_op:
    :param refl_op:
    :param print_results:
    :return:
    """

    # Project onto subspaces associated with each irreducible representation of C_4v
    id = sp.eye(refl_op.shape[0], format="csr")
    char_table = np.array([[1,  1,  1,  1],
                           [1,  1, -1, -1],
                           [1, -1,  1, -1],
                           [1, -1, -1,  1]])
    conj_classes = [[id], [rot_op], [refl_op], [rot_op * refl_op]]
    projs = get_symmetry_projectors(char_table, conj_classes, print_results)

    return projs


def getD3Projectors(rot_op,
                    refl_op,
                    print_results: bool = False):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is :math:`C_{3v} = D_3`, the 3rd dihedral group.
    This is the symmetry group of the triangle, generated by a three-fold rotation and a reflection and
    has 6 element, i.e. :math:`|D_3] = 4`.

    There are three irreducible representations of D_3, and the character table is given by

    .. csv-table:: Irreducible representations of :math:`D_3`
      :header: "Rep", ":math:`E`", ":math:`2C_3 = (R, R^2)`", ":math:`3C_2 = (\\sigma, R\\sigma, R^2\\sigma)`"

      ":math:`A_1`", "1", "1", "1"
      ":math:`A_2`", "1", "1", "-1"
      ":math:`E`", "2", "-1", "0"

    :param rot_op:
    :param refl_op:
    :param print_results:
    :return: list of projection operators on to symmetry subspaces
    """

    # Project onto subspaces associated with each irreducible representation of C_4v
    id = sp.eye(refl_op.shape[0], format="csr")

    char_table = np.array([[1, 1, 1],
                           [1, 1, -1],
                           [2, -1, 0]])
    conj_classes = [[id], [rot_op, rot_op ** 2],
                    [refl_op, rot_op * refl_op, rot_op ** 2 * refl_op]]

    projs = get_symmetry_projectors(char_table, conj_classes, print_results)

    return projs


def getD4Projectors(rot_op,
                    refl_op,
                    print_results: bool = False):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is :math:`C_{4v} = D_4`, the 4th dihedral group.
    This is the symmetry group of the square, generated by a four-fold rotation and a reflection and
    has 8 element.

    There are five irreducible representations of :math:`D_4`.
    Four are one dimensional and one (E) is two dimensional.

    .. csv-table:: Irreducible representations of :math:`D_4`
      :header: "Rep", ":math:`E`", ":math:`(R,R^3)`", ":math:`(R^2)`", ":math:`(\\sigma, R^2\\sigma)`", ":math:`(R \\sigma, R^3 \\sigma)`"
      :align: center

      ":math:`A_1`", "1",  "1", "1",  "1",  "1"
      ":math:`A_2`", "1",  "1", "1", "-1", "-1"
      ":math:`B_1`", "1", "-1", "1",  "1", "-1"
      ":math:`B_2`", "1", "-1", "1", "-1",  "1"
      ":math:`E`",   "2", "0", "-2",  "0",  "0"

    :param rot_op: sparse matrix. fourfold rotation operator, acting on state space.
    :param refl_op: sparse matrix. reflection operator, acting on state space
    :param print_results:
    :return: list of projection operators onto symmetry subspaces
    """
    id = sp.eye(refl_op.shape[0], format="csr")
    char_table = np.array([[1,  1,  1,  1,  1],
                           [1,  1,  1, -1, -1],
                           [1, -1,  1,  1, -1],
                           [1, -1,  1, -1,  1],
                           [2,  0, -2,  0,  0]])
    conj_classes = [[id], [rot_op, rot_op ** 3],
                    [rot_op ** 2],
                    [refl_op, rot_op ** 2 * refl_op],
                    [rot_op * refl_op, rot_op ** 3 * refl_op]]

    projs = get_symmetry_projectors(char_table, conj_classes, print_results=print_results)

    return projs


def getD5Projectors(rot_op,
                    refl_op,
                    print_results: bool = False):
    """

    :param rot_op:
    :param refl_op:
    :param print_results:
    :return:
    """
    id = sp.eye(refl_op.shape[0], format="csr")

    char_table = np.array([[1,  1,  1, 1],
                           [1, -1,  1, 1],
                           [2,  0,  -(1 + np.sqrt(5))/2,  (-1 + np.sqrt(5))/2],
                           [2,  0,  (-1 + np.sqrt(5))/2, -(1 + np.sqrt(5))/2]])
    conj_classes = [[id],
                    [refl_op, rot_op * refl_op, rot_op**2 * refl_op, rot_op**3 * refl_op, rot_op**4 * refl_op],
                    [rot_op, rot_op ** 4],
                    [rot_op ** 2, rot_op ** 3]]

    projs = get_symmetry_projectors(char_table, conj_classes, print_results=print_results)

    return projs


def getD6Projectors(rot_op,
                    refl_op,
                    print_results: bool = False):
    """

    :param rot_op:
    :param refl_op:
    :param print_results:
    :return:
    """
    id = sp.eye(refl_op.shape[0], format="csr")

    char_table = np.array([[1,  1,  1,  1,  1,  1],
                           [1,  1, -1, -1,  1,  1],
                           [1, -1, -1,  1,  1, -1],
                           [1,  -1,  1, -1,  1, -1],
                           [2,  2,  0,  0, -1, -1],
                           [2, -2,  0,  0, -1,  1]])
    conj_classes = [[id],
                    [rot_op**3],
                    [rot_op * refl_op, rot_op ** 3 * refl_op, rot_op ** 5 * refl_op],
                    [refl_op, rot_op**2 * refl_op, rot_op**4 * refl_op],
                    [rot_op**2, rot_op**4],
                    [rot_op, rot_op**5]]

    projs = get_symmetry_projectors(char_table, conj_classes, print_results=print_results)

    return projs


def getD7Projectors(rot_op,
                    refl_op,
                    print_results: bool = False):
    """

    :param rot_op:
    :param refl_op:
    :param print_results:
    :return:
    """
    id = sp.eye(refl_op.shape[0], format="csr")

    char_table = np.array([[1,  1, 1, 1, 1],
                           [1, -1, 1, 1, 1],
                           [2,  0, 2*np.cos(2*np.pi/7), 2*np.cos(4*np.pi/7), 2*np.cos(6*np.pi/7)],
                           [2,  0, 2*np.cos(4*np.pi/7), 2*np.cos(6*np.pi/7), 2*np.cos(2*np.pi/7)],
                           [2,  0, 2*np.cos(6*np.pi/7), 2*np.cos(2*np.pi/7), 2*np.cos(4*np.pi/7)]])
    conj_classes = [[id],
                    [refl_op, rot_op * refl_op, rot_op**2 * refl_op, rot_op**3 * refl_op,
                     rot_op**4 * refl_op, rot_op**5 * refl_op, rot_op**6 * refl_op],
                    [rot_op, rot_op ** 6],
                    [rot_op**3, rot_op**4],
                    [rot_op**2, rot_op**5]]

    projs = get_symmetry_projectors(char_table, conj_classes, print_results=print_results)

    return projs


def getC4V_and_3byb3(rot_op,
                     refl_op,
                     tx_op,
                     ty_op,
                     print_results: bool = False):
    """
    Symmetry group = semidirect product of :math:`Z3 + Z3` and :math:`C_{4v}`

    .. csv-table:: Irreducible representations
       :header: "Rep", ":math:`C_1`", ":math:`C_2`", ":math:`C_3`", ":math:`C_4`", ":math:`C_5`", ":math:`C_6`", ":math:`C_7`", ":math:`C_8`", ":math:`C_9`"

       "X.1",     "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1"
       "X.2",     "1", "-1",  "1", "-1",  "1",  "1", "-1", "-1",  "1"
       "X.3",     "1", "-1",  "1",  "1", "-1",  "1", "-1",  "1",  "1"
       "X.4",     "1",  "1",  "1", "-1", "-1",  "1",  "1", "-1",  "1"
       "X.5",     "2",  ".", "-2",  ".",  ".",  "2",  ".",  ".",  "2"
       "X.6",     "4", "-2",  ".",  ".",  ".",  "1",  "1",  ".", "-2"
       "X.7",     "4",  ".",  ".", "-2",  ".", "-2",  ".",  "1",  "1"
       "X.8",     "4",  ".",  ".",  "2",  ".", "-2",  ".", "-1",  "1"
       "X.9",     "4 ", "2",  ".",  ".",  ".",  "1", "-1",  ".", "-2"

    SmallGroup(72, 40) in GAP. (S3 x S3) : C2 AKA the Wreath product of S3 and C2
    semi_direct_prod(C3 x C3, D4)
    more info here https://people.maths.bris.ac.uk/~matyd/GroupNames/61/S3wrC2.html
    #1 #6 #9 #6 #18#4 #12#12 #4
    C1 C2 C3 C4 C5 C6 C7 C8 C9

    Conjugacy classes:

    C1 = {e} (order=1, #=1)

    C2 = {Refl, R^2*Refl, Tx*Refl, Tx^2*Refl, Ty*R^2*Refl, Ty^2*R^2*Refl} (order=2, #=6)

    C3 = {R^2, Tx^nTy^mR^2} (order=2, #=9)

    C4 = {R*Refl, R^3*Refl, Tx*Ty^2*R*Refl, Tx^2*Ty*R*Refl, Tx*Ty*R^3*Refl, Tx^2*Ty^2*R^3*Refl} (order=2, #=6)

    C5 = {R, R^3, Tx^n*Ty^m*R, Tx^n*Ty^m*R^3} (order=4, #=18)

    C6 = {Tx, Tx^2, Ty, Ty^2} (order=3, #=4)

    C7 = {Tx^n*Ty^m*Refl except (n,m) = (0,0), (1,0) and (2,0)
          Tx^n*Ty^m*R^2*Refl except (n,m) = (0,0), (0,1), and (0, 2)} (order=6, #=12)

    C8 = {Tx^n*Ty^m*R*Refl except (n,m) = (0,0), (1,2) and (2, 1),
          Tx^n*Ty^m*R^3*Refl except (n,m) = (0,0), (1,1) and (2,2)} (order=6, #=12)

    C9 = {TxTy, Tx^2Ty, TxTy^2 Tx^2Ty^2} (order=3, #=4)

    :param rot_op:
    :param refl_op:
    :param tx_op:
    :param ty_op:
    :param print_results:
    :return:
    """
    #(c2, c7, c6) 0.335493
    #(c2, c7, c9) dim 1856
    #(c2, c8, c6) dim 2064
    #(c2, c8, c9) dim 2064
    #(c4, c7, c6) dim 2064
    #(c4, c7, c9) dim 2064
    #(c4, c8, c6) dim 1856
    #(c4, c8, c9) 0.0904164

    c1 = [sp.eye(refl_op.shape[0], format="csr")]

    c4 = [refl_op, rot_op**2 * refl_op, tx_op * refl_op, tx_op**2 * refl_op,
          ty_op * rot_op**2 * refl_op, ty_op**2 * rot_op**2 * refl_op]
    c2 = [rot_op * refl_op, rot_op ** 3 * refl_op,
          tx_op * ty_op ** 2 * rot_op * refl_op, tx_op ** 2 * ty_op * rot_op * refl_op,
          tx_op * ty_op * rot_op ** 3 * refl_op, tx_op ** 2 * ty_op ** 2 * rot_op ** 3 * refl_op]

    c3 = [rot_op**2, tx_op * rot_op**2, tx_op**2 * rot_op**2,
          ty_op * rot_op**2, ty_op**2 * rot_op**2, tx_op * ty_op * rot_op**2,
          tx_op * ty_op**2 * rot_op**2, tx_op**2 * ty_op * rot_op**2, tx_op**2 * ty_op**2 * rot_op**2]
    c5 = [rot_op, rot_op**3,
          tx_op * rot_op, tx_op**2 * rot_op,
          ty_op * rot_op, ty_op**2 * rot_op,
          tx_op * ty_op * rot_op, tx_op * ty_op**2 * rot_op,
          tx_op ** 2 * ty_op * rot_op, tx_op**2 * ty_op**2 * rot_op,
          tx_op * rot_op**3, tx_op**2 * rot_op**3,
          ty_op * rot_op**3, ty_op**2 * rot_op**3,
          tx_op * ty_op * rot_op**3, tx_op * ty_op**2 * rot_op**3,
          tx_op**2 * ty_op * rot_op**3, tx_op**2 * ty_op**2 * rot_op**3]

    c8 = [ty_op * refl_op, ty_op**2 * refl_op,
          tx_op * ty_op * refl_op, tx_op * ty_op**2 * refl_op,
          tx_op**2 * ty_op * refl_op, tx_op**2 * ty_op**2 * refl_op,
          tx_op * rot_op**2 * refl_op, tx_op**2 * rot_op**2 * refl_op,
          tx_op * ty_op * rot_op**2 * refl_op, tx_op * ty_op**2 * rot_op**2 * refl_op,
          tx_op**2 * ty_op * rot_op**2 * refl_op, tx_op**2 * ty_op**2 * rot_op**2 * refl_op]
    c7 = [tx_op * rot_op * refl_op, tx_op**2 * rot_op * refl_op,
          ty_op * rot_op * refl_op, ty_op**2 * rot_op * refl_op,
          tx_op * ty_op * rot_op * refl_op, tx_op**2 * ty_op**2 * rot_op * refl_op,
          tx_op * rot_op**3 * refl_op, tx_op**2 * rot_op**3 * refl_op,
          ty_op * rot_op**3 * refl_op, ty_op**2 * rot_op**3 * refl_op,
          tx_op * ty_op**2 * rot_op**3 * refl_op, tx_op**2 * ty_op * rot_op**3 * refl_op]

    c9 = [tx_op, tx_op**2, ty_op, ty_op**2]
    c6 = [tx_op * ty_op, tx_op ** 2 * ty_op, tx_op * ty_op ** 2, tx_op ** 2 * ty_op ** 2]

    conj_classes = [c1, c2, c3, c4, c5, c6, c7, c8, c9]

    char_table = np.array([[1,  1,  1,  1,  1,  1,  1,  1,  1],
                           [1, -1,  1, -1,  1,  1, -1, -1,  1],
                           [1, -1,  1,  1, -1,  1, -1,  1,  1],
                           [1,  1,  1, -1, -1,  1,  1, -1,  1],
                           [2,  0, -2,  0,  0,  2,  0,  0,  2],
                           [4, -2,  0,  0,  0,  1,  1,  0, -2],
                           [4,  0,  0, -2,  0, -2,  0,  1,  1],
                           [4,  0,  0,  2,  0, -2,  0, -1,  1],
                           [4,  2,  0,  0,  0,  1, -1,  0, -2]])

    projs = get_symmetry_projectors(char_table, conj_classes, print_results)

    return projs


def getC4V_and_4by4(rot_op,
                    refl_op,
                    tx_op,
                    ty_op,
                    print_results: bool = False):
    """

    :param rot_op:
    :param refl_op:
    :param tx_op:
    :param ty_op:
    :param print_results:
    :return:
    """
    raise NotImplementedError()


def validate_char_table(char_table,
                        conj_classes):
    """
    Verify character table is sensible by checking row/column orthogonality relations

    :param char_table:
    :param conj_classes:
    :return:
    """
    n = char_table.shape[0]
    cc_sizes = np.array([len(cc) for cc in conj_classes])
    order = int(np.sum(cc_sizes))

    valid = True

    # check column orthogonality relation
    col_cross_sums = np.zeros((n, n), dtype=int)
    for ii in range(n):
        for jj in range(n):
            val = np.sum(char_table[:, ii] * char_table[:, jj].conj(), axis=0).real
            col_cross_sums[ii, jj] = np.round(val)

    if not np.array_equal(np.diag(np.array(order / cc_sizes, dtype=int)), col_cross_sums):
        valid = False

    # row orthogonality relation
    row_cross_sums = np.zeros((n, n), dtype=int)
    for ii in range(n):
        for jj in range(n):
            val = np.sum(char_table[ii, :] * char_table[jj, :].conj() * cc_sizes, axis=0).real
            row_cross_sums[ii, jj] = np.round(val)

    if not np.array_equal(order * np.eye(n, dtype=int), row_cross_sums):
        valid = False

    return valid
