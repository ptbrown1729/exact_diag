from __future__ import print_function
import time
import numpy as np
import scipy.sparse as sp

_author = "Peter Brown"
_round_decimals = 14

# #################################################
# coordinate transformation functions
# #################################################

def getRotFn(n_rotations, cx=0, cy=0):
    """
    Returns a function which performs a coordinate rotation about a given origin

    :param n_rotations: Int, number of rotations required to go 360 deg. Rotation angle = 2pi/NumRots
    :return: function
    """
    angle = 2 * np.pi / n_rotations
    # should I do the rounding after computing the values? Or possibly should rely on getTransformedSites.
    # That seems like the right philosophical approach to me...
    rotation_mat = np.round(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), _round_decimals)
    rotation_fn = lambda Xs, Ys: rotation_mat.dot(np.concatenate((Xs[None, :] - cx, Ys[None, :] - cy), 0)) + np.array([[cx], [cy]])

    return rotation_fn

def getReflFn(reflection_vect, cx=0, cy=0):
    """
    Returns a function which performs a coordinate reflection about a given axis.

    # TODO: the reflection center implementation is not correct...

    :param reflection_vect: vector about which to do the reflection
    :return:
    reflection_fn:
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
    reflection_mat = np.round(rotation_mat.transpose().dot(reflection_y_axis.dot(rotation_mat)), _round_decimals)
    reflection_fn = lambda Xs, Ys: reflection_mat.dot(np.concatenate((Xs[None, :] - cx, Ys[None, :] - cy), 0)) + np.array([[cx], [cy]])

    return reflection_fn

def getInversionFn(cx=0, cy=0):
    """
    Returns a function that performs inversion of coordinates about a given center.

    :param cx: x coordinate of inversion center
    :param cy: y coordinate of inversion center
    :return:
    """
    inversion_fn = lambda xs, ys: np.concatenate([ -(xs[None, :] - cx) + cx, -(ys[None, :]- cy) + cy])
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
    transl_fn = lambda xs, ys: np.round(np.concatenate([xs[None, :] + translation_vector[0], ys[None, :] + translation_vector[1]], 0), _round_decimals)
    return transl_fn

# #################################################
# Functions to determine how sites transform
# #################################################

def getTransformedSites(transform_fn, sites, geom_obj):
    """
    Determine how sites are permuted under the action of a given transformation.

     TODO: actually TransSites[ii] transforms into Sites[ii]. I think this is why have to take the transpose in get_xform_op. Should fix this...
    :param transform_fn: A function of the form f(x, y) which returns a 2 x n matrix where each column represents the
    transformed position of site at (x,y).
    :param sites:
    :param geom_obj: instances of geometry class
    :return:
    initial_sites:
    transformed_sites:
    """

    # use periodicity vectors to get reduced locations, to avoid problems where two equivalent locations
    # will not evaluate as equal
    xlocs_red, ylocs_red, _, _ = geom_obj.lattice.reduce_to_unit_cell(geom_obj.xlocs, geom_obj.ylocs, "centered")
    xlocs_red = np.round(xlocs_red, _round_decimals)
    ylocs_red = np.round(ylocs_red, _round_decimals)

    # still need to use actual locations for transform function
    TransCoors = transform_fn(geom_obj.xlocs, geom_obj.ylocs)
    trans_xlocs = TransCoors[0, :]
    trans_ylocs = TransCoors[1, :]
    # also reduce the transformed locations
    trans_xlocs_red, trans_ylocs_red, _, _ = geom_obj.lattice.reduce_to_unit_cell(trans_xlocs, trans_ylocs, "centered")
    trans_xlocs_red = np.round(trans_xlocs_red, _round_decimals)
    trans_ylocs_red = np.round(trans_ylocs_red, _round_decimals)
    #TransCoors = np.concatenate([trans_xcoords_red, trans_ycoords_red], 0)
    #TransCoors = np.round(TransCoors, round_decimals)
    TransSites = np.zeros(len(sites))

    for ii in range(0, len(sites)):
        #Index = np.where((TransCoors[0, :] == Coords[0, ii]) & (TransCoors[1, :] == Coords[1, ii]))[0][0]
        Index = np.where((trans_xlocs_red == xlocs_red[ii]) & (trans_ylocs_red == ylocs_red[ii]))
        if Index[0].shape[0] == 0:
            print('site %d at x = %0.2f, y = %0.2f did not transform to another site' % (ii, xlocs_red[ii], ylocs_red[ii]))
            raise Exception
        TransSites[ii] = sites[Index[0][0]]
    # TODO: add descriptive error when one site doesn't have a partner under transformation.
    return np.array(sites), TransSites

def findSiteCycles(transform_fn, geom_obj):
    """
    Find closed cycles of sites which transform into each other under a given transformation.
    The transformation operator should be unity after NumTransToClose.
    E.g. for pi/2 rotation on a 3x3 lattices we sould have cycles = [[0,2,8,6],[1,5,7,3],[4]]

    :param transform_fn:
    :param geom_obj:
    :return:
    cycles: a list of lists, where each list defines a cycle of sites mapped to each other by successive transformations
    max_cycle_len: length of the longest cycle
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
        _, current_sites = getTransformedSites(transform_fn, trans_sites[:, ii - 1], geom_obj)
        trans_sites = np.concatenate([trans_sites, current_sites[:, None]], 1)
    max_cycle_len = ii
    if max_cycle_len == max_iter:
        raise Exception("Number of cycles required to close transformation greater than maximum allowed iterations")

    # TODO: I think this function returns the cycles backwards.
    # TODO: Actually I think the problem is in getTransformedSites
    cycles = []
    for ii in range(0, len(sites)):
        if sites[ii] not in [x for cycle in cycles for x in cycle]:

            _, indices = np.unique(trans_sites[ii, :], return_index=True)
            indices.sort()
            if indices.size == 0:
                print("sites do not transform correctly according to this symmetry")
                return [], 0
            cycle = np.ndarray.tolist(np.ndarray.flatten(trans_sites[ii, indices]))
            # cycle = np.ndarray.tolist(np.unique(trans_sites[ii,:]))
            cycle = list(map(int, cycle))
            cycles.append(cycle)

    return cycles, max_cycle_len

# #################################################
# Functions to determine how states transform
# #################################################

def getTransReducedProj(projop_full2full, print_results=0):
    """
    Create an operator which projects onto a certain subspace which has definite transformation properties with
    respet to a given transformation operator.

    Typically this function is called on the output of the getCyclicProj function, which produces projop_full2full

    :param projop_full2full: Needs to be a csc sparse matrix. Takes a state in the full space and projects it onto
    a state of a given symmetry. Returns a state in the initial basis. Does not reduce the size of the space.
    projOp_full2full * vector does not necessarily produce a normalized vector.
    :param print_results: Boolean indicating whether or not to print information to the terminal
    :return:
    proj_full2reduced: sparse csr matrix. symm_proj is defined such that symm_proj*Op*symm_proj.transpose() is the operator Op in the
    projected subspace
    """
    if print_results:
        tstart = time.time()

    if not sp.isspmatrix_csc(projop_full2full):
        print("warning, projop_full2full was not csc. Converted it to csc")
        projop_full2full = projop_full2full.tocsc()

    # first, round to avoid any finite precision issues
    # TODO: check if this works _round_decimals value
    projop_full2full.data[np.abs(projop_full2full.data) < 1e-10] = 0
    projop_full2full.eliminate_zeros()
    # next, remove any empty columns
    projop_full2full.indptr = np.unique(projop_full2full.indptr)
    projop_full2full = sp.csc_matrix((projop_full2full.data, projop_full2full.indices, projop_full2full.indptr),
                                     shape=(projop_full2full.shape[0], len(projop_full2full.indptr) - 1))
    # TODO: is it more efficient to do this after normalizing? This function is fast, not limiting the ED speed.

    # next, find unique columns. We only need to check if the first indices are unique to find the unique columns.
    # Stucture of the problem tells us_interspecies that if these are the same then so are the entire columns.
    FirstIndices = projop_full2full.indices[projop_full2full.indptr[0:-1]]
    _, indices = np.unique(FirstIndices, return_index=True)
    proj_full2reduced_unnormed = projop_full2full[:, indices]
    # Finally, normalize each column
    norms = np.sqrt(np.asarray(
        proj_full2reduced_unnormed.multiply(proj_full2reduced_unnormed.conj()).sum(axis=0))).flatten()
    proj_full2reduced = proj_full2reduced_unnormed * sp.diags(np.reciprocal(norms), 0, format="csc")

    if print_results:
        tend = time.time()
        print("Finding projector took %0.2f s" % (tend - tstart))

    return proj_full2reduced.conj().transpose()

def getCyclicProj(xform_op, nxform_to_close, eigval_index=0, print_results=0):
    """
    Get operators which project states onto eigenstates of transop with eigenvalues which are roots of unity.

    :param xform_op: Sparse matrix, operator which implements a symmetry operations on a given state. It is cycle,
    so it should have the property that transop ** NumTransToClose = Identity
    :param nxform_to_close: Int, the order of the cyclic operator. I.e. the smallest integer such that
    transop ** NumTransToClose = Identity
    :param eigval_index: Int, the index of the eigenvalue subspace to project onto. The eigenvalue will be
    exp(1j * 2 * pi * EigValIndex / NumTransToClose)
    :param print_results: Boolean, if True print results to terminal
    :return:
    projop_full2full: Sparse matrix
    """
    if print_results:
        tstart = time.time()

    projop_full2full = sp.eye(xform_op.shape[0], format="csc")
    for ii in range(1, nxform_to_close):
        const = np.round(np.exp(-1j * 2 * np.pi * float(eigval_index) * float(ii) / float(nxform_to_close)),
                         _round_decimals)
        if np.abs(const.imag) < 10 ** (-1 * _round_decimals):
            const = const.real
        projop_full2full = projop_full2full + const * xform_op ** ii

    if print_results:
        tend = time.time()
        print("Finding projector took %0.2f s" % (tend - tstart))
    return projop_full2full

def getZnProjectors(xform_op, n_xforms, print_results=0):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is a cyclic group. Examples of appropriate
    symmetries are translational or pure rotational symmetry.

    Representations of Z_n, the nth cyclic group: There are n irreducible representations, and
    all are one dimensional (since Z_n is abelian). Let zr be the nth root of unity,
    zr = exp(-2*pi*1j/n * r) = z1^r.
    We can regard 2*pi*r/n as the wave-vector (kr) associated with a representation.
    The character table is given by:
           ----------------------------------------------------
    IrrRep |E,    R,    R^2,    ...,    R^l,   ...,    R^(n-1)|
           ----------------------------------------------------
         0 |1,    1,    1  ,    ...,    1  ,   ...,    1      |
         1 |1,   z1,   z1^2,    ...,   z1^l,   ...,   z1^(n-1)|
       ... |                                                  |
         p |1,   zp,    zp^2,   ...,   zp^l,   ...,   zp^(n-1)|
       ... |                                                  |
       n-1 |...                                               |
           ----------------------------------------------------

    :param xform_op: transformation operator acting on state space
    :param n_xforms: minimum number of transformations to return to the starting condition, i.e. xform_op**n = 1
    :param print_results: if 1, print timing information
    :return:
    projs: a list of projectors
    ks: a numpy array of ks associated with each projector
    """

    projs = []
    for ii in range(0, n_xforms):
        projs.append(getTransReducedProj(getCyclicProj(xform_op, n_xforms, ii), print_results))

    # test projector size
    proj_ndims = np.asarray([p.shape[0] for p in projs]).sum()
    if proj_ndims != xform_op.shape[0]:
        print("dimensions of projectors did not match dimensions of operator")
        raise Exception

    ks = 2 * np.pi * np.arange(0, n_xforms) / n_xforms
    return projs, ks

def get2DTranslationProjectors(translation_op1, n_translations1, translation_op2, n_translations2, print_results = 0):
    """

    :param translation_op1:
    :param n_translations1:
    :param translation_op2:
    :param n_translations2:
    :param print_results:
    :return:
    """

    projs1, ks1 = getZnProjectors(translation_op1, n_translations1, print_results)

    proj_ndims = np.asarray([p.shape[0] for p in projs1]).sum()
    if proj_ndims != translation_op1.shape[0]:
        print("Translation 1 projector dimension did not match dimension of full operator")
        raise Exception

    all_projs = []
    for proj, k1 in zip(projs1, ks1):
        # TODO: do I need a conj here?
        sub_projs2, ks2 = getZnProjectors(proj * translation_op2 * proj.conj().transpose(), n_translations2)
        all_projs = all_projs + [sub_proj * proj for sub_proj in sub_projs2]

        # check size at each step
        sub_proj_ndims = np.asarray([p.shape[0] for p in sub_projs2]).sum()
        if sub_proj_ndims != proj.shape[0]:
            print("At translation 1 projector %d, translation 2 sub-projector size did not match translation 1 projector dimension" % k1)
            raise Exception

    # ks1 = [ka, ka, ..., ka, kb, kb,...]
    ks1 = np.repeat(ks1, n_translations2, 0)
    # ks2 = [ka, kb, kc, ..., ka, kb, kc, ...]
    ks2 = np.reshape(np.kron(np.ones([1, n_translations1]), ks2), n_translations1 * n_translations2)

    # remove empty projectors
    a = [[proj, ii] for proj, ii in zip(all_projs, range(0, len(all_projs))) if proj.size > 0]
    all_projs, allowed_indices = zip(*a)
    ks1 = ks1[list(allowed_indices)]
    ks2 = ks2[list(allowed_indices)]

    return all_projs, ks1, ks2

def getCnProjectors(xform_op, n_xforms, print_results=0):
    """
    A wrapper function for getZnProjectors. See that function for documentation.

    :param xform_op:
    :param n_xforms:
    :param print_results:
    :return:
    """

    projs, ks = getZnProjectors(xform_op, n_xforms, print_results)
    return projs

def getC4VProjectors(fourfold_rotation_op, reflection_op, print_results=0):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is C_4V or D_4, the 4th dihedral group.
    This is the symmetry group of the square, generated by a four-fold rotation and a reflection and
    has 8 element, i.e. |C_4V] = 8.

    Irreducible representations of C_4v = D_4: There are five irreducible representations. Four are
    one dimensional and one (E) is two dimensional.
    The character table is given by:
    IrrRep |E, [R,R^3], [R^2], [Refl, R^2*Refl], [R*Refl, R^3*Refl]
           ---------------------------------------------------
        A1 |1,    1   ,   1  ,         1       ,         1
        A2 |1,    1   ,   1  ,         -1      ,         -1
        B1 |1,    -1  ,   1  ,         1       ,         -1
        B2 |1,    -1  ,   1  ,         -1      ,         1
         E |2,     0  ,   -2  ,         0      ,         0
           ---------------------------------------------------
    The labels in the leftmost column label the irreducible representations. Each row is associated with a single
    irreducible representation. The columns are associated with each conjugacy class of the symmetry group. The members
    of each conjugacy class are listed at the top of the column.

    :param fourfold_rotation_op: sparse matrix. fourfold rotation operator, acting on state space.
    :param reflection_op: sparse matrix. reflection operator, acting on state space
    :param print_results:
    :return:
    projs: list of projection operators onto symmetry subspaces
    """

    # Project onto subspaces associated with each irreducible representation of C_4v
    id = sp.eye(reflection_op.shape[0], format="csc")

    proj_A1 = getTransReducedProj(id + fourfold_rotation_op + fourfold_rotation_op ** 3 + fourfold_rotation_op ** 2 + reflection_op +
                                  fourfold_rotation_op ** 2 * reflection_op +
                                  fourfold_rotation_op * reflection_op + fourfold_rotation_op ** 3 * reflection_op,
                                  print_results)

    proj_A2 = getTransReducedProj(id + fourfold_rotation_op + fourfold_rotation_op ** 3 + fourfold_rotation_op ** 2 -
                                  reflection_op - fourfold_rotation_op ** 2 * reflection_op -
                                  fourfold_rotation_op * reflection_op - fourfold_rotation_op ** 3 * reflection_op,
                                  print_results)

    proj_B1 = getTransReducedProj(id - fourfold_rotation_op - fourfold_rotation_op ** 3 + fourfold_rotation_op ** 2 +
                                  reflection_op + fourfold_rotation_op ** 2 * reflection_op - fourfold_rotation_op * reflection_op
                                  - fourfold_rotation_op ** 3 * reflection_op,
                                  print_results)

    proj_B2 = getTransReducedProj(id - fourfold_rotation_op - fourfold_rotation_op ** 3 + fourfold_rotation_op ** 2 -
                                  reflection_op - fourfold_rotation_op ** 2 * reflection_op +
                                  fourfold_rotation_op * reflection_op + fourfold_rotation_op ** 3 * reflection_op,
                                  print_results)

    proj_E = getTransReducedProj(2 * id - 2 * fourfold_rotation_op ** 2, print_results)

    projs = [proj_A1, proj_A2, proj_B1, proj_B2, proj_E]
    return projs

def getD2Projectors(twofold_rotation_op, reflection_op, print_results = 0):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is C_2v = D_2, the 2nd dihedral group.
    This is the symmetry group of the rectangle, generated by a two-fold rotation and a reflection and
    has 4 element, i.e. |D_2] = 4.

    Irreducible representions of D_2 = C_2V: There are four irreducible representations. All are one dimensional.
    The character table is given by:
    IrrRep |E, [R = C_2(z)] ,    [sigma = C_2(y)],    [R*sigma = C_2(x)]
           ------------------------------------------------------------
        A  |1,            1 ,                 1  ,              1
        B1 |1,            1 ,                -1  ,             -1
        B2 |1,           -1 ,                 1  ,             -1
        B3 |1,           -1 ,                -1  ,              1

    The labels in the leftmost column label the irreducible representations. Each row is associated with a single
    irreducible representation. The columns are associated with each conjugacy class of the symmetry group. The members
    of each conjugacy class are listed at the top of the column.
    """

    # Project onto subspaces associated with each irreducible representation of C_4v
    id = sp.eye(reflection_op.shape[0], format="csc")

    proj_A = getTransReducedProj(id + twofold_rotation_op + reflection_op + twofold_rotation_op * reflection_op,
        print_results)

    proj_B1 = getTransReducedProj(id + twofold_rotation_op - reflection_op - twofold_rotation_op * reflection_op,
                                  print_results)

    proj_B2 = getTransReducedProj(id - twofold_rotation_op + reflection_op - twofold_rotation_op * reflection_op,
                                  print_results)

    proj_B3 = getTransReducedProj(id - twofold_rotation_op - reflection_op + twofold_rotation_op * reflection_op,
                                  print_results)

    projs = [proj_A, proj_B1, proj_B2, proj_B3]
    return projs

def getD3Projectors(threefold_rotation_op, reflection_op, print_results = 0):
    """
    Returns operators which project on to symmetry subspaces associated with a given
    symmetry transformation if the symmetry group is C_3v = D_3, the 3rd dihedral group.
    This is the symmetry group of the triangle, generated by a three-fold rotation and a reflection and
    has 6 element, i.e. |D_3] = 4.


    Irreducible representations of D_3: There are three irreducible representations.
    The character table is given by:
    IrrRep |E, [R, R^2] = 2C_3 ,    [sigma, R*sigma, R^2*sigma] = 3C_2
        A1 |1,               1 ,                          1
        A2 |1,               1 ,                         -1
        E  |2,              -1 ,                          0
    The labels in the leftmost column label the irreducible representations. Each row is associated with a single
    irreducible representation. The columns are associated with each conjugacy class of the symmetry group. The members
    of each conjugacy class are listed at the top of the column.

    :param threefold_rotation_op:
    :param reflection_op:
    :param print_results:
    :return:
    projs: list of projection operators on to symmetry subspaces
    """

    # Project onto subspaces associated with each irreducible representation of C_4v
    id = sp.eye(reflection_op.shape[0], format="csc")

    proj_A1 = getTransReducedProj(id +
                                  threefold_rotation_op + threefold_rotation_op ** 2 +
                                  reflection_op + reflection_op * threefold_rotation_op + reflection_op * threefold_rotation_op ** 2,
                                  print_results)

    proj_A2 = getTransReducedProj(id +
                                  threefold_rotation_op + threefold_rotation_op ** 2 +
                                  - reflection_op - reflection_op * threefold_rotation_op - reflection_op * threefold_rotation_op ** 2,
                                  print_results)

    proj_E = getTransReducedProj(2 * id +
                                 - threefold_rotation_op - threefold_rotation_op ** 2,
                                 print_results)

    projs = [proj_A1, proj_A2, proj_E]
    return projs

