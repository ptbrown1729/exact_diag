from __future__ import print_function
import time
import numpy as np
import scipy.special
import scipy.sparse as sp
import ed_geometry as geom
import ed_symmetry as symm
import ed_fermions as hubb
import ed_spins as tvi

# TODO: some ideas
# 1. Identify clusters which are topologically the same, to save on work done for diagonalization
# 2. Automatically identify symmetry group of cluster to reduce work during diagonalization. Expect this is not so
#    important because many clusters will not have much symmetry, especially for larger orders (?)
# 3. Improve speed of various functions. Right now using lots of loops. Not clear to me how to avoid this because
#    I don't know a natural way to order clusters. A few ideas for doing this, which might speed up some functions,
#    for example doing a binary search over a sorted list instead of searching through all clusters of the same order.

def get_clusters_next_order(cluster_list=None, lattice_vect1=np.array([1, 0]), lattice_vect2=np.array([0, 1]), use_symmetry=0):
    """
    Get all clusters that can be generated from a list of clusters with one fewer site.
    TODO: should I be keeping track of the number of symmetric clusters? This is the multiplicity in the thermodynamic
    limit, so probably need this...
    :param cluster_list: List of clusters to used for generating the next clusters
    :param lattice_vect1: Lattice vector 1, giving allowed moves to add sites to our cluster
    :param lattice_vect2:
    :param use_symmetry: Boolean. If using symmetry, will only keep a single cluster representing each symmetry group.
    :return: cluster_list_next
    cluster_list_next: a list of clusters of one higher order.
    """
    cluster_list_next = []
    multiplicity = []
    vect_list = [lattice_vect1, -lattice_vect1, lattice_vect2, -lattice_vect2]

    if cluster_list is None:
        # to zeroth order, cluster of one site
        gm = geom.Geometry.createNonPeriodicGeometry([0], [0])
        cluster_list_next.append(gm)
        multiplicity.append(1)
    else:
        # for each site in cluster add +- each lattice vector. produce a new cluster if we haven't already
        # counted that one
        for c_index, cluster in enumerate(cluster_list):
            # loop over clusters
            coords = zip(cluster.xlocs, cluster.ylocs)
            for (xloc, yloc) in coords:
                # for each cluster, loop over sites
                for vect in vect_list:
                    # for each site, add +/- lattice vectors and check if we have a new cluster
                    xloc_new = xloc + vect[0]
                    yloc_new = yloc + vect[1]
                    if (xloc_new, yloc_new) not in coords:
                        new_xlocs = np.concatenate((cluster.xlocs, np.array([xloc_new])))
                        new_ylocs = np.concatenate((cluster.ylocs, np.array([yloc_new])))
                        new_geom = geom.Geometry.createNonPeriodicGeometry(new_xlocs, new_ylocs)
                        new_geom.permute_sites(new_geom.get_sorting_permutation())

                        new_geom_symmetric_partners = [new_geom]
                        if use_symmetry:
                            new_geom_symmetric_partners = get_clusters_rel_by_symmetry(new_geom)

                        # test if any cluster related to our new cluster by symmetry is already on our list
                        if not [c for c in cluster_list_next if
                                [cn for cn in new_geom_symmetric_partners if cn.isequal_adjacency(c)]]:
                            cluster_list_next.append(new_geom)
                            multiplicity.append(len(new_geom_symmetric_partners))

    return cluster_list_next, multiplicity

def get_all_clusters(max_cluster_order, lattice_vect1=np.array([1, 0]), lattice_vect2=np.array([0, 1]), use_symmetry=1):
    """
    Get all clusters of infinite lattice up to a given order.
    :param max_cluster_order:
    :param lattice_vect1:
    :param lattice_vect2:
    :param use_symmetry: Boolean value. If 1, will use D4 symmetry to reduce the number of clusters
    :return: clusters, multiplicities, order_edge_indices
    order_edge_indices is a list of indices, where the iith entry is the index of the first cluster of order ii.
    multiplicities is a list of the number of times a given cluster can be embedded in an infinite lattice. It is equal
    to the number of distinct clusters produced by point symmetry operations on the cluster
    """

    first_cluster, first_multiplicity = get_clusters_next_order(lattice_vect1=lattice_vect1, lattice_vect2=lattice_vect2, use_symmetry=use_symmetry)

    order_edge_indices = [0, len(first_cluster)]
    cluster_list_list = [first_cluster]
    multiplicity_list_list = [first_multiplicity]

    for ii in range(1, max_cluster_order):
        clust, mult = get_clusters_next_order(cluster_list_list[ii - 1], lattice_vect1, lattice_vect2, use_symmetry)

        order_edge_indices.append(len(clust) + order_edge_indices[ii])
        cluster_list_list.append(clust)
        multiplicity_list_list.append(mult)

    clusters = [h for g in cluster_list_list for h in g]
    multiplicities = np.array([mult for g in multiplicity_list_list for mult in g])

    return clusters, multiplicities, order_edge_indices

def get_all_clusters_with_subclusters(max_cluster_order, lattice_vect1=np.array([1, 0]), lattice_vect2=np.array([0, 1]), use_symmetry=1, print_progress=0):
    """
    Obtain all sub-clusters of the infinite lattice up to a given order, including their multiplicities and sub-clusters
    :param max_cluster_order:
    :param lattice_vect1:
    :param lattice_vect2:
    :param use_symmetry:
    :return: return full_cluster_list, cluster_multiplicities, sub_cluster_mult_mat
    """

    full_cluster_list, cluster_multiplicities, order_indices_full = \
        get_all_clusters(max_cluster_order, lattice_vect1=lattice_vect1, lattice_vect2=lattice_vect2, use_symmetry=use_symmetry)

    # for each of these clusters, we want to identify all subclusters with multiplicity. To that end we define a matrix
    # sc[ii, jj] = m iff C[ii] > C[jj] exactly m times ... i.e. the iith row of this matrix tells you which clusters
    # are contained in cluster C[ii] with what multiplicityn
    cluster_mult_mat = sp.csr_matrix((len(full_cluster_list), len(full_cluster_list)))

    start_index = order_indices_full[-2]
    end_index = order_indices_full[-1]
    # loop over all clusters of the maximum size we are considering. The subcluster information of all smaller clusters
    # will naturally be generated during this process.
    for index in range(start_index, end_index):
        if print_progress:
            print("cluster index = %d" % index)
        cluster = full_cluster_list[index]
        # get all sub clusters of each cluster
        subclusters_list, subcluster_mult_mat, order_indices_subclusters = get_reduced_subclusters(cluster)
        cluster_reduction_mat = map_between_cluster_bases(full_cluster_list, order_indices_full, subclusters_list, order_indices_subclusters, use_symmetry=use_symmetry)

        # now convert subcluster_mult_mat to correct indices...
        final_sub_cluster_mat = cluster_reduction_mat.transpose().dot(subcluster_mult_mat.dot(cluster_reduction_mat))

        # need to avoid double counting certain sites ... get rid of any rows if they have any nonzero elements
        row_sums = np.asarray(np.sum(cluster_mult_mat, 1))
        row_sums = 1 - (row_sums > 0)
        a = row_sums.reshape([len(row_sums), ]).tolist()
        b = sp.diags(a, offsets=0, format='csr')
        cluster_mult_mat = cluster_mult_mat + b.dot(final_sub_cluster_mat)

    return full_cluster_list, cluster_multiplicities, cluster_mult_mat, order_indices_full

def map_between_cluster_bases(cluster_basis_larger, order_indices_larger, cluster_basis_smaller, order_indices_smaller, use_symmetry=1):
    """
    Create a matrix which maps between two different cluster bases.
    :param cluster_basis_larger: list of clusters. This list must contain all of the clusters in cluster_basis_smaller
    :param order_indices_larger:
    :param cluster_basis_smaller: a list of clusters
    :param order_indices_smaller:
    :param use_symmetry:
    :return: basis_change_mat
    """

    # TODO: could I speed this up? Instead of a double sum, comparing all elements, can I assign a single
    # integer to each cluster of a given order and then do an ordered search?
    # now loop over the sub clusters of this order in the full cluster list (jj's) and the sub-clusters of
    # the given cluster (ii)

    # create the matrix that will map the subcluster basis to the full cluster basis
    # cr[ii, jj] = 1 iff sc[ii] <-> c[jj]
    basis_change_mat = sp.csr_matrix((len(cluster_basis_smaller), len(cluster_basis_larger)))

    # so far the sub cluster matrix is given in a basis which only contains subclusters of a given cluster,
    # but we want this matrix in a basis containing all subclusters of the infinite lattice, so we need to
    # go through and find the mapping in this basis
    max_cluster_order = np.min([len(order_indices_larger) - 1, len(order_indices_smaller) - 1])
    # loop over cluster orders
    for order in range(0, max_cluster_order):
        for ii in range(order_indices_smaller[order], order_indices_smaller[order + 1]):
            for jj in range(order_indices_larger[order], order_indices_larger[order + 1]):
                sub_cluster = cluster_basis_smaller[ii]
                symm_cluster_list = [sub_cluster]
                if use_symmetry:
                    symm_cluster_list = get_clusters_rel_by_symmetry(sub_cluster)
                for symm_cluster in symm_cluster_list:
                    if symm_cluster.isequal_adjacency(cluster_basis_larger[jj]):
                        basis_change_mat[ii, jj] = 1

    return basis_change_mat

# functions work on subclusters
def get_subclusters_next_order(parent_geometry, cluster_list=None):
    """
    Given a list of subclusters of some parent geometry, generate all possible connected subclusters with one extra
      site. Also return which of the initial subclusters are contained in the higher order subclusters. This is in the
      form of a list of lists, where each each sublist corresponds to a cluster in cluster_list_next_order. Each sublist
      contains the indices of the clusters in cluster_list that are contained within the cluster in
      cluster_list_next_order. This is useful because in general we only care about subclusters one order lower
      than the cluster we are considering.(???). All lower order clusters will also be subclusters of one of these.
    :param parent_geometry: Geometry to generate subclusters from
    :param cluster_list: A collection of subclusters.
    :return: cluster_list_next_order, old_cluster_contained_in_new_clusters
    cluster_list_next_order, a list of clusters
    old_cluster_contained_in_new_clusters, a list of lists. Each sublist contains the indices of the clusters of lower
    order (i.e. the indices in the list cluster_list) representing clusters contained in the given cluster in
    cluster_list_next_order
    """
    cluster_list_next = []
    old_cluster_contained_in_new_clusters = []
    if cluster_list is None:
        # to zeroth order, add each site as own cluster
        for ii in range(0, parent_geometry.nsites):
            gm = geom.Geometry.createNonPeriodicGeometry(parent_geometry.xlocs[ii], parent_geometry.ylocs[ii])
            cluster_list_next.append(gm)
            old_cluster_contained_in_new_clusters.append([])
    else:
        for c_index, cluster in enumerate(cluster_list):
            # need convenient way to switch between parent cluster and sub-cluster indexing
            parent_coords = zip(parent_geometry.xlocs, parent_geometry.ylocs)
            cluster_coords = zip(cluster.xlocs, cluster.ylocs)
            # loop over sites in our cluster, and try to add additional sites adjacent to them
            for ii, (xloc, yloc) in enumerate(cluster_coords):
                # parent cluster coordinate? Check this
                jj = [aa for aa,coord in enumerate(parent_coords) if coord==(xloc, yloc)]
                jj = jj[0]
                # loop over sites in parent geometry and check if they are adjacent
                for kk in range(0, parent_geometry.nsites):
                    xloc_kk = parent_geometry.xlocs[kk]
                    yloc_kk = parent_geometry.ylocs[kk]
                    # if site is adjacent and not already in our cluster, make a new cluster by adding that site
                    if parent_geometry.adjacency_mat[jj, kk] == 1 and (xloc_kk, yloc_kk) not in cluster_coords:
                        new_xlocs = np.concatenate((cluster.xlocs, np.array([xloc_kk])))
                        new_ylocs = np.concatenate((cluster.ylocs, np.array([yloc_kk])))
                        #TODO: also get adjacency from the previous cluster
                        new_geom = geom.Geometry.createNonPeriodicGeometry(new_xlocs, new_ylocs)
                        new_geom.permute_sites(new_geom.get_sorting_permutation())
                        #TODO: compare this cluster to  make sure a duplicate doesn't already exist? So far only
                        #dealing with real duplicates, not 'duplicates' that are the same shape and hence have the
                        #same hamiltonian
                        duplicates = [(ii, g) for ii, g in enumerate(cluster_list_next) if new_geom == g]
                        if duplicates == []:
                            cluster_list_next.append(new_geom)
                            old_cluster_contained_in_new_clusters.append([c_index])
                        else:
                            new_cluster_index, _ = zip(*duplicates)
                            new_cluster_index = new_cluster_index[0]
                            old_cluster_contained_in_new_clusters[new_cluster_index].append(c_index)

    return cluster_list_next, old_cluster_contained_in_new_clusters

def get_all_subclusters(parent_geometry):
    """
      Find all subclusters containing up to max_order sites of a given parent geometry, and return them as a list of
      lists. Each sublist contains all clusters with a given number of sites. The first sublist contains all clusters
      with one site, the second sublist contains all clusters with two sites, etc. For the purposes of this function,
      we regard clusters with different coordinates as being different clusters. This makes it easier to identify
      the multiplicity of a given subcluster.

      We can also think of these clusters as being numbered by their position in this list
      (if we imagine we have flattened the list of lists into a single
      list). Then for each cluster, sub_cluster_indices contains a list of the indices of all sub clusters of that
      cluster, where the indices are interpreted as in the previous sentence. E.g., we have
      cluster_order_list = [[gm00, gm01, gm02, ...], [gm10, gm11, ...], ...], then gm00 has index 0, gm01 has index 1,...
      sub_cluster_indiex = [[], [], ..., [1, 3], ...]. In this case, we interpret this as gm00 has no subclusters, and
      similarly for gm0n. Then gm10 has two subclusters, one with index 1 which is gm01, and on with index 3 which is
      gm02.
    :param parent_geometry:
    :return: cluster_order_list, sub_cluster_indices, sub_cluster_mat, order_start_indices
    cluster_order_list, a list of lists of clusters
    """
    # TODO sub_cluster_indices is redundant as an output, and should be removed...

    cluster_orders_list = [] # output variable
    sub_cluster_indices = [] # what are these?
    current_order = 0
    total_clusters = 0
    total_clusters_previous = 0

    clusters_next_order, contained_cluster_indices = get_subclusters_next_order(parent_geometry, cluster_list=None)

    # continue generating the next order of clusters until we exceed the maximum order argument or reach the full
    # cluster specified in parent_geometry
    while not clusters_next_order == [] and current_order < parent_geometry.nsites:
        # append new cluster indices
        for jj,_ in enumerate(clusters_next_order):
            curr_indices = []
            if not contained_cluster_indices[jj] == []:
                curr_indices = [ci + total_clusters_previous for ci in contained_cluster_indices[jj]]
                for ii in range(0, len(curr_indices)):
                    ci = curr_indices[ii]
                    if not sub_cluster_indices[ci] == []:
                        curr_indices = curr_indices + sub_cluster_indices[ci]
                curr_indices = sorted(list(set(curr_indices)))
                #print curr_indices
            sub_cluster_indices.append(curr_indices)

        # append new clusters
        cluster_orders_list.append(clusters_next_order)

        # get clusters of the next higher order
        total_clusters_previous = total_clusters
        total_clusters = total_clusters + len(clusters_next_order)
        clusters_next_order, contained_cluster_indices = get_subclusters_next_order(parent_geometry,
                                                                                    cluster_orders_list[current_order])
        current_order = current_order + 1

    # generate sub_cluster_mat ... this way might be ineficient. Possibly nicer if can do it as we go along above
    nclusters = np.sum([len(cl) for cl in cluster_orders_list])
    sub_cluster_mat = sp.csc_matrix((nclusters, nclusters))
    for ii, sc_indices in enumerate(sub_cluster_indices):
        if sc_indices != []:
            sub_cluster_mat[ii, sc_indices] = 1

    # if you want a flattened version of cluster_orders_list, one way to get that is
    # flat = [g for h in cluster_orders_list for g in h]
    # return cluster_orders_list, sub_cluster_indices, sub_cluster_mat
    return cluster_orders_list, sub_cluster_indices, sub_cluster_mat

def reduce_clusters_by_geometry(cluster_orders_list, use_symmetry=1):
    """
    Reduce clusters to those which are geometrically and/or symmetrically distinct. In contrast to get_all_subclusters,
    in this function we regard clusters with different coordinates for their sites as identical, provided their
    adjacency matrices and distancs between sites agree. Checking this properly requires us_interspecies to put our clusters in
    some sort of normal order before comparing them.

    :param cluster_orders_list:
    :return: clusters_geom_distinct, clusters_geom_multiplicity, cluster_reduction_mat

    clusters_geom_distinct is a list of lists. Each sublist contains all the distinct clusters for a given order.

    clusters_geom_multiplicity is a list of lists. Each sublist contains the multiplicities of the corresponding
    cluster in the corresponding sublist of clusters_geom_distinct. TODO: this is now redundant with the addition
    of cluster_reduction_mat.

    cluster_reduction_mat is an n_reduced_clusters x n_full_clusters matrix, where M[ii, jj] = 1 if and only if
    the cluster with index ii in the full list of clusters is geometrically the same as the cluster with index jj
    in the list of reduced clusters. In some sense we can think of this as a basis transformation matrix ...
    """
    # TODO: would like to rewrite thise to use cluster_list and order_start_indices instead of cluster_orders_list
    nclusters = np.sum(np.array([len(cl) for cl in cluster_orders_list]))

    clusters_geom_distinct = []
    clusters_geom_multiplicity = []
    cluster_reduction_mat = sp.csr_matrix((nclusters, nclusters))
    running_full_cluster_total = 0
    running_reduced_cluster_total = 0

    # loop over each order of clusters and accumulate unique clusters and their multiplicity
    for cluster_list in cluster_orders_list:
        clusters_this_order = []
        multiplicity_this_order = []
        # loop over clusters
        for ii,cluster in enumerate(cluster_list):
            cluster_index_full = ii + running_full_cluster_total  # cluster index in full list
            # check if cluster is already in our list

            # list all clusters that are symmetrically equivalent to our cluster. If we are not using symmetries, this
            # list contains only our cluster
            cluster_symm_partners = [cluster]
            if use_symmetry:
                cluster_symm_partners = get_clusters_rel_by_symmetry(cluster)

            duplicates = [(jj, g) for jj, g in enumerate(clusters_this_order) if
                          [h for h in cluster_symm_partners if h.isequal_adjacency(g)]]

            if duplicates == []:
                # if not a duplicate, add this cluster to our list
                clusters_this_order.append(cluster)
                multiplicity_this_order.append(1)
                cluster_index_reduced = running_reduced_cluster_total + len(clusters_this_order) - 1
            else:
                # if is a duplicate, find index of duplicate
                indices_of_duplicate, _ = zip(*duplicates)
                indices_of_duplicate = indices_of_duplicate[0]
                multiplicity_this_order[indices_of_duplicate] = multiplicity_this_order[indices_of_duplicate] + 1
                cluster_index_reduced = indices_of_duplicate + running_reduced_cluster_total
            cluster_reduction_mat[cluster_index_full, cluster_index_reduced] = 1
        # append results to output variables
        clusters_geom_distinct.append(clusters_this_order)
        clusters_geom_multiplicity.append(multiplicity_this_order)
        # increment cluster counters
        running_full_cluster_total = running_full_cluster_total + len(cluster_list)
        running_reduced_cluster_total = running_reduced_cluster_total + len(clusters_this_order)
    # reduce size of cluster_reduction_mat by trimming trailing zeros.
    cluster_reduction_mat = cluster_reduction_mat[:, 0:running_reduced_cluster_total]
    # actually nicer to work with the transpose ... could rewrite the above to constrcut it directly
    cluster_reduction_mat = cluster_reduction_mat.transpose()

    return clusters_geom_distinct, clusters_geom_multiplicity, cluster_reduction_mat

def get_reduced_subclusters(parent_geometry, print_progress=0):
    """
    For a given parent geometry, produce all subclusters and the number of times each subcluster is contained in the
    parent
    :param parent_geometry:
    :return: cluster_list, sub_cluster_mat, order_edge_indices
    cluster_list is a list of distinct sub clusters of the parent_geometry (including the parent geometry itself)

    sub_cluster_mat is a square matrix which gives the number of times cluster j can be embedded in cluster i, if
    cluster j is a proper sub-cluster of i (i.e. the diagonal of this matrix is zero).
    sub_cluster_mat[ii, jj] = # C_j < C_i

    order_edge_indices give the indices in cluster_list where clusters of increasingly larger order appear.
    """

    cluster_orders_list, sub_cluster_indices, sub_cluster_mat = get_all_subclusters(parent_geometry)

    # get unique geometric clusters of each order and multiplicity
    clusters_geom_distinct, clusters_geom_multiplicity, cluster_reduction_mat = reduce_clusters_by_geometry(cluster_orders_list, use_symmetry=1)
    clusters_list = [g for h in clusters_geom_distinct for g in h]

    # indices
    order_edge_indices = [0]
    for ii, cl_list in enumerate(clusters_geom_distinct):
        order_edge_indices.append(len(cl_list) + order_edge_indices[ii])

    # for each cluster, need to know all sub-clusters and multiplicities
    # M[ii, jj] = # of times cluster ii contains cluster jj
    # can do this in several steps.

    # First, let us_interspecies find the multiplicity of C^R_j, the reduced cluster of index j in C^F_i, the full cluster of index i
    # i.e. RF_ij = # C^R_j < C^F_i
    # this is just RF_ij = \sum_k SC[i, k] * C[j, k] , with C = cluster_reduction_mat and SC = sub_cluster_mat
    red_cluster_mult_in_full = sub_cluster_mat.dot(cluster_reduction_mat.transpose())

    # Second, let us_interspecies convert the full clusters in the first index of the above matrix to reduce cluster
    # M_ij = # C^R_j < C^R_i
    # M_ij = \sum_k  C[i ,k] * RF[k, j] is our first guess, but this overcounts because we are converting each reduced
    # cluster to the sum of all full clusters which map onto it. We really only wanted to pick a single instantiation.
    # To account for this, we must divide each row by the number of full clusters that map onto that reduced cluster
    # i.e. we want to normalize the columns of M. This can be done by left multiplying M with a diagonal matrix of the
    # normalization factors
    d_mat = sp.diags(np.ravel(np.divide(1, cluster_reduction_mat.sum(1))), format='csc')
    reduced_sub_cluster_mat = d_mat.dot(cluster_reduction_mat.dot(red_cluster_mult_in_full))

    return clusters_list, reduced_sub_cluster_mat, order_edge_indices

def get_clusters_rel_by_symmetry(cluster, symmetry='d4'):
    """
    Return all distinct clusters related to the initial cluster by symmetry. Currently, only D4 symmetry is implented.
    TODO: allow for other symmetries
    :param cluster: geometry object representing a cluster
    :param symmetry: TODO implement
    :return: cluster_symm_partners
    cluster_symm_partners is a list of geometry objects, including the initial cluster.
    """
    rot_fn = symm.getRotFn(4)
    refl_fn = symm.getReflFn([0, 1])

    cluster_symm_partners = [cluster]
    # list coordinates of all D4 symmetries
    xys_list = []
    xys_list.append(rot_fn(cluster.xlocs, cluster.ylocs))
    xys_list.append(rot_fn(xys_list[0][0, :], xys_list[0][1, :]))
    xys_list.append(rot_fn(xys_list[1][0, :], xys_list[1][1, :]))
    xys_list.append(refl_fn(cluster.xlocs, cluster.ylocs))
    xys_list.append(refl_fn(xys_list[0][0, :], xys_list[0][1, :]))
    xys_list.append(refl_fn(xys_list[1][0, :], xys_list[1][1, :]))
    xys_list.append(refl_fn(xys_list[2][0, :], xys_list[2][1, :]))

    # add distinct clusters to a list
    for xys in xys_list:
        c = geom.Geometry.createNonPeriodicGeometry(xys[0, :], xys[1, :])
        c.permute_sites(c.get_sorting_permutation())
        if not [h for h in cluster_symm_partners if h.isequal_adjacency(c)]:
            cluster_symm_partners.append(c)

    return cluster_symm_partners

# nlce functions

def get_nlce_exp_val(exp_vals_clusters, sub_cluster_multiplicity_mat,
                     parent_clust_multiplicity_vect, order_start_indices, nsites):
    """
    Compute linked cluster expansion weights and expectation values from expectation values on individual clusters
    :param exp_vals_clusters:
    :param sub_cluster_multiplicity_mat:
    :param nsites:
    :param include_full_cluster:
    :return: expectation_vals, cluster_weights
    """
    # TODO: make this function flexible enough to handle NCLE for infinite cluster or finite cluster
    # TODO: ensure parent_clust_multiplicity_vect is a row vector
    # TODO: this function goes awry if our sub_cluster_multiplicty_mat has extra clusters which we don't want to use
    # TODO: also return different orders of nlce expansion
    # e.g., if it contains all subclusters of a parent cluster, but we only diagonalized and want to work with up to
    # order 10 of them
    if sp.issparse(parent_clust_multiplicity_vect):
        parent_clust_multiplicity_vect = parent_clust_multiplicity_vect.toarray()
    if isinstance(parent_clust_multiplicity_vect, list):
        parent_clust_multiplicity_vect = np.array(parent_clust_multiplicity_vect)
    parent_clust_multiplicity_vect = np.reshape(parent_clust_multiplicity_vect, (1, parent_clust_multiplicity_vect.size))


    nclusters = exp_vals_clusters.shape[0]
    # want to accept arbitrary exp_vals shapes, subject only to the condition that the first dimension loops over clusters
    weights = np.zeros(exp_vals_clusters.shape)

    # compute weights in a nice tensorial way
    # W_p(c) = P(c) - \sum_{s<c} W_p(s)
    # W_exp[ii]   = P[ii] - \sum_j sc[ii, jj] * W_exp[jj]
    # sc_ij = # of times C^R_j < C^R_i
    for ii in range(0, nclusters):
        # in the 2d case, we can writep
        # weights[ii, ...] = exp_vals_clusters[ii, ...] - sub_cluster_multiplicity_mat[ii, :].dot(weights)
        # in the general case, we need to sum over index 1 of sub_cluster_multiplicty_mat, and index 0 of weights.
        # np.dot no longer works, as this sums over the last index of the first array, and the second to last index of
        # the second array. To sum over arbitrary axes, use np.tensordot
        weights[ii, ...] = exp_vals_clusters[ii, ...] - np.squeeze(
            np.tensordot(sub_cluster_multiplicity_mat[ii, :].toarray(), weights, axes=(1, 0)))

    #exp_val_nlce[jj] = sub_cluster_multiplicity_mat[-1, :] * weights[:, jj] / nsites
    # exp_val_nlce = np.squeeze(np.tensordot(sub_cluster_multiplicity_mat[-1, :].toarray(), weights, axes=(1,0))) / nsites
    exp_val_nlce = np.squeeze(np.tensordot(parent_clust_multiplicity_vect, weights, axes=(1, 0))) / nsites

    b = list(exp_vals_clusters[0, ...].shape)
    size = [len(order_start_indices) - 1] + [int(c) for c in b]
    nlce_orders = np.zeros(tuple(size))
    for ii in range(0, len(order_start_indices) - 1):
        nlce_orders[ii, ...] = np.squeeze(np.tensordot(parent_clust_multiplicity_vect[0, order_start_indices[ii]:order_start_indices[ii+1]][None, :],
                                                 weights[order_start_indices[ii]:order_start_indices[ii+1],...], axes=(1, 0))) / nsites

    return exp_val_nlce, nlce_orders, weights

def euler_resum(exp_vals_orders, y):
    # TODO: test
    euler_orders = np.zeros(exp_vals_orders.shape)

    for ii in range(0, euler_orders.shape[0]):
        jjs = np.arange(0, ii + 1)
        yjjs = np.power(y, jjs + 1)
        exp_vals_partial_orders = exp_vals_orders[0 : ii + 1, ...]
        binomial_coeffs = np.array([scipy.special.binom(ii, jj) for jj in jjs])
        partial_sum = np.tensordot(binomial_coeffs * yjjs, exp_vals_partial_orders, axes=(0, 0))
        euler_orders[ii, ...] = 1. / (1 + y) ** (ii + 1) * partial_sum

    euler_resum = np.sum(euler_orders, 0)

    return euler_resum, euler_orders

def wynn_resum(exp_vals_orders):
    pass

if __name__ == "__main__":
    import time
    import datetime
    import numpy as np
    import cPickle as pickle
    import ed_geometry as geom
    from ed_nlce import *

    ########################################
    # general settings
    ########################################

    max_cluster_order = 9
    do_thermodynamic_limit = 1
    max_cluster_order_thermo_limit = 7
    diag_full_cluster = 00
    display_results = 1
    temps = np.logspace(-1, 0.5, 50)

    if display_results:
        import matplotlib.pyplot as plt


    ########################################
    # create geometry
    ########################################
    # xlocs = [1, 1, 2, 2, 3, 3, 4, 4]
    # ylocs = [1, 2, 2, 3, 3, 4, 4, 5]
    # parent_geometry = geom.Geometry.createNonPeriodicGeometry(xlocs, ylocs)
    parent_geometry = geom.Geometry.createSquareGeometry(4, 4, 0, 0, 1, 1)
    parent_geometry.permute_sites(parent_geometry.get_sorting_permutation())
    clusters_list, sub_cluster_mult, order_start_indices = get_reduced_subclusters(parent_geometry)

    jx = 0.5
    jy = 0.5
    jz = 0.5
    hx = 0.0
    hy = 0.0
    hz = 0.0

    ########################################
    # diagonalization of each cluster and computation of properties for each cluster
    ########################################
    # initialize variables which will store expectation values
    energies = np.zeros((len(clusters_list), len(temps)))
    entropies = np.zeros((len(clusters_list), len(temps)))
    specific_heats = np.zeros((len(clusters_list), len(temps)))

    for ii, cluster in enumerate(clusters_list):
        if cluster.nsites > max_cluster_order:
            continue
        print("%d/%d" % (ii + 1, len(clusters_list)))
        model = tvi.spinSystem(cluster, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)
        H = model.createH(print_results=1)
        eig_vals, eig_vects = model.diagH(H, print_results=1)

        t_start = time.clock()
        for jj, T in enumerate(temps):
            # calculate properties for each temperature
            energies[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, T, 0)
            Z = np.sum(np.exp(-eig_vals / T))
            entropies[ii, jj] = (np.log(Z) + energies[ii, jj] / T)
            specific_heats[ii, jj] = 1. / T ** 2 * \
                                     (
                                             model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, T, 0) - energies[ii, jj] ** 2)
        t_end = time.clock()
        print("Computing %d finite temperature expectation values took %0.2fs" % (len(temps), t_end - t_start))

    # nlce computation
    parent_cluster_mult_vector = sub_cluster_mult[-1, :]
    if max_cluster_order == parent_geometry.nsites:
        parent_cluster_mult_vector[0, -1] = 1

    energy_nlce, orders_energy, weight_energy = \
        get_nlce_exp_val(energies[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         parent_cluster_mult_vector[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], parent_geometry.nsites)
    entropy_nlce, order_entropy, weight_entropy = \
        get_nlce_exp_val(entropies[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         parent_cluster_mult_vector[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], parent_geometry.nsites)
    specific_heat_nlce, orders_specific_heat, weight_specific_heat = \
        get_nlce_exp_val(specific_heats[0:order_start_indices[max_cluster_order], :],
                         sub_cluster_mult[0:order_start_indices[max_cluster_order],
                         0:order_start_indices[max_cluster_order]],
                         parent_cluster_mult_vector[0, 0:order_start_indices[max_cluster_order]],
                         order_start_indices[0:max_cluster_order + 1], parent_geometry.nsites)

    ########################################
    # diagonalize full cluster
    ########################################
    if diag_full_cluster:
        model = tvi.spinSystem(parent_geometry, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)

        cx, cy = model.geometry.get_center_of_mass()
        # rot-fn
        # TODO: not correclty detecting failure to close under rotations
        rot_fn = symm.getRotFn(4, cx=cx, cy=cy)
        rot_cycles, max_cycle_len_rot = symm.findSiteCycles(rot_fn, model.geometry)
        rot_op = model.get_xform_op(rot_cycles)
        # reflection about y-axis
        refl_fn = symm.getReflFn(np.array([0, 1]), cx=cx, cy=cy)
        refl_cycles, max_cycle_len_ref = symm.findSiteCycles(refl_fn, model.geometry)
        refl_op = model.get_xform_op(refl_cycles)
        # symmetry projectors
        # symm_projs = symm.getC4VProjectors(rot_op, refl_op)
        symm_projs, _ = symm.getZnProjectors(rot_op, 4, print_results=1)

        # TODO calculate energy eigs for each...
        eig_vals_sectors = []
        energy_exp_sectors = np.zeros((len(symm_projs), len(temps)))
        energy_sqr_exp_sectors = np.zeros((len(symm_projs), len(temps)))
        for ii, proj in enumerate(symm_projs):
            H = model.createH(projector=proj, print_results=1)
            eig_vals, eig_vects = model.diagH(H, print_results=1)
            eig_vals_sectors.append(eig_vals)
            for jj in range(0, len(temps)):
                energy_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, temps[jj], print_results=0)
                energy_sqr_exp_sectors[ii, jj] = model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, temps[jj],
                                                                            print_results=0)
        eigs_all = np.sort(np.concatenate(eig_vals_sectors))

        energies_full = np.zeros(len(temps))
        entropies_full = np.zeros(len(temps))
        specific_heat_full = np.zeros(len(temps))
        for jj, temp in enumerate(temps):
            energies_full[jj] = model.thermal_avg_combine_sectors(energy_exp_sectors[:, jj], eig_vals_sectors,
                                                                  temp) / model.geometry.nsites
            Z = np.sum(np.exp(- eigs_all / temp))
            # for entropy calculation, need full energy, so must multiply energy by number of sites again
            entropies_full[jj] = 1. / model.geometry.nsites * (
                np.log(Z) + energies_full[jj] * model.geometry.nsites / temp)
            # for specific heat, need full energy instead of energy per site
            ham_sqr = model.thermal_avg_combine_sectors(energy_sqr_exp_sectors[:, jj], eig_vals_sectors, temp)
            specific_heat_full[jj] = 1. / (temp ** 2 * model.geometry.nsites) * (
                ham_sqr - (energies_full[jj] * model.geometry.nsites) ** 2)

    ########################################
    # generate all clusters up to certain order on the infinite lattice
    ########################################

    if do_thermodynamic_limit:
        # clusters_list, sub_cluster_mult, order_start_indices = get_reduced_subclusters(parent_geometry)
        # TODO: want cluster order indices from this function also
        clusters_list_tl, cluster_multiplicities_tl, sub_cluster_mult_tl = \
            get_all_clusters_with_subclusters(max_cluster_order_thermo_limit)

        # initialize variables which will store expectation values
        energies_tl = np.zeros((len(clusters_list_tl), len(temps)))
        entropies_tl = np.zeros((len(clusters_list_tl), len(temps)))
        specific_heats_tl = np.zeros((len(clusters_list_tl), len(temps)))

        for ii, cluster in enumerate(clusters_list_tl):
            print("%d/%d" % (ii + 1, len(clusters_list_tl)))
            model = tvi.spinSystem(cluster, jx, jy, jz, hx, hy, hz, use_ryd_detunes=0)
            H = model.createH(print_results=1)
            eig_vals, eig_vects = model.diagH(H, print_results=1)

            t_start = time.clock()
            for jj, T in enumerate(temps):
                # calculate properties for each temperature
                energies_tl[ii, jj] = model.get_exp_vals_thermal(eig_vects, H, eig_vals, T, 0)
                Z_tl = np.sum(np.exp(-eig_vals / T))
                entropies_tl[ii, jj] = (np.log(Z_tl) + energies_tl[ii, jj] / T)
                specific_heats_tl[ii, jj] = 1. / T ** 2 * \
                                            (model.get_exp_vals_thermal(eig_vects, H.dot(H), eig_vals, T, 0) - energies_tl[
                                                ii, jj] ** 2)
            t_end = time.clock()
            print("Computing %d finite temperature expectation values took %0.2fs" % (len(temps), t_end - t_start))

        # nlce computation
        energy_nlce_tl, weight_energy_tl = \
            get_nlce_exp_val(energies_tl, sub_cluster_mult_tl, cluster_multiplicities_tl, 1)
        entropy_nlce_tl, weight_entropy_tl = \
            get_nlce_exp_val(entropies_tl, sub_cluster_mult_tl, cluster_multiplicities_tl, 1)
        specific_heat_nlce_tl, weight_specific_heat_tl = \
            get_nlce_exp_val(specific_heats_tl, sub_cluster_mult_tl, cluster_multiplicities_tl, 1)

    ########################################
    # plot results
    ########################################
    if display_results:
        fig_handle = plt.figure()
        nrows = 2
        ncols = 2

        plt.subplot(nrows, ncols, 1)
        plt.semilogx(temps, energies[-1, :] / parent_geometry.nsites)
        plt.semilogx(temps, energy_nlce)
        for ii in range(6, orders_energy.shape[0]):
            plt.semilogx(temps, np.sum(orders_energy[0:ii, ...], 0))

        if diag_full_cluster:
            plt.semilogx(temps, energies_full)
        if do_thermodynamic_limit:
            plt.semilogx(temps, energy_nlce_tl)

        plt.grid()
        plt.xlabel('Temperature (J)')
        plt.ylabel('Energy/site (J)')
        plt.title('Energy/site')
        plt.ylim([-2, 0.25])
        plt.legend(['last cluster', 'finite nlce', 'full cluster',
                    'thermo limit nlce order = %d' % max_cluster_order_thermo_limit])

        plt.subplot(nrows, ncols, 2)
        plt.semilogx(temps, entropies[-1, :] / parent_geometry.nsites)
        plt.semilogx(temps, entropy_nlce)
        for ii in range(6, orders_energy.shape[0]):
            plt.semilogx(temps, np.sum(order_entropy[0:ii, ...], 0))
        if diag_full_cluster:
            plt.semilogx(temps, entropies_full)
        if do_thermodynamic_limit:
            plt.semilogx(temps, entropy_nlce_tl)
        plt.grid()
        plt.xlabel('Temperature (J)')
        plt.ylabel('Entropy/site ()')
        plt.ylim([0, 2])
        # plt.title('Entropy/site')

        plt.subplot(nrows, ncols, 3)
        plt.semilogx(temps, specific_heats[-1, :] / parent_geometry.nsites)
        plt.semilogx(temps, specific_heat_nlce)
        for ii in range(6, orders_energy.shape[0]):
            plt.semilogx(temps, np.sum(orders_specific_heat[0:ii, ...], 0))
        if diag_full_cluster:
            plt.semilogx(temps, specific_heat_full)
        if do_thermodynamic_limit:
            plt.semilogx(temps, specific_heat_nlce_tl)
        plt.grid()
        plt.xlabel('Temperature (J)')
        plt.ylabel('Specific heat/site ()')
        plt.ylim([0, 2])
        # plt.title('Specific heat / site')

    ########################################
    # save data
    ########################################

    data = [clusters_list, sub_cluster_mult, order_start_indices, temps,
            energies, weight_energy, entropies, weight_entropy, specific_heats, weight_specific_heat,
            energy_nlce, entropy_nlce, specific_heat_nlce]
    if diag_full_cluster:
        data.append([energies_full, entropies_full, specific_heat_full])
    if do_thermodynamic_limit:
        data.append([energies_tl, energy_nlce_tl, weight_energy_tl,
                     entropies_tl, entropy_nlce_tl, weight_entropy_tl,
                     specific_heats_tl, specific_heat_nlce_tl, weight_specific_heat_tl])

    today_str = datetime.datetime.today().strftime('%Y-%m-%d_%H;%M;%S')

    if display_results:
        fig_name = "nlce_results" + today_str + ".png"
        fig_handle.savefig(fig_name)
        plt.show()

    fname = "nlce_test_" + today_str + "_pickle.dat"
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    # read data example
    # with open(fname, 'rb') as f:
    #     f_loaded = pickle.load(f)
