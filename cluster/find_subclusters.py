import sys
import cPickle as pickle
import scipy.sparse as sp
import numpy as np
import time

import ed_spins as tvi
import ed_fermions as hubb
import ed_nlce

def find_subclusters(cluster_index, cluster_data_out_fname, cluster_data_in_fname):
    with open(cluster_data_in_fname, 'rb') as f:
        data = pickle.load(f)

    order_indices_full = data[2]
    full_cluster_list = data[1]
    cluster = full_cluster_list[cluster_index]

    cluster_mult_mat = sp.csr_matrix((len(full_cluster_list), len(full_cluster_list)))

    # get all sub clusters of each cluster
    subclusters_list, subcluster_mult_mat, order_indices_subclusters = ed_nlce.get_reduced_subclusters(cluster)
    cluster_reduction_mat = ed_nlce.map_between_cluster_bases(full_cluster_list, order_indices_full, subclusters_list,
                                                              order_indices_subclusters, use_symmetry=1)

    # now convert subcluster_mult_mat to correct indices...
    final_sub_cluster_mat = cluster_reduction_mat.transpose().dot(subcluster_mult_mat.dot(cluster_reduction_mat))

    data_out = [cluster_index, final_sub_cluster_mat, cluster]
    with open(cluster_data_out_fname, 'wb') as f:
        pickle.dump(data_out, f)

if __name__ == "__main__":
    # bash arrays start at 1, so lets keep that convention for terminal arguments
    cluster_index = int(sys.argv[1]) - 1
    out_fname = sys.argv[2]
    in_fname = sys.argv[3]
    find_subclusters(cluster_index, out_fname, in_fname)